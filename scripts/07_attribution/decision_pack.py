# -*- coding: utf-8 -*-
"""decision_pack.py —— 把待决策的三件事量化成数字，供用户直接判断

决策 1：Task 2 TimesNet 该不该开
决策 2：基准换不换（全市场等权 -> 中证1000 + beta 中性）
决策 3：要不要追查「小盘不崩时段也能跑赢」
"""
import os
import pandas as pd, numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "output")

RF = 0.015


def perf(r):
    ann = (1 + r).prod() ** (252 / len(r)) - 1
    vol = r.std() * np.sqrt(252)
    return ann, vol, (ann - RF) / vol


def load_nav(f):
    return pd.read_csv(os.path.join(OUT, f), index_col=0, parse_dates=True).iloc[:, 0].dropna()


# ---------- 公共基准 ----------
p = pd.read_parquet(os.path.join(ROOT, "data", "proc", "panel.parquet"),
                    columns=["date", "code", "adj_close"])
p["date"] = pd.to_datetime(p["date"])
d = p[(p["date"] >= "2024-01-01") & (p["date"] <= "2025-11-10")].dropna(subset=["adj_close"])
d = d.sort_values(["code", "date"]).assign(r=d.groupby("code")["adj_close"].pct_change())
ew_r = d.groupby("date")["r"].mean().fillna(0)
ew_r.index = pd.to_datetime(ew_r.index)

ma5 = load_nav("m9_nav_R1_1.0.csv").pct_change().fillna(0)
v3b = load_nav("m8d_nav_V3b.csv").pct_change().fillna(0)

print("=" * 74)
print("决策 1：Task 2 TimesNet 该不该开？")
print("=" * 74)
print("""
已有的上界（跨截面 46 因子 + LightGBM，2018-22 训练 / 2023 验证 / 2024-25 测试）：
  · test AUC            0.5525   （随机 = 0.5000）
  · test IC（自身标签）  0.0699
  · 无重叠 t (n=89)      6.33
  · 同风格格子内超额     +0.713% / 5 日
  · 但净值相对等权       −9.43%（真实价口径） / +0.06%（3亿池 / 15bp，年化差）

TimesNet 单股时序要成立，必须同时满足：
  (a) 单股时序信噪比 > 48 个横截面特征联合的信噪比  —— 目前 AUC 只有 0.5525，说明截面上
      已接近「可预测上界」，时序模型的边际信息有限
  (b) 必须超越的基准是「同风格格子内 +0.713%/5日」，不是 0
  (c) CPU-only（本机无 CUDA），单股训练成本远高于 LGBM

结论：**不建议开**。唯一合理的改法是把目标从「跑赢等权」改成
      「在给定候选池里提升胜率」，那时 0.713%/5 日 是可复用基准。
""")

print("=" * 74)
print("决策 2：基准换不换？")
print("=" * 74)

# 用已有的指数收益构造中证1000基准（panel 里有 idx_000852_fwd5，但那是 fwd5）
try:
    idx = pd.read_parquet(os.path.join(ROOT, "data", "proc", "panel.parquet"),
                          columns=["date", "code", "idx_000852_fwd5", "idx_000905_fwd5",
                                   "idx_000300_fwd5"])
    idx["date"] = pd.to_datetime(idx["date"])
    # idx_*_fwd5 是未来 5 日指数收益，取其 1/5 无法直接还原日收益；改用日频文件
    print("  (panel 里只有 fwd5 口径的指数收益，日频需从 data/raw/index 重新读)")
except Exception as e:
    print("  ", e)

idxdir = os.path.join(ROOT, "data", "raw", "index")
if os.path.isdir(idxdir):
    print(f"  data/raw/index 下文件: {sorted(os.listdir(idxdir))[:8]}")

for code, name in (("000852", "中证1000"), ("000905", "中证500"), ("000300", "沪深300")):
    for fn in (f"{code}.csv", f"sh{code}.csv", f"index_{code}.csv"):
        fp = os.path.join(idxdir, fn)
        if os.path.exists(fp):
            df = pd.read_csv(fp)
            print(f"  {name} 可读: {fn}  cols={list(df.columns)[:8]}  rows={len(df)}")
            break

print(f"""
当前主基准：全市场等权（3317 只，2024-01-02~2025-11-10）
  年化 {perf(ew_r)[0]*100:.2f}%   波动 {perf(ew_r)[1]*100:.2f}%   Sharpe {perf(ew_r)[2]:.3f}

若换成中证1000 + beta 中性：
  · 中证1000 在 2024H1 同样大跌，换基准后「小盘崩盘时它不在小盘」这个超额来源会被计入 alpha
  · 但论文的核心论点恰恰是「这个超额是风格规避，不是选股」
  · 换成 1000 + beta 中性 = 把风格规避合法化成 alpha，说服力下降

结论：**不换**。可折中：加一个附录表，同时报告两套基准，让审稿人自己看。
""")

print("=" * 74)
print("决策 3：小盘不崩时段，策略能不能跑赢？")
print("=" * 74)

# 定义「小盘不崩」= 等权日收益 > 0 的日子，或用季度层面：等权季度收益 > 0
q_port = (1 + ma5).groupby([ma5.index.year, ma5.index.quarter]).prod() - 1
q_bench = (1 + ew_r).groupby([ew_r.index.year, ew_r.index.quarter]).prod() - 1
q_exc = q_port - q_bench
df = pd.DataFrame({"port": q_port, "bench": q_bench, "exc": q_exc})
df.index = [f"{y}Q{q}" for y, q in df.index]
print(df.round(4).to_string())

up = df[df["bench"] > 0]
down = df[df["bench"] <= 0]
print(f"\n基准上涨的 {len(up)} 个季度：平均超额 {up['exc'].mean()*100:+.2f}pp，"
      f"胜 {int((up['exc']>0).sum())}/{len(up)}")
print(f"基准下跌的 {len(down)} 个季度：平均超额 {down['exc'].mean()*100:+.2f}pp，"
      f"胜 {int((down['exc']>0).sum())}/{len(down)}")

# t 检验：上涨季度超额是否显著异于 0
from math import sqrt
if len(up) > 1:
    m, s, n = up["exc"].mean(), up["exc"].std(ddof=1), len(up)
    t = m / (s / sqrt(n)) if s > 0 else np.nan
    print(f"\n上涨季度超额的 t 值 = {t:.2f}（n={n}）")

print("""
判据：若「基准上涨季度」的平均超额 > 0 且 t > 2，说明策略在小盘不崩时也有超额，
      结论可从「纯风格规避」升级为「风格规避 + 弱选股」。
      若 t 不显著或为负，则确认论文现有结论（超额全部来自崩盘期），不必再追。
""")
