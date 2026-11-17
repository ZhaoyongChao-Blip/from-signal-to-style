# -*- coding: utf-8 -*-
"""
t2_explore.py —— 全市场日线探索 + 标签可行性分析（含涨停污染诊断）

产出: output/explore.md
"""
import os, sys, glob
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
D_DAILY = os.path.join(ROOT, "data", "raw", "daily")
D_IDX = os.path.join(ROOT, "data", "raw", "index")
OUT = os.path.join(ROOT, "output")
os.makedirs(OUT, exist_ok=True)


def load_all(limit=None):
    files = sorted(glob.glob(os.path.join(D_DAILY, "*.parquet")))
    if limit:
        files = files[:limit]
    dfs = []
    for f in files:
        try:
            d = pd.read_parquet(f)
            if len(d) > 100:
                dfs.append(d)
        except Exception:
            pass
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()


def main(limit=None):
    px = load_all(limit)
    lines = []
    def P(*a):
        s = " ".join(str(x) for x in a)
        print(s)
        lines.append(s)

    P("# 探索报告\n")
    n_stock = px["code"].nunique()
    P(f"- 股票数: **{n_stock}**")
    P(f"- 记录数: **{len(px):,}**")
    P(f"- 日期范围: {px['date'].min().date()} ~ {px['date'].max().date()}")

    # 板块分布
    pfx = px["code"].str[:3]
    bucket = np.select(
        [pfx.isin(["600", "601", "603", "605"]), pfx.isin(["000", "001", "002", "003"]),
         pfx.str.startswith("30"), pfx.str.startswith("68"), pfx.str.startswith("8") | pfx.str.startswith("4") | pfx.str.startswith("92")],
        ["沪主板", "深主板", "创业板", "科创板", "北交所"], default="其他")
    px["board"] = bucket
    P("\n## 板块分布（按成交日）\n")
    for k, v in px["board"].value_counts().items():
        P(f"- {k}: {v:,} ({v/len(px):.1%})")

    # 涨跌幅分布
    P("\n## 日涨跌幅分布\n")
    q = px["pct_chg"].describe(percentiles=[.01, .05, .25, .5, .75, .95, .99])
    for k, v in q.items():
        P(f"- {k}: {v:.3f}")

    # 涨停识别（按板块）
    lim = np.where(px["board"].isin(["创业板", "科创板"]), 0.20, 0.10)
    px["lim_pct"] = lim * 100
    px["is_zt"] = px["pct_chg"] >= px["lim_pct"] - 0.6   # 留 0.6pp 容差
    P("\n## 涨停诊断（关键）\n")
    P(f"- 全样本涨停占比: **{px['is_zt'].mean():.2%}**")
    zt = px[px["is_zt"]]["pct_chg"]
    nzt = px[~px["is_zt"]]["pct_chg"]
    P(f"- 涨停样本均值涨跌幅: {zt.mean():.3f}%  (n={len(zt):,})")
    P(f"- 非涨停样本均值涨跌幅: {nzt.mean():.3f}%  (n={len(nzt):,})")
    P(f"- 非涨停中位数: {nzt.median():.3f}%")
    P(f"- 非涨停胜率(涨): {(nzt>0).mean():.2%}")
    P("\n> 对比 MEMORY.md 里 aibingchuan 的结论：A 股 31% 成交发生在涨停价，+5.48%，其余 +0.83%，中位数 −0.36%。")
    P("> 若这里的「非涨停中位数」也接近 0 或为负，说明**任何 5 日超额收益标签都必须先处理涨停**，否则模型学的是涨停。\n")

    # 未来 5 日收益（含与不含涨停）
    px = px.sort_values(["code", "date"])
    px["fwd5"] = px.groupby("code")["close"].shift(-5) / px["close"] - 1
    px["fwd5_nzt"] = px["fwd5"]
    # 「剔除未来5日内含涨停的样本」
    zt_fwd = (px.groupby("code")["is_zt"].shift(-1).fillna(False).astype(bool)
              | px.groupby("code")["is_zt"].shift(-2).fillna(False).astype(bool)
              | px.groupby("code")["is_zt"].shift(-3).fillna(False).astype(bool)
              | px.groupby("code")["is_zt"].shift(-4).fillna(False).astype(bool)
              | px.groupby("code")["is_zt"].shift(-5).fillna(False).astype(bool))
    px["fwd5_has_zt"] = zt_fwd

    valid = px.dropna(subset=["fwd5"])
    P("## 未来 5 日收益分布\n")
    P(f"- 全部样本: 均值 {valid['fwd5'].mean():.2%}, 中位数 {valid['fwd5'].median():.2%}, 胜率 {(valid['fwd5']>0).mean():.1%}")
    a = valid[~valid["fwd5_has_zt"]]
    b = valid[valid["fwd5_has_zt"]]
    P(f"- 未来5日**无涨停**样本: 均值 {a['fwd5'].mean():.2%}, 中位数 {a['fwd5'].median():.2%}, 胜率 {(a['fwd5']>0).mean():.1%} (n={len(a):,}, {len(a)/len(valid):.1%})")
    P(f"- 未来5日**含涨停**样本: 均值 {b['fwd5'].mean():.2%}, 中位数 {b['fwd5'].median():.2%}, 胜率 {(b['fwd5']>0).mean():.1%} (n={len(b):,}, {len(b)/len(valid):.1%})")

    # 基准
    P("\n## 基准指数\n")
    for c in ["000300", "000905", "000852"]:
        p = os.path.join(D_IDX, f"{c}.parquet")
        if os.path.exists(p):
            d = pd.read_parquet(p).sort_values("date")
            d["fwd5"] = d["close"].shift(-5) / d["close"] - 1
            P(f"- {c}: 未来5日均值 {d['fwd5'].mean():.2%}, 中位数 {d['fwd5'].median():.2%}")

    txt = "\n".join(lines)
    with open(os.path.join(OUT, "explore.md"), "w", encoding="utf-8") as f:
        f.write(txt)
    print(f"\n[saved] {os.path.join(OUT,'explore.md')}")


if __name__ == "__main__":
    lim = int(sys.argv[1]) if len(sys.argv) > 1 else None
    main(lim)
