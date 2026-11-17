# -*- coding: utf-8 -*-
"""t7e_placebo.py —— lo 十倍 placebo：同一套回测，把「模型分数」换成随机数

为什么要做：t7d 的风格匹配对照组（年化 27.87%、Sharpe 0.941）比模型本身（18.37%、0.634）还高，
这不合常理 —— 要么是「风格匹配」这个构造本身自带收益（那就是构造偏差），要么是巧合。
**只能靠零分布判断。** 所以把 picks 的分数全部换成随机数，跑同样的 matched 流程 100 次。

三组对照，同一批随机种子：
  A. 原样回测（Top30 / 5 日）
  B. 风格匹配对照组（找风格最接近的 30 只）
  C. 纯随机（真随机选 30 只，不做匹配）
如果 B 显著高于 A 和 C，说明「找风格匹配对照组」这个方法本身有正向偏差，不能用它做基准。

产出：output/m7e_placebo.md
"""
import os, sys
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
OUT = os.path.join(ROOT, "output")
STYLE = ["log_amt", "log_price", "vol_20", "turn_ma_20"]
L = []


def P(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    L.append(s)


def perf(nav):
    r = nav.pct_change().dropna()
    if len(r) < 20:
        return {}
    ann = (1 + r).prod() ** (252 / len(r)) - 1
    vol = r.std() * np.sqrt(252)
    return {"ann": ann, "vol": vol, "sharpe": (ann - 0.015) / vol,
            "mdd": (nav / nav.cummax() - 1).min()}


def main():
    panel = pd.read_parquet(os.path.join(ROOT, "data", "proc", "panel.parquet"),
                            columns=["date", "code", "board", "adj_open", "adj_close",
                                     "amount", "raw_close"] + STYLE)
    panel["date"] = pd.to_datetime(panel["date"])
    bt = panel[(panel["date"] >= "2024-01-01") & (panel["date"] <= "2025-11-10")].copy()
    bt = bt.dropna(subset=["adj_close", "adj_open"])
    dates = np.array(sorted(bt["date"].unique()))
    by = {d: g.set_index("code") for d, g in bt.groupby("date")}
    P(f"# 风格匹配基准的 placebo\n\n- 区间 2024-01-01 ~ {pd.Timestamp(dates[-1]).date()}，{len(dates)} 个交易日\n")

    def run(seed, mode, K=30, hold=5):
        rng = np.random.default_rng(seed)
        cash = 1.0
        hold_px, weights = {}, {}
        out = []
        for i, dt in enumerate(dates):
            g = by[dt]
            if weights:
                rets = {}
                for c, w in weights.items():
                    p0 = hold_px.get(c, np.nan)
                    p1 = g["adj_close"].get(c, np.nan)
                    rets[c] = (p1 / p0 - 1) if (np.isfinite(p0) and np.isfinite(p1) and p0 > 0) else 0.0
                cash *= (1 + sum(w * rets[c] for c, w in weights.items()))
            out.append((dt, cash))
            for c in weights:
                p = g["adj_close"].get(c, np.nan)
                if np.isfinite(p) and p > 0:
                    hold_px[c] = p
            if i % hold != 0 or i == len(dates) - 1:
                continue
            nxt = dates[i + 1]
            cand = g.dropna(subset=["adj_close"] + STYLE)
            cand = cand[cand["amount"] > 2e7]
            if len(cand) < 2 * K:
                continue
            cand = cand.copy()
            cand["_s"] = rng.random(len(cand))
            picks = cand.nlargest(K, "_s")
            if mode == "plain" or mode == "matched":
                target = picks
            if mode == "matched":
                mu = picks[STYLE].mean().to_numpy()
                rest = cand.drop(index=picks.index)
                sd = rest[STYLE].std().to_numpy() + 1e-9
                dist = np.sqrt((((rest[STYLE].to_numpy() - mu) / sd) ** 2).sum(axis=1))
                target = rest.iloc[np.argsort(dist)[:K]]
            g2 = by[nxt]
            new = {}
            for c in target.index:
                p = g2["adj_open"].get(c, np.nan)
                if np.isfinite(p) and p > 0:
                    new[c] = 1.0 / len(target)
                    hold_px[c] = p
            weights = new
        return pd.Series(dict(out)).sort_index()

    res = {"plain": [], "matched": []}
    for k in range(100):
        for mode in ("plain", "matched"):
            nv = run(5000 + k, mode)
            p = perf(nv)
            if p:
                res[mode].append(p)
        if (k + 1) % 20 == 0:
            print(f"  placebo {k+1}/100", flush=True)

    P("## 随机打分的零分布（100 次）\n")
    P("| 流程 | 年化 均值 | 年化 标准差 | Sharpe 均值 | Sharpe 标准差 | Sharpe 中位 | Sharpe 95% 分位 |")
    P("|---|---|---|---|---|---|---|")
    for mode, nm in (("plain", "A 原样回测"), ("matched", "B 风格匹配对照组")):
        d = pd.DataFrame(res[mode])
        P(f"| {nm} | {d['ann'].mean():.2%} | {d['ann'].std():.2%} | {d['sharpe'].mean():.3f} | "
          f"{d['sharpe'].std():.3f} | {d['sharpe'].median():.3f} | {d['sharpe'].quantile(.95):.3f} |")
        d.to_csv(os.path.join(OUT, f"m7e_placebo_{mode}.csv"), index=False)

    # 真实模型的百分位
    P("\n## 真实模型落在零分布的哪个位置\n")
    real = {"plain": {"y_exc": (0.0436, 0.102), "y_neut": (0.1837, 0.634)},
            "matched": {"y_exc": (0.2437, 0.790), "y_neut": (0.2787, 0.941)}}
    P("| 流程 | 模型 | 年化 | Sharpe | 年化百分位 | Sharpe 百分位 |")
    P("|---|---|---|---|---|---|")
    for mode, nm in (("plain", "A 原样"), ("matched", "B 匹配")):
        d = pd.DataFrame(res[mode])
        for lb, (ann, shp) in real[mode].items():
            pa = (d["ann"] < ann).mean()
            ps = (d["sharpe"] < shp).mean()
            P(f"| {nm} | LGB_{lb} | {ann:.2%} | {shp:.3f} | {pa:.1%} | {ps:.1%} |")

    P("\n> 读法：若「B 匹配」的随机零分布均值明显高于 A，说明匹配构造自带正偏差，")
    P("> 那么 t7d 里「匹配对照组 27.87% > 模型 18.37%」就不是模型失败，而是基准选错了。")

    with open(os.path.join(OUT, "m7e_placebo.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    print(f"[saved] {os.path.join(OUT,'m7e_placebo.md')}")


if __name__ == "__main__":
    main()
