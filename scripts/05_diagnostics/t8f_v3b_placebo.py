# -*- coding: utf-8 -*-
"""t8f_v3b_placebo.py —— V3b（成交额≥3亿）到底是不是模型在起作用？

可疑点（t8e 已查明）：
  · 「3 亿以上」那层全样本 5 日均值只有 +0.150%，中位 **−0.469%**，是三档里最差的
  · 但 V3b 组合年化 21.17%、Sharpe 0.955，追平全市场等权
  ⟹ 组合一定在某处加了东西。三个候选来源：
      ①模型在「大票池」里选股（真 alpha）
      ②组合其实没买满 30 只 / 权重集中（构造偏差）
      ③这一段大票恰好涨（beta 巧合）

检验方法：
  A. **同池 placebo**：在同一个 amount≥3亿 的池子里随机选 30 只，跑 100 次，得到零分布
     → 若 V3b 落在零分布里，说明它没有比「在这个池子里随便买」更好
  B. **池基准**：amount≥3亿 池的等权日收益（不再选股），这才是正确的对照
  C. 记录每次调仓实际买入只数

产出：output/m8f_v3b.md
"""
import os, sys
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
OUT = os.path.join(ROOT, "output")
L = []


def P(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    L.append(s)


def perf(nav):
    r = nav.pct_change().dropna()
    ann = (1 + r).prod() ** (252 / len(r)) - 1
    vol = r.std() * np.sqrt(252)
    return {"ann": ann, "vol": vol, "sharpe": (ann - 0.015) / vol,
            "mdd": (nav / nav.cummax() - 1).min()}


def main():
    import lightgbm as lgb
    import t4_lgb
    panel = pd.read_parquet(os.path.join(ROOT, "data", "proc", "panel.parquet"))
    panel["date"] = pd.to_datetime(panel["date"])
    fc = t4_lgb.feat_cols(panel)
    bt = panel[(panel["date"] >= "2024-01-01") & (panel["date"] <= "2025-11-10")].copy()
    bt = bt.dropna(subset=["adj_close", "adj_open", "raw_close", "raw_open"])
    bt["lim"] = np.where(bt["board"].isin(["GEM", "STAR"]), 0.20, 0.10)
    p = os.path.join(ROOT, "models", "lgb_y_neut.txt")
    bt["s"] = lgb.Booster(model_file=p).predict(bt[fc])
    bt = bt.sort_values(["date", "code"]).reset_index(drop=True)

    dates = np.array(sorted(bt["date"].unique()))
    by = {d: g.set_index("code") for d, g in bt.groupby("date")}

    def run(seed=None, K=30, hold=5, cost=0.0015, min_amt=3e8):
        cash = 1.0
        hold_px, weights = {}, {}
        out, nb, pickinfo = [], [], []
        rng = np.random.default_rng(seed) if seed is not None else None
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
                p_ = g["adj_close"].get(c, np.nan)
                if np.isfinite(p_) and p_ > 0:
                    hold_px[c] = p_
            if i % hold != 0 or i == len(dates) - 1:
                continue
            nxt = dates[i + 1]
            g2 = by[nxt]
            cand = g.dropna(subset=["s" if seed is None else "s", "adj_close"])
            cand = cand[cand["amount"] > min_amt]
            if len(cand) < K:
                continue
            if seed is None:
                picks = cand.nlargest(K, "s").index.tolist()
            else:
                picks = cand.index[rng.random(len(cand)).argsort()[:K]].tolist()
            buy = []
            for c in picks:
                if c not in g2.index:
                    continue
                ro = g2.at[c, "raw_open"]
                rcp = g.at[c, "raw_close"]
                lm = g2.at[c, "lim"]
                if not (np.isfinite(ro) and np.isfinite(rcp) and rcp > 0):
                    continue
                if ro >= rcp * (1 + lm - 0.005):
                    continue
                buy.append(c)
            nb.append(len(buy))
            if not buy:
                continue
            new = {c: 1.0 / len(buy) for c in buy}
            for c in buy:
                hold_px[c] = g2.at[c, "adj_open"]
            old = set(weights)
            to = 1.0 - len(old & set(new)) / max(len(new), 1) if old else 1.0
            cash *= (1 - cost * to * 2)
            weights = new
        nav = pd.Series(dict(out)).sort_index()
        return nav, np.array(nb)

    # 池基准：amount>=3亿 每日等权
    P("# V3b 拆解：模型 vs 池子 vs 随机\n")
    pool_r = []
    for dt in dates:
        g = by[dt]
        pool = g[g["amount"] > 3e8]
        pool_r.append((dt, pool["adj_close"].pct_change().mean() if len(pool) > 1 else 0.0))
    # 用复权收益构造池净值
    dd = bt[["date", "code", "adj_close", "amount"]].sort_values(["code", "date"])
    dd["r"] = dd.groupby("code")["adj_close"].pct_change()
    po = dd[dd["amount"] > 3e8]
    pool_nav = (1 + po.groupby("date")["r"].mean().fillna(0)).cumprod()
    all_nav = (1 + dd.groupby("date")["r"].mean().fillna(0)).cumprod()

    nav_v3b, nb = run()
    P("| 组合 | 年化 | 波动 | Sharpe | 最大回撤 |")
    P("|---|---|---|---|---|")
    for nm, nv in (("全市场等权", all_nav), ("amount≥3亿 池等权", pool_nav), ("V3b（模型选 Top30）", nav_v3b)):
        pf = perf(nv)
        P(f"| {nm} | {pf['ann']:.2%} | {pf['vol']:.2%} | {pf['sharpe']:.3f} | {pf['mdd']:.1%} |")
    P(f"\n- V3b 每次调仓实际买入只数：中位 {np.median(nb):.0f}，最少 {nb.min()}，均值 {nb.mean():.1f}")

    P("\n## A. 同池 placebo（在 amount≥3亿 池里随机选 30 只，100 次）\n")
    res = []
    for k in range(100):
        nv, _ = run(seed=9000 + k)
        res.append(perf(nv))
        if (k + 1) % 25 == 0:
            print(f"  placebo {k+1}/100", flush=True)
    D = pd.DataFrame(res)
    real = perf(nav_v3b)
    P("| 指标 | 零分布均值 | 零分布标准差 | V3b 实际 | V3b 百分位 |")
    P("|---|---|---|---|---|")
    for c in ("ann", "sharpe"):
        P(f"| {c} | {D[c].mean():.4f} | {D[c].std():.4f} | {real[c]:.4f} | {(D[c] < real[c]).mean():.1%} |")
    D.to_csv(os.path.join(OUT, "m8f_placebo.csv"), index=False)

    P("\n> 读法：若 V3b 在「同池随机」的零分布中只排到 50~70 百分位，")
    P("> 说明 V3b 的 21% 年化主要是「选中了 is 3 亿以上这个池子」，模型贡献很小。")

    with open(os.path.join(OUT, "m8f_v3b.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    print(f"[saved] {os.path.join(OUT,'m8f_v3b.md')}")


if __name__ == "__main__":
    main()
