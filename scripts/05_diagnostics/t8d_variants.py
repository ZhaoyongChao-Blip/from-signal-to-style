# -*- coding: utf-8 -*-
"""t8d_variants.py —— 诚实评估：在「不能做空、不能买涨停、不能超容量」的约束下，
y_neut 模型的 alpha 到底值多少？

已知条件：
  · t7f：控制 4 个风格格子后，picks 每 5 天 +0.713%（t=3.47，8 个季度全正）→ **alpha 真实存在**
  · t8c：真实价判成交后，原样 Top30 多头年化 11.68%、相对等权 −9.43%
  · t8：同格子对冲年化 −3.81%（构造有问题，不再采用）

本脚本穷举「可执行」的构造，看哪一个能把 t7f 的 alpha 兑现：
  V1  TopK 网格（K=20/30/50/100），持有 5/10/20 日，真实价判成交
  V2  只买「格子内分数最高」的股票，每格 1 只（真正的中性选股）
  V3  剔除「模型分数高但成交额极小」的股票（容量约束），再取 TopK：amount >= 1亿 / 3亿
  V4  对 y_neut 分数做「格子内 demean」：score' = score − mean(score | cell)，
      这样排序不再被格子本身带偏（A 股可执行的近似中性化）

产出：output/m8d_variants.md
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


def perf(nav, bench=None):
    r = nav.pct_change().dropna()
    if len(r) < 20:
        return {}
    ann = (1 + r).prod() ** (252 / len(r)) - 1
    vol = r.std() * np.sqrt(252)
    out = {"ann": ann, "vol": vol, "sharpe": (ann - 0.015) / vol,
           "mdd": (nav / nav.cummax() - 1).min(), "n": len(r)}
    if bench is not None:
        br = bench.pct_change().reindex(r.index).fillna(0)
        bann = (1 + br).prod() ** (252 / len(r)) - 1
        tr = r - br
        out["excess"] = ann - bann
        out["ir"] = tr.mean() / tr.std() * np.sqrt(252)
    return out


def main():
    import lightgbm as lgb
    import t4_lgb
    panel = pd.read_parquet(os.path.join(ROOT, "data", "proc", "panel.parquet"))
    panel["date"] = pd.to_datetime(panel["date"])
    fc = t4_lgb.feat_cols(panel)
    bt = panel[(panel["date"] >= "2024-01-01") & (panel["date"] <= "2025-11-10")].copy()
    bt = bt.dropna(subset=["adj_close", "adj_open", "raw_close", "raw_open"] + STYLE)
    bt["lim"] = np.where(bt["board"].isin(["GEM", "STAR"]), 0.20, 0.10)
    for lb in ("y_exc", "y_neut"):
        p = os.path.join(ROOT, "models", f"lgb_{lb}.txt")
        if os.path.exists(p):
            bt[f"s_{lb}"] = lgb.Booster(model_file=p).predict(bt[fc])
    bt = bt.sort_values(["date", "code"]).reset_index(drop=True)

    def cell_id(g):
        c = np.zeros(len(g), dtype=np.int64)
        for i, cname in enumerate(STYLE):
            q = pd.qcut(g[cname].rank(method="first"), 5, labels=False, duplicates="drop")
            c = c + q.fillna(2).to_numpy().astype(np.int64) * (5 ** i)
        return pd.Series(c, index=g.index)

    bt["cell"] = bt.groupby("date", group_keys=False)[STYLE + ["code"]].apply(cell_id)
    # V4：格子内 demean
    for lb in ("y_neut", "y_exc"):
        c = f"s_{lb}"
        if c in bt.columns:
            bt[f"{c}_dn"] = bt[c] - bt.groupby(["date", "cell"])[c].transform("mean")

    dates = np.array(sorted(bt["date"].unique()))
    by = {d: g.set_index("code") for d, g in bt.groupby("date")}

    def run(score_col, K=30, hold=5, cost=0.0015, min_amt=2e7, mode="top", per_cell=1):
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
            g2 = by[nxt]
            cand = g.dropna(subset=[score_col, "adj_close"])
            cand = cand[cand["amount"] > min_amt]
            if len(cand) < K:
                continue
            if mode == "top":
                picks = cand.nlargest(K, score_col).index.tolist()
            elif mode == "cell_top":
                # 每格取分数最高的 per_cell 只，再按分数取前 K 只
                sub = cand.sort_values(score_col, ascending=False).groupby("cell").head(per_cell)
                picks = sub.nlargest(K, score_col).index.tolist()
            else:
                raise ValueError(mode)
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
            if not buy:
                continue
            new = {c: 1.0 / len(buy) for c in buy}
            for c in buy:
                hold_px[c] = g2.at[c, "adj_open"]
            old = set(weights)
            to = 1.0 - len(old & set(new)) / max(len(new), 1) if old else 1.0
            cash *= (1 - cost * to * 2)
            weights = new
        return pd.Series(dict(out)).sort_index()

    d = bt[["date", "code", "adj_close"]].sort_values(["code", "date"])
    d["r"] = d.groupby("code")["adj_close"].pct_change()
    ew = (1 + d.groupby("date")["r"].mean().fillna(0)).cumprod()
    P("# 可执行构造穷举（真实价判成交，复权价算收益）\n")
    P(f"- 区间 {bt['date'].min().date()} ~ {bt['date'].max().date()}，全市场等权年化 "
      f"{perf(ew)['ann']:.2%} Sharpe {perf(ew)['sharpe']:.3f}\n")
    P("| 变体 | 信号 | K | 持有 | 最小成交额 | 年化 | Sharpe | 相对等权 | IR |")
    P("|---|---|---|---|---|---|---|---|---|")

    tests = [
        ("V1a", "s_y_neut", 30, 5, 2e7, "top"),
        ("V1b", "s_y_neut", 20, 5, 2e7, "top"),
        ("V1c", "s_y_neut", 50, 5, 2e7, "top"),
        ("V1d", "s_y_neut", 30, 10, 2e7, "top"),
        ("V2", "s_y_neut_dn", 30, 5, 2e7, "cell_top"),
        ("V4", "s_y_neut_dn", 30, 5, 2e7, "top"),
        ("V3a", "s_y_neut", 30, 5, 1e8, "top"),
        ("V3b", "s_y_neut", 30, 5, 3e8, "top"),
    ]
    for tag, col, K, H, ma, mode in tests:
        if col not in bt.columns:
            continue
        nav = run(col, K=K, hold=H, min_amt=ma, mode=mode)
        p = perf(nav, ew)
        if not p:
            continue
        P(f"| {tag} | {col} | {K} | {H} | {ma/1e8:.0f}亿 | {p['ann']:.2%} | {p['sharpe']:.3f} | "
          f"{p['excess']:+.2%} | {p['ir']:+.2f} |")
        nav.to_frame("nav").to_csv(os.path.join(OUT, f"m8d_nav_{tag}.csv"))

    with open(os.path.join(OUT, "m8d_variants.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    print(f"[saved] {os.path.join(OUT,'m8d_variants.md')}")


if __name__ == "__main__":
    main()
