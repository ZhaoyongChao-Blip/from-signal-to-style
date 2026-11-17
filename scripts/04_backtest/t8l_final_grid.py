# -*- coding: utf-8 -*-
"""t8l_final_grid.py —— 最终网格：持有期 × 池下限 × 成本，用已验证口径

口径（与 t8d/t8j 完全一致，已交叉验证）：
  · dt 日收盘后筛池（amount > min_amt）→ t+1 日开盘买入（跳过开盘涨停）
  · 用复权价算收益、原始价判成交
  · 每 hold 天换仓一次，逐日收益累乘
  · 基准：全市场等权（逐日 adj_close 收益的算术平均）

输出一张完整网格，并标出「在 15bp 成本下相对等权为正」的格子。

产出：output/m8l_final_grid.md
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
    if len(r) < 20:
        return {}
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
    bt = bt.dropna(subset=["adj_close", "adj_open", "raw_close", "raw_open", "amount"])
    bt["lim"] = np.where(bt["board"].isin(["GEM", "STAR"]), 0.20, 0.10)
    for lb in ("y_exc", "y_neut"):
        fp = os.path.join(ROOT, "models", f"lgb_{lb}.txt")
        if os.path.exists(fp):
            bt[f"s_{lb}"] = lgb.Booster(model_file=fp).predict(bt[fc])
    bt = bt.sort_values(["date", "code"]).reset_index(drop=True)
    dates = np.array(sorted(bt["date"].unique()))
    by = {d: g.set_index("code") for d, g in bt.groupby("date")}

    dd = bt[["date", "code", "adj_close"]].sort_values(["code", "date"])
    dd["r"] = dd.groupby("code")["adj_close"].pct_change()
    ew = (1 + dd.groupby("date")["r"].mean().fillna(0)).cumprod()
    ew_r = ew.pct_change()

    def run(col, min_amt, hold, cost=0.0015, K=30):
        cash = 1.0
        hold_px, weights = {}, {}
        out = []
        for i, dt in enumerate(dates):
            g = by[dt]
            if weights:
                s = 0.0
                for c, w in weights.items():
                    p0 = hold_px.get(c, np.nan)
                    p1 = g["adj_close"].get(c, np.nan)
                    if np.isfinite(p0) and np.isfinite(p1) and p0 > 0:
                        s += w * (p1 / p0 - 1)
                cash *= (1 + s)
            out.append((dt, cash))
            for c in weights:
                p_ = g["adj_close"].get(c, np.nan)
                if np.isfinite(p_) and p_ > 0:
                    hold_px[c] = p_
            if i % hold != 0 or i == len(dates) - 1:
                continue
            nxt = dates[i + 1]
            g2 = by[nxt]
            cand = g.dropna(subset=[col, "adj_close"])
            cand = cand[cand["amount"] > min_amt]
            if len(cand) < K:
                continue
            picks = cand.nlargest(K, col)
            buy = []
            for c in picks.index:
                if c not in g2.index:
                    continue
                ro, rcp, lm = g2.at[c, "raw_open"], g.at[c, "raw_close"], g2.at[c, "lim"]
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

    P("# 最终网格：持有期 × 池下限 × 成本\n")
    pe = perf(ew)
    P(f"- 基准（全市场等权）：年化 {pe['ann']:.2%}，Sharpe {pe['sharpe']:.3f}，"
      f"波动 {pe['vol']:.2%}，最大回撤 {pe['mdd']:.1%}")
    P(f"- 区间 2024-01-02 ~ {bt['date'].max().date()}，{bt['date'].nunique()} 个交易日\n")

    for col in ("s_y_neut", "s_y_exc"):
        if col not in bt.columns:
            continue
        P(f"## {col}\n")
        P("| 池下限 | 持有 | 成本 | 年化 | 波动 | Sharpe | 相对等权 |")
        P("|---|---|---|---|---|---|---|")
        for ma, nm in ((3e8, "3亿"), (1e8, "1亿"), (2e7, "2千万")):
            for H in (5, 10, 15, 20):
                for c in (0.0010, 0.0015):
                    nv = run(col, ma, H, c)
                    pf = perf(nv)
                    if not pf:
                        continue
                    rr = nv.pct_change().fillna(0)
                    rb = ew_r.reindex(nv.index).fillna(0)
                    exc = (1 + (rr - rb)).prod() ** (252 / len(rr)) - 1
                    mark = " ✅" if exc > 0 else ""
                    P(f"| {nm} | {H}日 | {c*1e4:.0f}bp | {pf['ann']:.2%} | {pf['vol']:.2%} | "
                      f"{pf['sharpe']:.3f} | {exc:+.2%}{mark} |")
        P("")

    with open(os.path.join(OUT, "m8l_final_grid.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    print(f"[saved] {os.path.join(OUT,'m8l_final_grid.md')}")


if __name__ == "__main__":
    main()
