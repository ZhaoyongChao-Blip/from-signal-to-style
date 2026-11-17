# -*- coding: utf-8 -*-
"""t8c_pricecheck.py —— 按「真实实现价」重算回测

t8 暴露的问题：t5/t8 的回测用**后复权价** adj_open 建仓、adj_close 估值，
而涨停不可买要用**原始价**判断。两者混在一起时，跨除权日会出现假收益。

本脚本：
  1. 用原始价（raw_open / raw_close）判断能否成交，用复权价算收益（这是唯一正确做法）
  2. 对每个调仓日的 picks 统计「开盘涨停买不进」的比例
  3. 重算 P1（y_neut 原样多头）与 P1'（y_exc），与 t7d/t8 的数对比
  4. 把「买不进」的比例单列出来 —— 这是策略能在多大程度上被执行的直接指标

产出：output/m8c_pricecheck.md
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
    ann = (1 + r).prod() ** (252 / len(r)) - 1
    vol = r.std() * np.sqrt(252)
    out = {"ann": ann, "vol": vol, "sharpe": (ann - 0.015) / vol,
           "mdd": (nav / nav.cummax() - 1).min(), "n": len(r)}
    if bench is not None:
        br = bench.pct_change().reindex(r.index).fillna(0)
        bann = (1 + br).prod() ** (252 / len(r)) - 1
        tr = r - br
        out["excess_ann"] = ann - bann
        out["ir"] = tr.mean() / tr.std() * np.sqrt(252)
    return out


def main():
    import lightgbm as lgb
    import t4_lgb
    panel = pd.read_parquet(os.path.join(ROOT, "data", "proc", "panel.parquet"))
    panel["date"] = pd.to_datetime(panel["date"])
    fc = t4_lgb.feat_cols(panel)
    bt = panel[(panel["date"] >= "2024-01-01") & (panel["date"] <= "2025-11-10")].copy()
    bt = bt.dropna(subset=["adj_close", "adj_open", "raw_close", "raw_open"])
    bt["lim"] = np.where(bt["board"].isin(["GEM", "STAR"]), 0.20, 0.10)
    for lb in ("y_exc", "y_neut"):
        p = os.path.join(ROOT, "models", f"lgb_{lb}.txt")
        if os.path.exists(p):
            bt[f"s_{lb}"] = lgb.Booster(model_file=p).predict(bt[fc])
    bt = bt.sort_values(["date", "code"]).reset_index(drop=True)

    dates = np.array(sorted(bt["date"].unique()))
    by = {d: g.set_index("code") for d, g in bt.groupby("date")}

    def run(score_col, K=30, hold=5, cost=0.0015):
        cash = 1.0
        hold_px = {}
        weights = {}
        out = []
        skip_stat = []
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
            cand = cand[cand["amount"] > 2e7]
            if len(cand) < K:
                continue
            picks = cand.nlargest(K, score_col)

            # 用**原始价**判断能否成交：nxt 日开盘价 vs dt 日收盘价 × (1+limit−0.5%)
            buy = []
            for c in picks.index:
                if c not in g2.index:
                    continue
                ro = g2.at[c, "raw_open"]
                rc_prev = g.at[c, "raw_close"]
                lm = g2.at[c, "lim"]
                if not (np.isfinite(ro) and np.isfinite(rc_prev) and rc_prev > 0):
                    continue
                if ro >= rc_prev * (1 + lm - 0.005):
                    continue   # 开盘涨停，买不进
                buy.append(c)
            skip_stat.append((K - len(buy)) / K)
            if not buy:
                continue
            new = {c: 1.0 / len(buy) for c in buy}
            for c in buy:
                hold_px[c] = g2.at[c, "adj_open"]      # 收益用复权价算
            old = set(weights)
            to = 1.0 - len(old & set(new)) / max(len(new), 1) if old else 1.0
            cash *= (1 - cost * to * 2)
            weights = new
        return pd.Series(dict(out)).sort_index(), np.array(skip_stat)

    d = bt[["date", "code", "adj_close"]].sort_values(["code", "date"])
    d["r"] = d.groupby("code")["adj_close"].pct_change()
    ew = (1 + d.groupby("date")["r"].mean().fillna(0)).cumprod()

    P("# 用真实价判成交 + 复权价算收益\n")
    P("| 组合 | 年化 | 波动 | Sharpe | 最大回撤 | 相对等权 | IR | 买不进的占比（均值） |")
    P("|---|---|---|---|---|---|---|---|")
    pe = perf(ew)
    P(f"| 全市场等权 | {pe['ann']:.2%} | {pe['vol']:.2%} | {pe['sharpe']:.3f} | {pe['mdd']:.1%} | — | — | — |")
    for lb in ("y_exc", "y_neut"):
        col = f"s_{lb}"
        if col not in bt.columns:
            continue
        nav, sk = run(col)
        p = perf(nav, ew)
        P(f"| LGB_{lb} | {p['ann']:.2%} | {p['vol']:.2%} | {p['sharpe']:.3f} | {p['mdd']:.1%} | "
          f"{p['excess_ann']:+.2%} | {p['ir']:+.2f} | {sk.mean():.1%} |")
        nav.to_frame("nav").to_csv(os.path.join(OUT, f"m8c_nav_{lb}.csv"))
        P(f"  - 买不进比例分位 10/50/90: {np.percentile(sk,10):.1%} / {np.percentile(sk,50):.1%} / {np.percentile(sk,90):.1%}")

    with open(os.path.join(OUT, "m8c_pricecheck.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    print(f"[saved] {os.path.join(OUT,'m8c_pricecheck.md')}")


if __name__ == "__main__":
    main()
