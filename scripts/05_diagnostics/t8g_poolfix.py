# -*- coding: utf-8 -*-
"""t8g_poolfix.py —— 修 t8f 的前视偏差，并给 V3b 做稳健性 + 容量

t8f 的 bug：`amount≥3亿 池等权` 年化 220%。
原因：用**当日 amount** 筛池子、用**当日** adj_close.pct_change() 算收益
     → 成交额放量的那天恰好是涨幅最大的那天，事后筛选 = 前视偏差。

正确做法（与 V3b 回测口径一致）：
  dt 日收盘后用 dt 日的 amount 筛池 → t+1 日**开盘**建仓 → 持有到 t+2 日开盘
  用复权价算收益，用原始价判涨停（池等权不判，因为不选股；但要报告买不进比例）

本脚本输出：
  A. 修正后的池等权基准（t 日筛 / t+1 开盘买 / 持 1 日、5 日两种）
  B. V3b 的季度稳定性
  C. V3b 容量（picks 的成交额合计 × 参与率）

产出：output/m8g_poolfix.md
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
            "mdd": (nav / nav.cummax() - 1).min(), "n": len(r)}


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

    P("# 修正后的池基准 + V3b 稳健性 + 容量\n")

    # ---------- A. 池等权（t 日筛，t+1 开盘买，持有 H 日） ----------
    def pool_nav(min_amt=3e8, H=1):
        """每个调仓日 t（每 H 天一次），t 日筛池，t+1 开盘买入，等权持有 H 天"""
        cash = 1.0
        out = []
        weights = {}
        hold_px = {}
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
            if i % H != 0 or i == len(dates) - 1:
                continue
            nxt = dates[i + 1]
            g2 = by[nxt]
            pool = g[g["amount"] > min_amt]
            if len(pool) == 0:
                continue
            new = {}
            for c in pool.index:
                if c in g2.index and np.isfinite(g2.at[c, "adj_open"]):
                    new[c] = 1.0 / len(pool)
                    hold_px[c] = g2.at[c, "adj_open"]
            weights = new
        return pd.Series(dict(out)).sort_index()

    dd = bt[["date", "code", "adj_close"]].sort_values(["code", "date"])
    dd["r"] = dd.groupby("code")["adj_close"].pct_change()
    ew = (1 + dd.groupby("date")["r"].mean().fillna(0)).cumprod()

    P("## A. 池等权基准（修正前视偏差）\n")
    P("| 组合 | 年化 | 波动 | Sharpe | 最大回撤 |")
    P("|---|---|---|---|---|")
    for nm, nv in (("全市场等权", ew), ("amount≥1亿 池等权（持5日）", pool_nav(1e8, 5)),
                   ("amount≥3亿 池等权（持5日）", pool_nav(3e8, 5)),
                   ("amount≥3亿 池等权（持1日，日频）", pool_nav(3e8, 1))):
        pf = perf(nv)
        if pf:
            P(f"| {nm} | {pf['ann']:.2%} | {pf['vol']:.2%} | {pf['sharpe']:.3f} | {pf['mdd']:.1%} |")
            nv.to_frame("nav").to_csv(os.path.join(OUT, f"m8g_nav_{nm.split('（')[0].replace(' ','_').replace('≥','ge')}.csv"))
    P("\n> 若「3亿以上池等权」年化接近 0 或为负，就和 t8e 的「3 亿以上中位 −0.469%」对上了，")
    P("> 那么 t8f 的随机零分布 −11.82% 也就有了合理解释：随机选 30 只 ≈ 池子本身的收益。\n")

    # ---------- B. V3b 季度稳健性 ----------
    def v3b_nav(min_amt=3e8, K=30, hold=5, cost=0.0015):
        cash = 1.0
        out, hold_px, weights = [], {}, {}
        recs = []
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
            cand = g.dropna(subset=["s", "adj_close"])
            cand = cand[cand["amount"] > min_amt]
            if len(cand) < K:
                continue
            picks = cand.nlargest(K, "s")
            buy = []
            for c in picks.index:
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
            recs.append({"date": dt, "n": len(buy), "amt_sum": picks.loc[buy, "amount"].sum()})
            new = {c: 1.0 / len(buy) for c in buy}
            for c in buy:
                hold_px[c] = g2.at[c, "adj_open"]
            old = set(weights)
            to = 1.0 - len(old & set(new)) / max(len(new), 1) if old else 1.0
            cash *= (1 - cost * to * 2)
            weights = new
        return pd.Series(dict(out)).sort_index(), pd.DataFrame(recs)

    nav, rec = v3b_nav()
    P("## B. V3b 季度稳健性\n")
    r = nav.pct_change().dropna()
    q = r.groupby([r.index.year, r.index.quarter])
    P("| 季度 | 交易日 | 收益 |")
    P("|---|---|---|")
    for (y, qq), s in q:
        if len(s) < 10:
            continue
        P(f"| {y}Q{qq} | {len(s)} | {(1+s).prod()-1:+.2%} |")
    P(f"\n- V3b 全期年化 {perf(nav)['ann']:.2%}，Sharpe {perf(nav)['sharpe']:.3f}")

    # ---------- C. 容量 ----------
    P("\n## C. V3b 容量\n")
    P("| 参与率 | 单次可投（中位） | 单次可投（10 分位） |")
    P("|---|---|---|")
    for pr in (0.01, 0.005, 0.002):
        P(f"| {pr:.1%} | {rec['amt_sum'].median()*pr/1e8:.2f} 亿 | {rec['amt_sum'].quantile(.1)*pr/1e8:.2f} 亿 |")
    P(f"\n- picks 成交额合计中位 {rec['amt_sum'].median()/1e8:.2f} 亿（对比 t5e 原 Top30 的 11.36 亿）")

    with open(os.path.join(OUT, "m8g_poolfix.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    print(f"[saved] {os.path.join(OUT,'m8g_poolfix.md')}")


if __name__ == "__main__":
    main()
