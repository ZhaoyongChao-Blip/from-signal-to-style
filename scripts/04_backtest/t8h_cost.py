# -*- coding: utf-8 -*-
"""t8h_cost.py —— V3b 的换手率与成本敏感性（决定这个结论能不能站住）

为什么必须做：
  · V3b 年化 21.17% 只用了单边 15bp，若换手率极高（5 日一轮全换），15bp 可能严重低估
  · A 股真实成本：佣金 2.5bp（双边）+ 印花税 5bp（卖出）+ 冲击成本（取决于参与率）
  · 3 亿以上的票冲击成本低，但不是 0。必须做成本扫描。

同时修 t8g 的「持 1 日」口径错误：
  错误做法：dt 日筛池 → t+1 开盘买 → 当日又用 adj_close 估值（重复计一次）
  正确做法：dt 日筛池 → t+1 开盘买 → t+2 开盘卖（持有 1 个完整交易日）

产出：output/m8h_cost.md
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
    bt = bt.dropna(subset=["adj_close", "adj_open", "raw_close", "raw_open", "amount"])
    bt["lim"] = np.where(bt["board"].isin(["GEM", "STAR"]), 0.20, 0.10)
    bt["s"] = lgb.Booster(model_file=os.path.join(ROOT, "models", "lgb_y_neut.txt")).predict(bt[fc])
    bt = bt.sort_values(["date", "code"]).reset_index(drop=True)
    dates = np.array(sorted(bt["date"].unique()))
    by = {d: g.set_index("code") for d, g in bt.groupby("date")}
    P("# V3b 成本敏感性与换手率\n")

    # ---------- 1. 修正「持 1 日」的池基准 ----------
    def pool_nav_correct(min_amt=3e8, H=1):
        """dt 日筛池 → t+1 开盘买 → 持有 H 个完整交易日后卖出（用 adj_open 买卖，口径自洽）"""
        cash = 1.0
        out = []
        weights, buy_px = {}, {}
        cnt = 0
        for i, dt in enumerate(dates):
            g = by[dt]
            if weights:
                rets = {}
                for c, w in weights.items():
                    p0 = buy_px.get(c, np.nan)
                    p1 = g["adj_open"].get(c, np.nan)   # ⚠️ 用开盘价估值 → 与开盘建仓对齐
                    rets[c] = (p1 / p0 - 1) if (np.isfinite(p0) and np.isfinite(p1) and p0 > 0) else 0.0
                cash *= (1 + sum(w * rets[c] for c, w in weights.items()))
            out.append((dt, cash))
            cnt += 1
            if cnt % H != 0 or i == len(dates) - 1:
                continue
            nxt = dates[i + 1]
            g2 = by[nxt]
            pool = g[g["amount"] > min_amt]
            if len(pool) == 0:
                continue
            new, bp = {}, {}
            for c in pool.index:
                if c in g2.index and np.isfinite(g2.at[c, "adj_open"]):
                    new[c] = 1.0 / len(pool)
                    bp[c] = g2.at[c, "adj_open"]
            weights, buy_px = new, bp
        return pd.Series(dict(out)).sort_index()

    P("## 1. 修正后的池基准（开仓与估值都用开盘价，口径自洽）\n")
    P("| 组合 | 年化 | 波动 | Sharpe | 最大回撤 |")
    P("|---|---|---|---|---|")
    dd = bt[["date", "code", "adj_close"]].sort_values(["code", "date"])
    dd["r"] = dd.groupby("code")["adj_close"].pct_change()
    ew = (1 + dd.groupby("date")["r"].mean().fillna(0)).cumprod()
    for nm, nv in (("全市场等权（收盘价口径）", ew),
                   ("amount≥3亿 池等权（持1日，修正）", pool_nav_correct(3e8, 1)),
                   ("amount≥3亿 池等权（持5日，修正）", pool_nav_correct(3e8, 5)),
                   ("amount≥1亿 池等权（持5日，修正）", pool_nav_correct(1e8, 5))):
        pf = perf(nv)
        if pf:
            P(f"| {nm} | {pf['ann']:.2%} | {pf['vol']:.2%} | {pf['sharpe']:.3f} | {pf['mdd']:.1%} |")
    P("\n> 「持1日」修正前是 74.45%，修正后应落到合理区间。差异来自：原版用 adj_close 估值")
    P("> 而建仓用的是 adj_open，导致**开盘到收盘这一段被算了两次**（建仓日一次、次日估值一次）。\n")

    # ---------- 2. V3b 换手率 ----------
    def v3b(cost=0.0015, min_amt=3e8, K=30, hold=5):
        cash = 1.0
        out, hold_px, weights = [], {}, {}
        to_list = []
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
            to_list.append(to)
            cash *= (1 - cost * to * 2)
            weights = new
        return pd.Series(dict(out)).sort_index(), np.array(to_list)

    nav, to = v3b()
    P("## 2. V3b 换手率\n")
    P(f"- 单边换手率（每 5 日一轮）：中位 **{np.median(to):.1%}**，均值 {to.mean():.1%}")
    P(f"- 年化换手 = {np.median(to):.1%} × 2（双边）× 50.4 轮 ≈ **{np.median(to)*2*50.4:.0f} 倍/年**")
    P(f"- 每轮成本 15bp × 双边 × 中位换手 = {0.0015*2*np.median(to):.4%}/轮")
    P(f"- 年化成本 ≈ {0.0015*2*np.median(to)*50.4:.2%}\n")

    # ---------- 3. 成本扫描 ----------
    P("## 3. 成本敏感性（单边）\n")
    P("| 单边成本 | 年化 | 波动 | Sharpe | 相对全市场等权 |")
    P("|---|---|---|---|---|")
    for c in (0.0, 0.0005, 0.0010, 0.0015, 0.0020, 0.0030, 0.0050):
        nv, _ = v3b(cost=c)
        pf = perf(nv)
        rb = ew.reindex(nv.index).ffill().pct_change().fillna(0)
        rr = nv.pct_change().fillna(0)
        exc = (1 + (rr - rb)).prod() ** (252 / len(rr)) - 1
        P(f"| {c*1e4:.0f} bp | {pf['ann']:.2%} | {pf['vol']:.2%} | {pf['sharpe']:.3f} | {exc:+.2%} |")
    P("\n> A 股真实成本参考：佣金双边 5bp + 印花税卖出 5bp + 冲击成本。")
    P("> 3 亿以上票在 1% 参与率下冲击成本约 5~10bp，合计单边约 10~15bp。")
    P("> **若 30bp 时仍打平全市场等权，结论稳健；若 20bp 就崩，说明不可执行。**")

    with open(os.path.join(OUT, "m8h_cost.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    print(f"[saved] {os.path.join(OUT,'m8h_cost.md')}")


if __name__ == "__main__":
    main()
