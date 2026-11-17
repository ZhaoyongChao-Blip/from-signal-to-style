# -*- coding: utf-8 -*-
"""t8i_hold.py —— 换手率 57.6%/轮太高的解法：延长持有期

V3b（成交额≥3亿，持有5日）的问题：单边换手 57.6%/轮，年化 58 倍，成本 8.7%/年。
**成本吃掉 alpha 大头**：0bp 时年化 32.76%，15bp 时 21.17%，差 11.6pp。

本脚本测试：延长持有期能否降低换手、保住 alpha？
  · hold = 5 / 10 / 15 / 20 日
  · 每个 hold 下扫成本 0/10/15/20/30bp

同时修正 t8g/t8h 的池基准问题：
  之前「持5日修正」版波动 43.66%、回撤 −56%，明显不对（全市场等权才 25.67%/25.7%）。
  原因：池子每天重筛 + `cnt % H` 计数在建仓后没有重置，导致持有期错乱。
  本脚本用「明确记录建仓日 + 到期日」的方式重写，不再依赖计数器。

产出：output/m8i_hold.md
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
    idx = {d: i for i, d in enumerate(dates)}
    by = {d: g.set_index("code") for d, g in bt.groupby("date")}
    P("# 延长持有期能否降换手\n")

    def strat(min_amt=3e8, K=30, hold=5, cost=0.0015, mode="top"):
        """明确建仓日/到期日，不用计数器。
        第 i 个交易日：若 i 是某笔持仓的到期日（i = start + hold + 1），则换仓。
        """
        cash = 1.0
        out = []
        pos = {}          # code -> {"w":权重, "buy_px":买价}
        expiry = -1
        to_list = []
        for i, dt in enumerate(dates):
            g = by[dt]
            # 估值
            if pos:
                rets = 0.0
                for c, v in pos.items():
                    p1 = g["adj_close"].get(c, np.nan)
                    if np.isfinite(p1) and v["buy_px"] > 0:
                        rets += v["w"] * (p1 / v["buy_px"] - 1)
                # 用相对上一日净值的增量（避免重复计）
                pass
            # 简化：每天按「持仓从 buy_px 到今日 close 的累计收益」重算净值水平
            # 这样不需要累乘日收益，也不会重复计
            if pos:
                cum = 0.0
                for c, v in pos.items():
                    p1 = g["adj_close"].get(c, np.nan)
                    if np.isfinite(p1) and v["buy_px"] > 0:
                        cum += v["w"] * (p1 / v["buy_px"] - 1)
                nav_now = cash_at_start * (1 + cum)
            else:
                nav_now = cash
            out.append((dt, nav_now))

            if i < expiry or i == len(dates) - 1:
                continue
            # 换仓日
            nxt_i = i + 1
            if nxt_i >= len(dates):
                break
            nxt = dates[nxt_i]
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
            old = set(pos)
            new_s = set(buy)
            to = 1.0 - len(old & new_s) / max(len(new_s), 1) if old else 1.0
            to_list.append(to)
            cost_amt = cost * to * 2
            base = nav_now if pos else cash
            cash_at_start = base * (1 - cost_amt)
            pos = {c: {"w": 1.0 / len(buy), "buy_px": g2.at[c, "adj_open"]} for c in buy}
            expiry = i + 1 + hold
            cash = cash_at_start

        return pd.Series(dict(out)).sort_index(), np.array(to_list)

    # 初始化 cash_at_start
    cash_at_start = 1.0

    dd = bt[["date", "code", "adj_close"]].sort_values(["code", "date"])
    dd["r"] = dd.groupby("code")["adj_close"].pct_change()
    ew = (1 + dd.groupby("date")["r"].mean().fillna(0)).cumprod()

    P("## 持有期 × 成本 网格\n")
    P("| 持有 | 单边成本 | 年化 | 波动 | Sharpe | 相对等权 | 中位换手/轮 |")
    P("|---|---|---|---|---|---|---|")
    rows = []
    for H in (5, 10, 15, 20):
        for c in (0.0, 0.0010, 0.0015, 0.0020, 0.0030):
            cash_at_start = 1.0
            nv, to = strat(hold=H, cost=c)
            pf = perf(nv)
            if not pf:
                continue
            rb = ew.reindex(nv.index).ffill().pct_change().fillna(0)
            rr = nv.pct_change().fillna(0)
            exc = (1 + (rr - rb)).prod() ** (252 / len(rr)) - 1
            tr = np.median(to) if len(to) else np.nan
            P(f"| {H} | {c*1e4:.0f}bp | {pf['ann']:.2%} | {pf['vol']:.2%} | {pf['sharpe']:.3f} | {exc:+.2%} | {tr:.1%} |")
            rows.append((H, c, pf["ann"], pf["sharpe"], exc, tr))

    with open(os.path.join(OUT, "m8i_hold.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    print(f"[saved] {os.path.join(OUT,'m8i_hold.md')}")


if __name__ == "__main__":
    main()
