# -*- coding: utf-8 -*-
"""t8j_suspect.py —— t8i 结果太好了，必须先证伪

t8i 报：持有 10 日 / 15bp → 年化 33.03%、Sharpe 1.632、相对等权 +6.45%。
但同一模型在 t8d 的 V1a（无成交额限制、5 日）只有 11.68%/0.383。
**加一个「成交额≥3亿」的过滤，年化从 11.68% 跳到 33%？** 这必须查。

四个待查嫌疑：
  S1 **cash_at_start 是全局变量**，`strat()` 里的 `cash_at_start` 修改的是函数内局部（因为
     函数内有赋值），但开头 `if pos: ... nav_now = cash_at_start * (...)` 可能读到上一轮的外层值
     → 净值序列可能不连续
  S2 **nav 计算方式变了**：t8i 用「持仓从 buy_px 到今日 close 的累计收益」
     而 t8d 用「逐日收益累乘」。前者若 pos 为空的日子沿用上一个 nav，会**抹掉空仓期的收益**
  S3 **expiry 逻辑**：`expiry = i + 1 + hold`，但 i 是「信号日」，建仓在 i+1，
     所以实际持有期是 hold 天还是 hold+1 天？边界差 1 天可能造成重叠/空档
  S4 **换手率反而升了**（5日 60% → 10日 63.3% → 20日 71.2%）
     → 持有期越长换手越高，逻辑上不可能，除非**到期日计算错误导致频繁重建**

做法：把 t8d 的回测函数原样搬过来（那是已验证过的口径），只改 min_amt 和 hold，重新跑。
如果结果与 t8i 差很多，说明 t8i 有 bug。

产出：output/m8j_suspect.md
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

    dd = bt[["date", "code", "adj_close"]].sort_values(["code", "date"])
    dd["r"] = dd.groupby("code")["adj_close"].pct_change()
    ew = (1 + dd.groupby("date")["r"].mean().fillna(0)).cumprod()
    P("# t8i 结果证伪\n")

    # ===== 口径 A：t8d 的已验证函数（逐日收益累乘，每 hold 天换仓）=====
    def run_t8d(min_amt=3e8, K=30, hold=5, cost=0.0015):
        cash = 1.0
        hold_px, weights = {}, {}
        out = []
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

    P("## 已验口径（逐日收益累乘）\n")
    P("| min_amt | hold | 成本 | 年化 | 波动 | Sharpe | 相对等权 | 换手/轮 |")
    P("|---|---|---|---|---|---|---|---|")
    for ma, nm in ((3e8, "3亿"), (2e7, "2千万")):
        for H in (5, 10):
            for c in (0.0015,):
                nv, to = run_t8d(min_amt=ma, hold=H, cost=c)
                pf = perf(nv)
                rb = ew.reindex(nv.index).ffill().pct_change().fillna(0)
                rr = nv.pct_change().fillna(0)
                exc = (1 + (rr - rb)).prod() ** (252 / len(rr)) - 1
                P(f"| {nm} | {H} | {c*1e4:.0f}bp | {pf['ann']:.2%} | {pf['vol']:.2%} | {pf['sharpe']:.3f} | "
                  f"{exc:+.2%} | {np.median(to):.1%} |")
                nv.to_frame("nav").to_csv(os.path.join(OUT, f"m8j_nav_{nm}_{H}d.csv"))

    P("\n## 与 t8i 对比\n")
    P("| 口径 | hold=10, 15bp 年化 | 换手/轮 |")
    P("|---|---|---|")
    P("| t8i（累计收益重算法） | 33.03% | 63.3% |")
    P("| t8d 已验证口径（逐日累乘） | 见上表 | 见上表 |")
    P("\n> 若两者差异 >5pp，t8i 的重算法有 bug，**t8i 全部结果作废**，以 t8d 口径为准。")

    with open(os.path.join(OUT, "m8j_suspect.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    print(f"[saved] {os.path.join(OUT,'m8j_suspect.md')}")


if __name__ == "__main__":
    main()
