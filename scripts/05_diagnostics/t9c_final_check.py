# -*- coding: utf-8 -*-
"""t9c_final_check.py —— 分数平滑的最后三道检验

t9b 结果：
  · s_ma5（含当日分数）0bp 39.20%、15bp 29.71%（+3.35%）
  · s_lag5（不含当日分数）0bp 33.86%、15bp 24.68%（−0.42%）
两者差 5.3pp，说明**当日分数贡献了相当大一部分收益**。

三种解释，必须分开：
  E1 **正常**：当日分数确实是最新信息，模型每天更新，用最新的当然更好。
     `s_ma5` 在收盘后可用、次日开盘买入，**时序上没有前视**。
     `s_lag5` 主动丢弃一天信息，差是应该的。
  E2 **前视**：`s` 本身是 in-sample 预测。→ 已排除（模型 train 2018-22 / test 2024+）。
  E3 **过拟合**：平滑窗口 5 日是我看了结果才选的。→ 必须做窗口敏感性 + 分年度。

本脚本做三件事：
  1. **窗口敏感性**：S = 1/2/3/5/10/20，看是否有「选了个好窗口」
  2. **分年度/分季度**：s_ma5 vs s 的差异是否只在某一段
  3. **vs 全市场等权 + 全市场 TopK 随机**：placebo，确认 29.71% 不是 β 巧合

注：`s_ma5` 的时序安全性——模型在 dt 日收盘后用 dt 日数据算分，t+1 日开盘买入。
窗口 [dt-4, dt] 全在决策时刻之前，**无前视**。与 `s` 的唯一差别是「用了 5 天而非 1 天」。

产出：output/m9c_final.md
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
    panel = panel.dropna(subset=["adj_close", "adj_open", "raw_close", "raw_open", "amount"])
    panel["lim"] = np.where(panel["board"].isin(["GEM", "STAR"]), 0.20, 0.10)
    panel["s"] = lgb.Booster(model_file=os.path.join(ROOT, "models", "lgb_y_neut.txt")).predict(panel[fc])
    panel = panel.sort_values(["code", "date"])
    for S in (2, 3, 5, 10, 20):
        panel[f"s_ma{S}"] = (panel.groupby("code")["s"]
                             .transform(lambda x, S=S: x.rolling(S, min_periods=max(2, S // 2)).mean()))
    panel = panel.sort_values(["date", "code"]).reset_index(drop=True)

    bt = panel[(panel["date"] >= "2024-01-01") & (panel["date"] <= "2025-11-10")].copy()
    dates = np.array(sorted(bt["date"].unique()))
    by = {d: g.set_index("code") for d, g in bt.groupby("date")}
    dd = bt[["date", "code", "adj_close"]].sort_values(["code", "date"])
    dd["r"] = dd.groupby("code")["adj_close"].pct_change()
    ew = (1 + dd.groupby("date")["r"].mean().fillna(0)).cumprod()
    ew_r = ew.pct_change()
    P("# 分数平滑最终检验\n")
    P(f"- 基准年化 {perf(ew)['ann']:.2%}，Sharpe {perf(ew)['sharpe']:.3f}\n")

    def run(col, min_amt=3e8, hold=5, cost=0.0015, K=30):
        cash = 1.0
        hold_px, weights = {}, {}
        out, to_list = [], []
        for i, dt in enumerate(dates):
            g = by[dt]
            if weights:
                s_ = 0.0
                for c, w in weights.items():
                    p0 = hold_px.get(c, np.nan)
                    p1 = g["adj_close"].get(c, np.nan)
                    if np.isfinite(p0) and np.isfinite(p1) and p0 > 0:
                        s_ += w * (p1 / p0 - 1)
                cash *= (1 + s_)
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
            picks = cand.nlargest(K, col).index.tolist()
            buy = []
            for c in picks:
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

    P("## 1. 平滑窗口敏感性（15bp，3亿池）\n")
    P("| 窗口 | 年化 | Sharpe | 相对等权 | 单次换手 |")
    P("|---|---|---|---|---|")
    navs = {}
    for S, col in [(1, "s")] + [(S, f"s_ma{S}") for S in (2, 3, 5, 10, 20)]:
        if col not in bt.columns:
            continue
        nv, to = run(col)
        pf = perf(nv)
        rr = nv.pct_change().fillna(0)
        rb = ew_r.reindex(nv.index).fillna(0)
        exc = (1 + (rr - rb)).prod() ** (252 / len(rr)) - 1
        P(f"| {S} 日 | {pf['ann']:.2%} | {pf['sharpe']:.3f} | {exc:+.2%} | {np.median(to):.1%} |")
        navs[S] = nv

    P("\n> 若只有 5 日好、其他窗口差 → 是过拟合。若 3/5/10 都好 → 平滑是真机制。\n")

    P("## 2. 分年度（s vs s_ma5）\n")
    P("| 年份 | s 收益 | s_ma5 收益 | 差 |")
    P("|---|---|---|---|")
    for y in (2024, 2025):
        a = navs.get(1)
        b = navs.get(5)
        if a is None or b is None:
            continue
        ra = a[a.index.year == y].pct_change().fillna(0)
        rb2 = b[b.index.year == y].pct_change().fillna(0)
        P(f"| {y} | {(1+ra).prod()-1:+.2%} | {(1+rb2).prod()-1:+.2%} | {((1+rb2).prod()-(1+ra).prod())*100:+.2f}pp |")

    P("\n## 3. 分季度（s_ma5）\n")
    b = navs.get(5)
    if b is not None:
        rb2 = b.pct_change().dropna()
        q = rb2.groupby([rb2.index.year, rb2.index.quarter])
        P("| 季度 | 收益 |")
        P("|---|---|")
        for (y, qq), s in q:
            if len(s) < 10:
                continue
            P(f"| {y}Q{qq} | {(1+s).prod()-1:+.2%} |")

    P("\n## 4. 结论要素\n")
    P("- `s_ma5` 与 `s` 的时序安全性相同：都在 dt 日收盘后决策、t+1 开盘买入。")
    P("- 差别只是「用 5 天平均分」而不是「用当天分数」。**没有前视。**")
    P("- 但窗口选择必须看敏感性表：只有一个窗口好 = 过拟合。")

    with open(os.path.join(OUT, "m9c_final.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    print(f"[saved] {os.path.join(OUT,'m9c_final.md')}")


if __name__ == "__main__":
    main()
