# -*- coding: utf-8 -*-
"""t9b_smooth_suspect.py —— 「分数平滑」会不会是变相的前视？

t9 结果：R1 平滑 5 日 → 年化 30.76%、相对等权 +4.17%，换手降到 21.8 倍。
比原版好 9.6pp，比「缓冲区」所有档都好。**太好的东西必须先证伪。**

嫌疑：
  S1 **rolling 用了未来**：`bt.groupby("code")["s"].transform(lambda x: x.rolling(S).mean())`
     rolling 默认居中=False（向后看），这个是对的。**但我在 `bt` 上做 rolling，
     而 `bt` 已按日期过滤到 2024+，前 S 天的分数被截掉了** —— 这只会让前几行 NaN，不会前视。
  S2 **更严重的可能：`s` 这个分数本身是全样本训练的模型给出的 in-sample 预测？**
     不是。模型 train 2018-22 / valid 2023 / test 2024+，`s` 是 test 段预测，没有问题。
  S3 **平滑会不会实际上是「降低换手 → 少交税」而非「alpha 变强」？**
     0bp 成本下对比就能分辨：若平滑在 0bp 也更好，说明信号确实变强；若只在 15bp 好，纯成本效应。
  S4 **rolling 跨了股票分组边界？**  transform 在 groupby 内，不会。
  S5 **`min_periods=1` 让开头几天用了不足 S 天的数据** —— 无前视，但会引入噪声，不影响正确性。

验证方法（三条都对上才算真）：
  1. 0bp / 10bp / 15bp / 20bp 四档成本下都跑，看平滑的优势是否随成本上升而扩大（→ 纯成本效应）
     还是各档都一致更好（→ 信号变强）
  2. 用 `shift(1)` 的严格版本重跑：`s_lag = s.shift(1)` 后再 rolling，确保不含当日分数
  3. 分季度看稳定性

产出：output/m9b_smooth.md
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
    # ⚠️ 关键：面板用**全历史**（不预先截到 2024），这样 rolling 才有足够历史
    panel = panel.dropna(subset=["adj_close", "adj_open", "raw_close", "raw_open", "amount"])
    panel["lim"] = np.where(panel["board"].isin(["GEM", "STAR"]), 0.20, 0.10)
    panel["s"] = lgb.Booster(model_file=os.path.join(ROOT, "models", "lgb_y_neut.txt")).predict(panel[fc])
    panel = panel.sort_values(["code", "date"])

    # 严格版：先 shift(1) 再用过去 S 日均值（不含当日），避免任何前视争议
    panel["s_lag5"] = (panel.groupby("code")["s"]
                       .transform(lambda x: x.shift(1).rolling(5, min_periods=3).mean()))
    panel["s_ma5"] = (panel.groupby("code")["s"]
                      .transform(lambda x: x.rolling(5, min_periods=3).mean()))
    panel = panel.sort_values(["date", "code"]).reset_index(drop=True)

    bt = panel[(panel["date"] >= "2024-01-01") & (panel["date"] <= "2025-11-10")].copy()
    dates = np.array(sorted(bt["date"].unique()))
    by = {d: g.set_index("code") for d, g in bt.groupby("date")}
    dd = bt[["date", "code", "adj_close"]].sort_values(["code", "date"])
    dd["r"] = dd.groupby("code")["adj_close"].pct_change()
    ew = (1 + dd.groupby("date")["r"].mean().fillna(0)).cumprod()
    ew_r = ew.pct_change()
    P("# 分数平滑：是信号变强还是纯成本效应？\n")
    P(f"- 基准年化 {perf(ew)['ann']:.2%}，Sharpe {perf(ew)['sharpe']:.3f}")
    P("- 严格版 `s_lag5` = 先 shift(1) 再 rolling(5)，**完全不含当日分数**\n")

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

    P("## 成本 × 信号矩阵\n")
    P("| 信号 | 单边成本 | 年化 | Sharpe | 相对等权 | 单次换手 |")
    P("|---|---|---|---|---|---|")
    res = {}
    for col in ("s", "s_ma5", "s_lag5"):
        for cost in (0.0, 0.0010, 0.0015, 0.0020, 0.0030):
            nv, to = run(col, cost=cost)
            pf = perf(nv)
            rr = nv.pct_change().fillna(0)
            rb = ew_r.reindex(nv.index).fillna(0)
            exc = (1 + (rr - rb)).prod() ** (252 / len(rr)) - 1
            P(f"| {col} | {cost*1e4:.0f}bp | {pf['ann']:.2%} | {pf['sharpe']:.3f} | {exc:+.2%} | "
              f"{np.median(to):.1%} |")
            res[(col, cost)] = pf["ann"]
        P("| | | | | | |")

    P("\n## 判据\n")
    for col in ("s", "s_ma5", "s_lag5"):
        if (col, 0.0) in res:
            P(f"- {col}: 0bp {res[(col,0.0)]:.2%} → 30bp {res.get((col,0.003),float('nan')):.2%}")
    P("\n> 若 `s_ma5` 在 **0bp 就明显好于 `s`**，说明平滑让信号本身更强（不是省成本）。")
    P("> 若只在 15bp+ 好，说明纯粹是降换手的成本效应。")
    P("> 若 `s_lag5` 明显差于 `s_ma5`，说明 `s_ma5` 的好结果部分依赖当日分数。")

    with open(os.path.join(OUT, "m9b_smooth.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    print(f"[saved] {os.path.join(OUT,'m9b_smooth.md')}")


if __name__ == "__main__":
    main()
