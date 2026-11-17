# -*- coding: utf-8 -*-
"""t5e_capacity.py —— 容量检验：Top-K 里有多少钱买得进

为什么必须做：D4 分十组显示，模型分数最高的一组（Q10）**中位成交额只有 0.37 亿/日、
中位股价 7.1 元**；等权 Top30 意味着每只买 1/30 的资金。
散户自有资金问题不大，但「策略能不能复制」必须给出资金上限。

口径（写死，便于复核）：
  · 单票单日资金上限 = 该股当日成交额 × 参与率 p（p 取 1% / 5% / 10%）
  · 每次调仓买 K 只 → 单次可投上限 = Σ_{i∈picks} 成交额_i × p
  · 由于每 hold 日调仓一次、每次替换大部分仓位，年周转约 252/hold 次
  · 报告：picks 的成交额分位数、以及不同 p 下的「单次可投金额」

产出：output/m5e_capacity.md
"""
import os, sys
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
OUT = os.path.join(ROOT, "output")
import t5_backtest as T5

L = []


def P(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    L.append(s)


def main():
    panel = T5.load_panel()
    import lightgbm as lgb
    import t4_lgb
    fc = t4_lgb.feat_cols(panel)
    S, E = "2024-01-01", "2025-11-10"

    for lb in ("y_raw", "y_exc"):
        mp = os.path.join(ROOT, "models", f"lgb_{lb}.txt")
        if not os.path.exists(mp):
            continue
        bst = lgb.Booster(model_file=mp)
        panel[f"score_{lb}"] = bst.predict(panel[fc])

    d = panel[(panel["date"] >= S) & (panel["date"] <= E)].copy()
    P("# 容量检验\n")
    P("> 「成交额」= 当日金额（元）。参与率 p = 单票买入额 / 当日成交额。\n")

    rows = []
    for lb in ("y_raw", "y_exc"):
        col = f"score_{lb}"
        for K in (30, 100):
            picks_amt = []
            picks_px = []
            for dt, g in d.groupby("date"):
                cand = g.dropna(subset=[col, "adj_close"])
                cand = cand[cand["amount"] > 2e7]
                if len(cand) < K:
                    continue
                pk = cand.nlargest(K, col)
                picks_amt.append(pk["amount"].sum())
                # ⚠️ volume 单位是「手」（100股），所以 amount/volume 是「元/手」，不是股价。
                # 用 raw_close 才是真实股价。
                picks_px.append(pk["raw_close"].median())
            a = np.array(picks_amt)
            rows.append({"model": lb, "K": K, "n_rebal": len(a),
                         "median_sum_amt_yi": np.median(a) / 1e8,
                         "p10_sum_amt_yi": np.percentile(a, 10) / 1e8,
                         "median_price": float(np.nanmedian(picks_px)),
                         "cap_p1_yi": np.median(a) * 0.01 / 1e8,
                         "cap_p5_yi": np.median(a) * 0.05 / 1e8,
                         "cap_p10_yi": np.median(a) * 0.10 / 1e8})
    R = pd.DataFrame(rows)
    P(R.round(2).to_string(index=False))
    P("\n> `cap_pX_yi` = 单次调仓在参与率 X% 下最多能投入的金额（亿元）。按持有 5 日、每年约 50 次换手，")
    P("> 实际可得资金量级与 `cap_pX_yi` 同阶（每次换掉大部分仓位）。\n")

    # 小市值/低价集中度
    q = d.groupby("date")[f"score_y_exc"].transform(lambda s: pd.qcut(s.rank(method="first"), 10, labels=False)) + 1
    top = d[q == 10]
    P(f"\n## 最高分组（Q10）画像\n")
    P(f"- 中位成交额: {top['amount'].median()/1e8:.2f} 亿元/日")
    P(f"- 中位股价: {top['raw_close'].median():.2f} 元")
    # 流通市值代理：收盘价 × 流通股数，其中流通股数 = 成交量(手)×100 / (换手率/100)
    free_sh = top["volume"] * 100 / (top["turnover"].replace(0, np.nan) / 100)
    mcap = (top["raw_close"] * free_sh).replace([np.inf, -np.inf], np.nan)
    P(f"- 中位流通市值（收盘 × 流通股数，流通股数 = 成交量×100/换手率）: {mcap.median()/1e8:.1f} 亿元")
    P(f"- 未来 5 日含涨停比例: {top['fwd5_has_zt'].mean():.2%}（全市场 {d['fwd5_has_zt'].mean():.2%}）")

    R.to_csv(os.path.join(OUT, "m5e_capacity.csv"), index=False)
    with open(os.path.join(OUT, "m5e_capacity.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    print(f"[saved] {os.path.join(OUT,'m5e_capacity.md')}")


if __name__ == "__main__":
    main()
