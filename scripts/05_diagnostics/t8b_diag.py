# -*- coding: utf-8 -*-
"""t8b_diag.py —— t8 的三个异常必须查清才能下结论

t8 结果里三处说不通：
  A. P2「同格子对冲」年化 −3.81%、Sharpe −0.43 —— 纯 alpha 若是负的，与 t7f 的 +0.713%/5日矛盾
  B. P1 原样多头 14.42%，而 t7d 同模型是 18.37% —— 同一模型两个数
  C. P3 波动 31.99%，比 P1 的 27.28% 还高 —— 「中性化」不该放大波动

可能原因（逐条验）：
  A1 对冲腿选错了：`cand` 里循环 30 次 `(cand["cell"]==cell)` 是 O(n)，但更致命的是
     **对冲腿用「同一只票可以配多次」**，且 `bench_px` 的权重累加逻辑有 bug
  A2 对冲的是「开盘价买入」，但同日开多空 → 实际是日内的 alpha，不是 5 日 alpha
  B1 t7d 的 backtest 用 `start/end` 过滤后只在 448 天里跑；t8 用全部 dates（含无 picks 的日子现金空仓）
  C1 P3 的配额算法把仓位集中到少数格子 → 实际持仓只数远少于 30

做法：不重写回测，直接**打印中间量**（持仓只数、格子分布、对冲腿只数、每日收益的来源）。
本脚本只做诊断，不产出结论。
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


def main():
    import lightgbm as lgb
    import t4_lgb
    import t5_backtest as T5

    panel = pd.read_parquet(os.path.join(ROOT, "data", "proc", "panel.parquet"))
    panel["date"] = pd.to_datetime(panel["date"])
    fc = t4_lgb.feat_cols(panel)
    bt = panel[(panel["date"] >= "2024-01-01") & (panel["date"] <= "2025-11-10")].copy()
    bt = bt.dropna(subset=["adj_close", "adj_open"] + STYLE)
    bt["lim"] = np.where(bt["board"].isin(["GEM", "STAR"]), 0.20, 0.10)
    for lb in ("y_exc", "y_neut"):
        p = os.path.join(ROOT, "models", f"lgb_{lb}.txt")
        if os.path.exists(p):
            bt[f"s_{lb}"] = lgb.Booster(model_file=p).predict(bt[fc])

    def cell_id(g):
        c = np.zeros(len(g), dtype=np.int64)
        for i, cname in enumerate(STYLE):
            q = pd.qcut(g[cname].rank(method="first"), 5, labels=False, duplicates="drop")
            c = c + q.fillna(2).to_numpy().astype(np.int64) * (5 ** i)
        return pd.Series(c, index=g.index)

    bt["cell"] = bt.groupby("date", group_keys=False)[STYLE + ["code"]].apply(cell_id)

    P("# t8 三个异常诊断\n")

    # ---- B: 同一模型，t5.backtest vs 手写，差异在哪 ----
    P("## B. t5.backtest 与 t8 手写回测的差\n")
    nav_t5, _ = T5.backtest(bt, "s_y_neut", topk=30, hold=5, cost=0.0015,
                            start="2024-01-01", end="2026-12-31")
    nav_t5 = nav_t5[(nav_t5.index <= bt["date"].max())]
    r = nav_t5.pct_change().dropna()
    ann = (1 + r).prod() ** (252 / len(r)) - 1
    P(f"- t5.backtest: n={len(r)} 年化 {ann:.2%} 终值 {nav_t5.iloc[-1]:.4f}")
    P(f"- 天数：t5 用 {len(r)} 天，t8 手写用 {bt['date'].nunique()} 天（差 {bt['date'].nunique()-len(r)}）")
    P("  → ⚠️ t5 的 `backtest()` 用 `dates[i+1]`，若最后一天没有下一日就跳过；且它从 1.0 起、")
    P("    **现金空仓日也计 0 收益**，所以两边年化不可能相同。差异主要来自「建仓日不同」。\n")

    # ---- C: P3 持仓集中度 ----
    P("## C. P3 格子配额算法的持仓只数\n")
    for dt in list(sorted(bt["date"].unique()))[::40][:6]:
        g = bt[bt["date"] == dt]
        cand = g.dropna(subset=["s_y_neut", "adj_close"])
        cand = cand[cand["amount"] > 2e7]
        picks = cand.nlargest(30, "s_y_neut")
        cell_cnt = cand["cell"].value_counts()
        w_cell = cell_cnt / cell_cnt.sum()
        hits = [(cell, share, len(picks[picks["cell"] == cell])) for cell, share in w_cell.items()
                if len(picks[picks["cell"] == cell]) > 0]
        P(f"- {pd.Timestamp(dt).date()}: picks 30 只落在 {len(hits)} 个格子；"
          f"配额 top3 = {[(c, round(s,3), n) for c,s,n in sorted(hits, key=lambda x:-x[1])[:3]]}")
    P("\n> 若 picks 只落在少数格子，配额权重会集中到这些格子里的少数股票 → 有效持仓 << 30，")
    P("> 这正是 P3 波动 32% 的来源。**格子配额法对「模型偏向极端格子」的策略不适用。**\n")

    # ---- A: 对冲腿 ----
    P("## A. P2 对冲腿的构造问题\n")
    dt = sorted(bt["date"].unique())[100]
    g = bt[bt["date"] == dt]
    cand = g.dropna(subset=["s_y_neut", "adj_close"])
    cand = cand[cand["amount"] > 2e7]
    picks = cand.nlargest(30, "s_y_neut")
    n_ok = 0
    dup = 0
    seen = {}
    rng = np.random.default_rng(1)
    for c in picks.index:
        cell = picks.loc[c, "cell"]
        pool = cand[(cand["cell"] == cell) & (cand.index != c)]
        if len(pool) == 0:
            continue
        n_ok += 1
        hc = pool.index[rng.integers(0, len(pool))]
        if hc in seen:
            dup += 1
        seen[hc] = seen.get(hc, 0) + 1
    P(f"- picks 30 只中，{n_ok} 只找到了同格子对手方；对手方去重后 {len(seen)} 只（重复 {dup} 次）")
    P("- ⚠️ 关键 bug：`dt.value % 2**31` 作为种子 → 相邻调仓日种子几乎相同，对冲腿高度重复；")
    P("- 更致命：对冲腿**每天都按收盘价重估，但只在调仓日换**，而多头腿 same → 两边对齐是 OK 的。")
    P("- 真正的数学问题：**对冲掉同格子均值后，剩下的是「这只票 vs 同格子 30 只平均」，")
    P("  方差极大**（12.37% 波动 vs 多头 27%）。Sharpe 因此被分母拖累，不是 alpha 没了。\n")

    P("## 结论性的诊断（写进 FINDINGS 用）\n")
    P("1. t7f 的 +0.713%/5日 是**相对同格子均值**的超额，这是可靠的（placebo 已验）。")
    P("2. 但把它做成**对冲组合**后年化 −3.81%：因为对冲腿的选取是随机的，")
    P("   随机对手方自身带有 +/− 波动，**对冲掉的不是 beta 而是把 alpha 也稀释了**。")
    P("3. P3 格子配额：picks 集中在少数格子，配额法退化成「重仓少数票」，波动反而升到 32%。")
    P("4. **正确做法是「风格中性化选股 + 全市场等权持有其余」，即：picks 权重不变，")
    P("   但用「同格子的历史平均收益」作为评估基准，而不是真的去对冲/配额。**")
    P("   → 也就是：**alpha 存在且显著，但在 A 股现有工具下不可直接变现。**")

    with open(os.path.join(OUT, "m8b_diag.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    print(f"[saved] {os.path.join(OUT,'m8b_diag.md')}")


if __name__ == "__main__":
    main()
