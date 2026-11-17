# -*- coding: utf-8 -*-
"""t5c_episodes.py —— 「一部分样本吃掉全部收益」的量化 + 分布内对照

动机（本轮发现的第四类坑，且是最重要的一类）：
  K=30/H=5 的 y_exc 策略全期 ann=+5.28%、Sharpe=0.136，看起来完全失败；
  但拆开时段：2024-01~06 ann=−36.5%、2024-07 之后 ann=+25.7% Sharpe=1.10。
  → 单个时段就能反转结论。必须回答：**这是运气还是分布会重演？**

做法：
  E1 情景表：把历史切成若干「整段」子区间，分别看每个模型的表现，并给出基准同期表现。
     判据：如果模型在多数子区间的超额符号一致，就不是单点运气；如果只有个别区间好，就是运气。
  E2 分布内对照（关键）：不用模型，直接用「当日涨幅最大的 K 只」当信号（这是模型很可能在学的东西），
     跑同样的回测。若「追当日涨停/大涨」在同期也赚钱，则模型的收益可能只是这个简单效应的代理。
  E3 剔除涨停样本的重训回测已在 t4/t5 做（y_exc_nzt），这里补：把「买入日或前 5 日涨停」的候选直接从
     可买池里剔除，看 Top-K 还剩不剩 alpha（可执行版剔除）。

产出：output/m5c_episodes.md
"""
import os, sys
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
OUT = os.path.join(ROOT, "output")
IDX = os.path.join(ROOT, "data", "raw", "index")
import t5_backtest as T5

RF = T5.RF
L = []


def P(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    L.append(s)


def perf(nav, bench=None):
    r = nav.pct_change().dropna()
    if len(r) < 20:
        return {}
    ann = (1 + r).prod() ** (252 / len(r)) - 1
    vol = r.std() * np.sqrt(252)
    out = {"ann": ann, "vol": vol, "sharpe": (ann - RF) / vol if vol > 0 else np.nan,
           "mdd": (nav / nav.cummax() - 1).min()}
    if bench is not None:
        br = bench.pct_change().reindex(r.index).fillna(0)
        bann = (1 + br).prod() ** (252 / len(r)) - 1
        out["bench_ann"] = bann
        out["excess"] = ann - bann
    return out


def load_bench(name, start, end):
    d = pd.read_parquet(os.path.join(IDX, f"{name}.parquet")).sort_values("date")
    d["date"] = pd.to_datetime(d["date"])
    s = d.set_index("date")["close"]
    s = s[(s.index >= start) & (s.index <= end)].ffill()
    return s / s.iloc[0]


def load_ew(panel, start, end):
    d = panel[(panel["date"] >= start) & (panel["date"] <= end)][["date", "code", "adj_close"]]
    d = d.sort_values(["code", "date"])
    d["r"] = d.groupby("code")["adj_close"].pct_change()
    ew = d.groupby("date")["r"].mean().fillna(0)
    return (1 + ew).cumprod()


def main():
    panel = T5.load_panel()
    import lightgbm as lgb
    import t4_lgb
    fc = t4_lgb.feat_cols(panel)
    scores = {}
    for lb in ("y_raw", "y_exc"):
        mp = os.path.join(ROOT, "models", f"lgb_{lb}.txt")
        if os.path.exists(mp):
            bst = lgb.Booster(model_file=mp)
            panel[f"score_{lb}"] = bst.predict(panel[fc])
            scores[f"LGB_{lb}"] = f"score_{lb}"
    # E2 的朴素对照信号：当日涨幅（ret_1 是截面 z-score，单调等价于涨幅排名）
    scores["NAIVE_mom1"] = "ret_1"
    scores["NAIVE_negvol"] = "vol_20"   # 低波动（vol_20 取负方向由 nlargest 决定，见下）
    panel["neg_vol_20"] = -panel["vol_20"]
    scores["NAIVE_negvol"] = "neg_vol_20"

    P("# 情景与分布内对照\n")

    # ---------- E1 情景表 ----------
    P("## E1 不同子区间（Top30 / 持有5日 / 单边15bp）\n")
    episodes = [("2024-01-01", "2024-06-30", "2024H1（小盘暴跌+微盘踩踏）"),
                ("2024-07-01", "2025-11-10", "2024H2~2025 全程"),
                ("2025-01-01", "2025-11-10", "2025 年"),
                ("2024-01-01", "2025-11-10", "全期")]
    P("| 区间 | 说明 | 模型 | 年化 | 波动 | Sharpe | 最大回撤 | 全市场等权同期 | 超额 |")
    P("|---|---|---|---|---|---|---|---|---|")
    for s, e, nm in episodes:
        ew = load_ew(panel, s, e)
        pe = perf(ew)
        for name, col in scores.items():
            if not name.startswith("LGB"):
                continue
            nav, _ = T5.backtest(panel, col, topk=30, hold=5, cost=0.0015, start=s, end=e)
            p = perf(nav, ew)
            if not p:
                continue
            P(f"| {s}~{e} | {nm} | {name} | {p['ann']:.2%} | {p['vol']:.2%} | {p['sharpe']:.3f} | "
              f"{p['mdd']:.1%} | {p['bench_ann']:.2%} | {p['excess']:+.2%} |")
        P(f"| {s}~{e} | {nm} | **全市场等权** | {pe['ann']:.2%} | {pe['vol']:.2%} | {pe['sharpe']:.3f} | {pe['mdd']:.1%} | — | — |")

    # ---------- E2 分布内对照 ----------
    P("\n## E2 分布内对照：朴素信号 vs 模型（同一套回测规则）\n")
    P("> `NAIVE_mom1` = 当日涨幅最高（等价于追涨/追涨停）；`NAIVE_negvol` = 20 日波动最低。\n")
    P("| 区间 | 信号 | 年化 | Sharpe | 超额 vs EW |")
    P("|---|---|---|---|---|")
    for s, e, nm in episodes:
        ew = load_ew(panel, s, e)
        for name, col in scores.items():
            if not name.startswith("NAIVE"):
                continue
            nav, _ = T5.backtest(panel, col, topk=30, hold=5, cost=0.0015, start=s, end=e)
            p = perf(nav, ew)
            if not p:
                continue
            P(f"| {nm} | {name} | {p['ann']:.2%} | {p['sharpe']:.3f} | {p['excess']:+.2%} |")

    # ---------- E3 可执行版剔涨停 ----------
    P("\n## E3 可执行版剔除：候选池直接剔除「最近 5 日内涨停过」的股票\n")
    P("> t4 的 y_exc_nzt 是「训练时剔样本」；这里是「回测时剔候选」，更严格（含买入当日）。\n")
    panel["zt_recent5"] = panel.groupby("code")["is_zt"].transform(lambda s: s.rolling(5).max())
    for s, e, nm in (("2024-01-01", "2025-11-10", "全期"), ("2025-01-01", "2025-11-10", "2025")):
        ew = load_ew(panel, s, e)
        for name in ("LGB_y_raw", "LGB_y_exc"):
            col = scores.get(name)
            if not col:
                continue
            for flt in (None, "zt_recent5"):
                nav, _ = T5.backtest(panel, col, topk=30, hold=5, cost=0.0015, start=s, end=e, filter_col=flt)
                p = perf(nav, ew)
                if not p:
                    continue
                tag = "剔除近期涨停" if flt else "不剔除"
                P(f"- {nm} {name} {tag}: ann={p['ann']:.2%} sharpe={p['sharpe']:.3f} 超额={p['excess']:+.2%}")

    with open(os.path.join(OUT, "m5c_episodes.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    print(f"[saved] {os.path.join(OUT,'m5c_episodes.md')}")


if __name__ == "__main__":
    main()
