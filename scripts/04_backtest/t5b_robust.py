# -*- coding: utf-8 -*-
"""t5b_robust.py —— 回测稳健性：参数敏感性 + 等风险（波动缩放）+ 涨停污染拆解

跑三件事（都不改模型，只改组合构造）：
  S1 TopK × hold 网格：K∈{10,20,30,50,100}, hold∈{5,10,20}，看 Sharpe / 超额是否稳定
  S2 波动缩放：用 t-1 之前的 20 日实现波动（目标 20% 年化）对组合做杠杆缩放（上限 2 倍），
     即用「同样的信号」降低组合波动，看能否追上基准的风险调整收益
  S3 涨停来源：统计模型持仓里「下一期含涨停」的比例，与全市场对比

注意：S2 的杠杆在 A 股散户不可行（融资成本 + 50 万门槛），仅作为「信号有没有 alpha」的检验，
      不作为策略推荐。目标是把「IC 显著但 Sharpe 不够」这件事定位清楚。

产出：output/m5b_grid.csv, output/m5b_volscale.csv, output/m5b_summary.md
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


def load_bench(name, start, end):
    p = os.path.join(IDX, f"{name}.parquet")
    d = pd.read_parquet(p).sort_values("date")
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


def perf(nav, bench=None):
    r = nav.pct_change().dropna()
    if len(r) < 20:
        return {}
    ann = (1 + r).prod() ** (252 / len(r)) - 1
    vol = r.std() * np.sqrt(252)
    out = {"ann_ret": ann, "ann_vol": vol, "sharpe": (ann - RF) / vol if vol > 0 else np.nan,
           "mdd": (nav / nav.cummax() - 1).min()}
    if bench is not None:
        br = bench.pct_change().reindex(r.index).fillna(0)
        bann = (1 + br).prod() ** (252 / len(r)) - 1
        out["bench_ann"] = bann
        out["excess_ann"] = ann - bann
        tr = r - br
        out["ir"] = tr.mean() / tr.std() * np.sqrt(252) if tr.std() > 0 else np.nan
    return out


def main():
    panel = T5.load_panel()
    print(f"[t5b] panel {panel.shape}")
    import lightgbm as lgb
    import t4_lgb
    fc = t4_lgb.feat_cols(panel)
    scores = {}
    for lb in ("y_raw", "y_exc"):
        mp = os.path.join(ROOT, "models", f"lgb_{lb}.txt")
        if not os.path.exists(mp):
            continue
        bst = lgb.Booster(model_file=mp)
        panel[f"score_{lb}"] = bst.predict(panel[fc])
        scores[f"LGB_{lb}"] = f"score_{lb}"

    S, E = "2024-01-01", "2026-12-31"
    bm = {n: load_bench(n, S, E) for n in ("000300", "000905", "000852")}
    bm["EW_ALL"] = load_ew(panel, S, E)

    # ---------------- S1 网格 ----------------
    rows = []
    for name, col in scores.items():
        for K in (10, 20, 30, 50, 100):
            for H in (5, 10, 20):
                nav, _ = T5.backtest(panel, col, topk=K, hold=H, cost=0.0015, start=S, end=E)
                p = perf(nav, bm["EW_ALL"])
                if not p:
                    continue
                rows.append({"model": name, "topk": K, "hold": H, **p})
                print(f"  {name:10s} K={K:3d} H={H:2d}  ann={p['ann_ret']:+.2%} "
                      f"sharpe={p['sharpe']:+.3f} exc_vs_EW={p['excess_ann']:+.2%}", flush=True)
    G = pd.DataFrame(rows)
    G.to_csv(os.path.join(OUT, "m5b_grid.csv"), index=False)

    # ---------------- S2 波动缩放 ----------------
    # 目标年化波动 20%；用 t-1 之前 20 日实现波动估计（严格前定）
    vs_rows, vs_navs = [], {}
    for name, col in scores.items():
        nav, _ = T5.backtest(panel, col, topk=30, hold=5, cost=0.0015, start=S, end=E)
        r = nav.pct_change().fillna(0)
        rv = r.rolling(20).std().shift(1) * np.sqrt(252)
        lev = (0.20 / rv).clip(0.0, 2.0).shift(0)
        # 未来 5 日固定杠杆（与调仓频率一致，避免日内反复调杠杆）
        lev5 = lev.where(np.arange(len(lev)) % 5 == 0).ffill()
        r_sc = r * lev5.fillna(1.0)
        nav_sc = (1 + r_sc).cumprod()
        vs_navs[name] = nav_sc
        for label, s in ((f"{name}_raw", nav), (f"{name}_volscale20", nav_sc)):
            for bnm, b in bm.items():
                p = perf(s, b)
                if p:
                    vs_rows.append({"model": label, "bench": bnm, **p})
        p0 = perf(nav_sc, bm["EW_ALL"])
        print(f"  [volscale] {name}: ann={p0['ann_ret']:+.2%} vol={p0['ann_vol']:.2%} "
              f"sharpe={p0['sharpe']:+.3f} exc_vs_EW={p0['excess_ann']:+.2%}", flush=True)
        nav_sc.to_frame("nav").to_csv(os.path.join(OUT, f"m5b_nav_{name}_volscale.csv"))
    V = pd.DataFrame(vs_rows)
    V.to_csv(os.path.join(OUT, "m5b_volscale.csv"), index=False)

    # ---------------- S3 涨停来源 ----------------
    all_zt = panel["fwd5_has_zt"].mean()
    print(f"\n[S3] 全市场「未来5日含涨停」比例 = {all_zt:.2%}")
    # 用每次调仓的 pick 统计
    lines = [f"# 回测稳健性\n", f"- 区间 {S} ~ {E}", f"- 全市场未来5日含涨停比例: {all_zt:.2%}\n",
             "\n## S1 TopK × 持有期（超额相对全市场等权）\n",
             G.pivot_table(index=["model", "topk"], columns="hold",
                           values=["ann_ret", "sharpe", "excess_ann"]).round(4).to_string(),
             "\n\n## S2 波动缩放（目标 20% 年化）\n",
             V[V["bench"] == "EW_ALL"].round(4).to_string(index=False),
             "\n\n## 基准明细\n"]
    for n, b in bm.items():
        p = perf(b)
        lines.append(f"- {n}: ann={p['ann_ret']:.2%} vol={p['ann_vol']:.2%} sharpe={p['sharpe']:.3f} mdd={p['mdd']:.2%}")
    with open(os.path.join(OUT, "m5b_summary.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"[saved] {os.path.join(OUT, 'm5b_summary.md')}")


if __name__ == "__main__":
    main()
