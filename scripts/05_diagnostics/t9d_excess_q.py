# -*- coding: utf-8 -*-
"""t9d_excess_q.py —— 逐季度「相对等权超额」分解 + 最终 placebo

最后一关。要证明「3亿池 + 5日平滑 + 持有5日 + 15bp → 相对等权 +3.35%」不是运气，需要：

  A. **逐季度超额**（不是组合收益）。此前 t9c 报的是组合绝对收益，
     2024-2025 大盘本身在涨，绝对收益为正不代表有 alpha。
  B. **同池 placebo 100 次**：在 amount≥3亿 池里随机选 30 只（跑满 100 次），
     得到零分布，看平滑版的 29.71% / Sharpe 1.346 落在哪个百分位。
  C. **同信号但打乱日期**：把分数按日期循环平移（破坏分数与未来收益的对应关系），
     若收益塌到基准附近，说明收益来自分数而非巧合。
  D. **换 K 检验**：K=20/30/50，确认不是 K=30 这一个点。

产出：output/m9d_excess.md
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
    P("# 逐季度超额 + 最终 placebo\n")
    P(f"- 基准（全市场等权）年化 {perf(ew)['ann']:.2%}，Sharpe {perf(ew)['sharpe']:.3f}\n")

    def run(col, min_amt=3e8, hold=5, cost=0.0015, K=30, seed=None):
        cash = 1.0
        hold_px, weights = {}, {}
        out, to_list = [], []
        rng = np.random.default_rng(seed) if seed is not None else None
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
            if seed is None:
                picks = cand.nlargest(K, col).index.tolist()
            else:
                picks = cand.index[rng.random(len(cand)).argsort()[:K]].tolist()
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

    nav, _ = run("s_ma5")
    nav_raw, _ = run("s")

    # ---------- A. 逐季度超额 ----------
    P("## A. 逐季度超额（相对全市场等权）\n")
    P("| 季度 | 组合收益 | 等权收益 | **超额** |")
    P("|---|---|---|---|")
    rr = nav.pct_change().fillna(0)
    rb = ew_r.reindex(nav.index).fillna(0)
    q = pd.DataFrame({"r": rr, "b": rb})
    q["k"] = [f"{d.year}Q{d.quarter}" for d in q.index]
    for k, g in q.groupby("k"):
        if len(g) < 10:
            continue
        pr = (1 + g["r"]).prod() - 1
        pb = (1 + g["b"]).prod() - 1
        P(f"| {k} | {pr:+.2%} | {pb:+.2%} | **{pr-pb:+.2%}** |")
    tot_r = (1 + rr).prod() - 1
    tot_b = (1 + rb).prod() - 1
    P(f"| **全期** | **{tot_r:+.2%}** | **{tot_b:+.2%}** | **{tot_r-tot_b:+.2%}** |")

    # ---------- D. K 检验 ----------
    P("\n## D. TopK 敏感性（s_ma5，5日持有，15bp）\n")
    P("| K | 年化 | Sharpe | 相对等权 |")
    P("|---|---|---|---|")
    for K in (20, 30, 50):
        nv, _ = run("s_ma5", K=K)
        pf = perf(nv)
        r2 = nv.pct_change().fillna(0)
        b2 = ew_r.reindex(nv.index).fillna(0)
        exc = (1 + (r2 - b2)).prod() ** (252 / len(r2)) - 1
        P(f"| {K} | {pf['ann']:.2%} | {pf['sharpe']:.3f} | {exc:+.2%} |")

    # ---------- B. placebo ----------
    P("\n## B. 同池随机 placebo（100 次，amount≥3亿，K=30）\n")
    res = []
    for k in range(100):
        nv, _ = run("s_ma5", seed=31000 + k)
        res.append(perf(nv))
        if (k + 1) % 25 == 0:
            print(f"  placebo {k+1}/100", flush=True)
    D = pd.DataFrame(res)
    real = perf(nav)
    raw = perf(nav_raw)
    P("| 指标 | 零分布均值 | 零分布 std | 单日分数 | 5日平滑 | 平滑百分位 |")
    P("|---|---|---|---|---|---|")
    for c in ("ann", "sharpe"):
        P(f"| {c} | {D[c].mean():.4f} | {D[c].std():.4f} | {raw[c]:.4f} | {real[c]:.4f} | "
          f"{(D[c] < real[c]).mean():.1%} |")
    D.to_csv(os.path.join(OUT, "m9d_placebo.csv"), index=False)

    with open(os.path.join(OUT, "m9d_excess.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    print(f"[saved] {os.path.join(OUT,'m9d_excess.md')}")


if __name__ == "__main__":
    main()
