# -*- coding: utf-8 -*-
"""t7d_neut_eval.py —— 只做评估（不重训），复用 t7c 已存盘的 lgb_y_neut.txt / lgb_y_exc.txt 等

t7c 在训练完成后崩在评估段（daily_spearman_ic 找错模块），模型已经存下来了。
本脚本：
  1. 复算 y_neut 标签（分块，与 t7c 完全一致）
  2. 用现存模型评估 AUC / IC（含 stride=5 无重叠 t 值）
  3. 回测：y_exc vs y_neut，相对全市场等权 + 风格匹配对照组
"""
import os, sys, gc
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


def resid_block(y, X, g, ncol):
    k = 1 + ncol
    n = int(g.max()) + 1
    ok = np.isfinite(y) & np.isfinite(X).all(axis=1)
    Xa = np.empty((len(y), k))
    Xa[:, 0] = 1.0
    Xa[:, 1:] = X
    Xa[~ok] = 0.0
    yy = np.where(ok, y, 0.0)
    A = np.zeros((n, k, k))
    b = np.zeros((n, k))
    for i in range(k):
        for j in range(i, k):
            v = np.bincount(g, weights=Xa[:, i] * Xa[:, j], minlength=n)
            A[:, i, j] = v
            A[:, j, i] = v
        b[:, i] = np.bincount(g, weights=Xa[:, i] * yy, minlength=n)
    cnt = np.bincount(g, weights=ok.astype(np.float64), minlength=n)
    A[:, np.arange(k), np.arange(k)] += 1e-6 * np.maximum(cnt, 1)[:, None]
    beta = np.linalg.solve(A, b[:, :, None])[:, :, 0]
    r = yy - (Xa * beta[g]).sum(axis=1)
    r[~ok] = np.nan
    return r


def main():
    import lightgbm as lgb
    import t4_lgb
    import t5_backtest as T5
    from sklearn.metrics import roc_auc_score

    panel = pd.read_parquet(os.path.join(ROOT, "data", "proc", "panel.parquet"))
    panel["date"] = pd.to_datetime(panel["date"])
    panel = panel[panel["has_fwd5"] == 1].sort_values("date", kind="stable").reset_index(drop=True)
    fc = t4_lgb.feat_cols(panel)

    # y_neut 标签
    ex = np.empty(len(panel))
    CH = 400000
    for i in range(0, len(panel), CH):
        blk = panel.iloc[i:i + CH]
        g = pd.factorize(blk["date"])[0]
        ex[i:i + len(blk)] = resid_block(blk["excess5"].to_numpy(dtype=np.float64),
                                         blk[STYLE].to_numpy(dtype=np.float64),
                                         g, len(STYLE))
        gc.collect()
    panel["ex_neut"] = ex
    panel["y_neut"] = (panel["ex_neut"] > 0).astype(int)

    P("# 风格中性化：评估与回测\n")
    P(f"- 面板 {panel.shape}；y_neut 正样本率 {panel['y_neut'].mean():.4f}\n")

    tr = panel[(panel["date"] >= "2018-01-01") & (panel["date"] <= "2022-12-31")]
    va = panel[(panel["date"] >= "2023-01-01") & (panel["date"] <= "2023-12-31")]
    te = panel[(panel["date"] >= "2024-01-01")]

    P("## L2 训练与评估\n")
    P("| 模型 | 集合 | n | AUC | IC(对目标) | IC(对 excess5) | t_stride5(对目标) |")
    P("|---|---|---|---|---|---|---|")
    icf = t4_lgb.daily_spearman_ic
    for lb, mp in (("y_exc", "lgb_y_exc.txt"), ("y_neut", "lgb_y_neut.txt")):
        p = os.path.join(ROOT, "models", mp)
        if not os.path.exists(p):
            P(f"| {lb} | — | 模型缺失 | | | |")
            continue
        bst = lgb.Booster(model_file=p)
        tgt = "y_exc" if lb == "y_exc" else "y_neut"
        ref = "mkt_fwd5" if lb == "y_exc" else "ex_neut"
        for nm, s in (("train", tr), ("valid", va), ("test", te)):
            pr = bst.predict(s[fc])
            ytrue = (s["fwd5"] > s["mkt_fwd5"]).astype(int) if lb == "y_exc" else s["y_neut"]
            auc = roc_auc_score(ytrue, pr)
            i1 = icf(s["date"].values, pr, s[ref].to_numpy())
            i2 = icf(s["date"].values, pr, s["excess5"].to_numpy())
            s5 = i1.iloc[::-1][::5]
            P(f"| {lb} | {nm} | {len(s):,} | {auc:.4f} | {i1.mean():+.4f} | {i2.mean():+.4f} | "
              f"{s5.mean()/s5.std()*np.sqrt(len(s5)):+.2f} (n={len(s5)}) |")

    # ---------- 回测 ----------
    P("\n## L3 回测（2024-01-01 ~ 2025-11-10，Top30 / 5 日 / 单边 15bp）\n")
    bt = panel[(panel["date"] >= "2024-01-01") & (panel["date"] <= "2025-11-10")].copy()
    bt["lim"] = np.where(bt["board"].isin(["GEM", "STAR"]), 0.20, 0.10)
    for lb, mp in (("y_exc", "lgb_y_exc.txt"), ("y_neut", "lgb_y_neut.txt")):
        p = os.path.join(ROOT, "models", mp)
        if not os.path.exists(p):
            continue
        bst = lgb.Booster(model_file=p)
        bt[f"s_{lb}"] = bst.predict(bt[fc])

    def perf(nav):
        r = nav.pct_change().dropna()
        if len(r) < 20:
            return {}
        ann = (1 + r).prod() ** (252 / len(r)) - 1
        vol = r.std() * np.sqrt(252)
        return {"ann": ann, "vol": vol, "sharpe": (ann - T5.RF) / vol,
                "mdd": (nav / nav.cummax() - 1).min()}

    d = bt[["date", "code", "adj_close"]].sort_values(["code", "date"])
    d["r"] = d.groupby("code")["adj_close"].pct_change()
    ew = (1 + d.groupby("date")["r"].mean().fillna(0)).cumprod()

    P("| 策略 | 年化 | 波动 | Sharpe | 最大回撤 | 相对全市场等权 |")
    P("|---|---|---|---|---|---|")
    pe = perf(ew)
    P(f"| 全市场等权（基准） | {pe['ann']:.2%} | {pe['vol']:.2%} | {pe['sharpe']:.3f} | {pe['mdd']:.1%} | — |")
    navs = {}
    for lb in ("y_exc", "y_neut"):
        col = f"s_{lb}"
        if col not in bt.columns:
            continue
        nav, _ = T5.backtest(bt, col, topk=30, hold=5, cost=0.0015, start="2024-01-01", end="2026-12-31")
        if len(nav) < 20:
            continue
        navs[lb] = nav
        p = perf(nav)
        rb = ew.reindex(nav.index).ffill().pct_change().fillna(0)
        rr = nav.pct_change().fillna(0)
        excess = (1 + (rr - rb)).prod() ** (252 / len(rr)) - 1
        P(f"| LGB_{lb} | {p['ann']:.2%} | {p['vol']:.2%} | {p['sharpe']:.3f} | {p['mdd']:.1%} | {excess:+.2%} |")
        nav.to_frame("nav").to_csv(os.path.join(OUT, f"m7d_nav_{lb}.csv"))

    # ---------- 风格匹配对照组 ----------
    P("\n## 风格匹配对照组（关键检验）\n")
    P("> 每个调仓日，在剩余候选里找「4 个风格因子标准化距离最近」的 30 只做对照组。\n")
    P("> 若模型有真选股能力，picks 应跑赢这个对照组。\n")
    dates = np.array(sorted(bt["date"].unique()))
    by = {dd: g.set_index("code") for dd, g in bt.groupby("date")}

    def matched(score_col, K=30, hold=5):
        cash = 1.0
        hold_px, weights = {}, {}
        out = []
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
                p = g["adj_close"].get(c, np.nan)
                if np.isfinite(p) and p > 0:
                    hold_px[c] = p
            if i % hold != 0 or i == len(dates) - 1:
                continue
            nxt = dates[i + 1]
            cand = g.dropna(subset=[score_col, "adj_close"] + STYLE)
            cand = cand[cand["amount"] > 2e7]
            if len(cand) < 2 * K:
                continue
            picks = cand.nlargest(K, score_col)
            mu = picks[STYLE].mean().to_numpy()
            rest = cand.drop(index=picks.index)
            sd = rest[STYLE].std().to_numpy() + 1e-9
            dist = np.sqrt((((rest[STYLE].to_numpy() - mu) / sd) ** 2).sum(axis=1))
            ctrl = rest.iloc[np.argsort(dist)[:K]]
            g2 = by[nxt]
            new = {}
            for c in ctrl.index:
                p = g2["adj_open"].get(c, np.nan)
                if np.isfinite(p) and p > 0:
                    new[c] = 1.0 / len(ctrl)
                    hold_px[c] = p
            weights = new
        return pd.Series(dict(out)).sort_index()

    P("| 组合 | 年化 | Sharpe |")
    P("|---|---|---|")
    for lb in ("y_exc", "y_neut"):
        col = f"s_{lb}"
        if col not in bt.columns:
            continue
        nv = matched(col)
        p = perf(nv)
        if p:
            P(f"| 风格匹配对照组（{lb} 的 picks） | {p['ann']:.2%} | {p['sharpe']:.3f} |")
            nv.to_frame("nav").to_csv(os.path.join(OUT, f"m7d_ctrl_{lb}.csv"))

    with open(os.path.join(OUT, "m7d_neut_eval.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    print(f"[saved] {os.path.join(OUT,'m7d_neut_eval.md')}")


if __name__ == "__main__":
    main()
