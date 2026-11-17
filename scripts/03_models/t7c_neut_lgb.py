# -*- coding: utf-8 -*-
"""t7c_neut_lgb.py —— 用「风格中性化后的超额收益」当标签重训 LGBM，然后回测

动机：t7b 显示 42 个非风格因子在剔除 4 个风格因子后，|IC| 从 0.044 掉到 0.019（掉 58%）。
但这个 0.019 里仍有 35/46 个因子 |t|>2。问题是：**这剩下的一点点 IC 能不能变成钱？**

做法：
  L1  标签 y_neut = 1{ 逐日截面回归 excess5 ~ log_amt + log_price + vol_20 + turn_ma_20 的残差 > 0 }
  L2  用 46 因子训练 LGBM（时间外推同 t4：train 2018-22 / valid 2023 / test 2024-25）
  L3  三套基准回测：全市场等权 / 中证1000 / 风格中性化基准
      其中「风格中性化基准」= 每个调仓日先选与 picks 在 4 个风格因子上匹配的对照组，
      取对照组等权收益。这才是「扣除风格后还剩多少」的直接检验。

产出：output/m7c_neut_lgb.md
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
    P(f"# 风格中性化标签重训\n\n- 面板 {panel.shape}，46 特征，4 风格因子\n")

    # ---- L1 标签：分块算残差 ----
    CH = 400000
    ex = np.empty(len(panel))
    keep = []
    for i in range(0, len(panel), CH):
        blk = panel.iloc[i:i + CH]
        g = pd.factorize(blk["date"])[0]
        base = int(g.min())
        X = blk[STYLE].to_numpy(dtype=np.float64)
        y = blk["excess5"].to_numpy(dtype=np.float64)
        r = resid_block(y, X, g - base, len(STYLE))
        ex[i:i + len(blk)] = r
        keep.append(i)
        gc.collect()
        print(f"  resid block {i//CH+1}/{(len(panel)+CH-1)//CH}", flush=True)
    panel["ex_neut"] = ex
    panel["y_neut"] = (panel["ex_neut"] > 0).astype(int)
    P(f"- y_neut 正样本率 {panel['y_neut'].mean():.4f}（y_exc 是 {(panel['fwd5']>panel['mkt_fwd5']).mean():.4f}）")

    # ---- L2 训练 ----
    tr = panel[(panel["date"] >= "2018-01-01") & (panel["date"] <= "2022-12-31")]
    va = panel[(panel["date"] >= "2023-01-01") & (panel["date"] <= "2023-12-31")]
    te = panel[(panel["date"] >= "2024-01-01")]
    pars = dict(objective="binary", metric="auc", learning_rate=0.03, num_leaves=63,
                min_data_in_leaf=500, feature_fraction=0.6, bagging_fraction=0.7, bagging_freq=5,
                lambda_l2=5.0, num_threads=4, verbosity=-1, seed=42)
    bst = lgb.train(pars, lgb.Dataset(tr[fc], tr["y_neut"]), num_boost_round=1500,
                    valid_sets=[lgb.Dataset(va[fc], va["y_neut"])],
                    callbacks=[lgb.early_stopping(80, verbose=False), lgb.log_evaluation(0)])
    bst.save_model(os.path.join(ROOT, "models", "lgb_y_neut.txt"), num_iteration=bst.best_iteration)
    P(f"\n## L2 训练结果（best_iter={bst.best_iteration}）\n")
    P("| 集合 | n | AUC | IC(对 ex_neut) | IC(对 excess5) | t_stride5(对 ex_neut) |")
    P("|---|---|---|---|---|---|")
    # ⚠️ daily_spearman_ic 在 t4_lgb 里，不在 t5_backtest 里
    icf = t4_lgb.daily_spearman_ic
    for nm, s in (("train", tr), ("valid", va), ("test", te)):
        pr = bst.predict(s[fc], num_iteration=bst.best_iteration)
        auc = roc_auc_score(s["y_neut"], pr)
        i1 = icf(s["date"].values, pr, s["ex_neut"].to_numpy())
        i2 = icf(s["date"].values, pr, s["excess5"].to_numpy())
        s5 = i1.iloc[::-1][::5]
        P(f"| {nm} | {len(s):,} | {auc:.4f} | {i1.mean():+.4f} | {i2.mean():+.4f} | "
          f"{s5.mean()/s5.std()*np.sqrt(len(s5)):+.2f} (n={len(s5)}) |")

    # ---- L3 回测 ----
    panel["score_neut"] = np.nan
    idx_te = panel.index[panel["date"] >= "2024-01-01"]
    panel.loc[idx_te, "score_neut"] = bst.predict(panel.loc[idx_te, fc], num_iteration=bst.best_iteration)

    S, E = "2024-01-01", "2025-11-10"
    bt = panel[(panel["date"] >= S) & (panel["date"] <= E)].copy()
    bt["lim"] = np.where(bt["board"].isin(["GEM", "STAR"]), 0.20, 0.10)

    def perf(nav):
        r = nav.pct_change().dropna()
        if len(r) < 20:
            return {}
        ann = (1 + r).prod() ** (252 / len(r)) - 1
        vol = r.std() * np.sqrt(252)
        return {"ann": ann, "vol": vol, "sharpe": (ann - T5.RF) / vol,
                "mdd": (nav / nav.cummax() - 1).min()}

    # 风格匹配基准：每次调仓日，对 picks 的 4 个风格因子均值找最近邻的同数量对照组
    def style_matched_nav(score_col, K=30, hold=5):
        dates = np.array(sorted(bt["date"].unique()))
        by = {d: g.set_index("code") for d, g in bt.groupby("date")}
        navs, cash = [], 1.0
        hold_px, weights = {}, {}
        ctrl_prev = None
        for i, dt in enumerate(dates):
            g = by[dt]
            if weights:
                rets = {}
                for c, w in weights.items():
                    p0 = hold_px.get(c, np.nan)
                    p1 = g["adj_close"].get(c, np.nan)
                    rets[c] = (p1 / p0 - 1) if (np.isfinite(p0) and np.isfinite(p1) and p0 > 0) else 0.0
                cash *= (1 + sum(w * rets[c] for c, w in weights.items()))
            navs.append((dt, cash))
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
            # 标准化后欧氏距离，取最近 K 只
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
        return pd.Series(dict(navs)).sort_index()

    # 全市场等权
    d = bt[["date", "code", "adj_close"]].sort_values(["code", "date"])
    d["r"] = d.groupby("code")["adj_close"].pct_change()
    ew = (1 + d.groupby("date")["r"].mean().fillna(0)).cumprod()

    P(f"\n## L3 回测（{S} ~ {E}，Top30 / 持有5日 / 单边15bp）\n")
    P("| 策略 | 年化 | 波动 | Sharpe | 最大回撤 | 相对全市场等权 |")
    P("|---|---|---|---|---|---|")
    pe = perf(ew)
    P(f"| 全市场等权（基准） | {pe['ann']:.2%} | {pe['vol']:.2%} | {pe['sharpe']:.3f} | {pe['mdd']:.1%} | — |")
    for lb, col in (("LGB_y_exc（旧）", "score_y_exc"), ("LGB_y_neut（新）", "score_neut")):
        if col not in bt.columns or bt[col].isna().all():
            continue
        nav, _ = T5.backtest(bt, col, topk=30, hold=5, cost=0.0015, start=S, end=E)
        p = perf(nav)
        br = ew.reindex(nav.index).ffill()
        br = br / br.iloc[0]
        rb = br.pct_change().fillna(0)
        rr = nav.pct_change().fillna(0)
        excess = (1 + (rr - rb)).prod() ** (252 / len(rr)) - 1
        P(f"| {lb} | {p['ann']:.2%} | {p['vol']:.2%} | {p['sharpe']:.3f} | {p['mdd']:.1%} | {excess:+.2%} |")
        nav.to_frame("nav").to_csv(os.path.join(OUT, f"m7c_nav_{lb.split('（')[0]}.csv"))

    # 风格匹配基准
    P("\n### 风格匹配基准（关键）\n")
    P("> 每个调仓日，在剩余股票里找「4 个风格因子最接近 picks 均值」的 30 只做对照组。\n")
    P("> 若模型真有选股能力，它在风格匹配的对照组上也应跑赢；若跑不赢，说明赚的全是风格。\n")
    nav_m = style_matched_nav("score_neut", 30, 5)
    nav_m2 = style_matched_nav("score_y_exc", 30, 5)
    P("| 组合 | 年化 | Sharpe |")
    P("|---|---|---|")
    for nm, nv in (("风格匹配对照组（y_neut picks）", nav_m), ("风格匹配对照组（y_exc picks）", nav_m2)):
        p = perf(nv)
        if p:
            P(f"| {nm} | {p['ann']:.2%} | {p['sharpe']:.3f} |")
            nv.to_frame("nav").to_csv(os.path.join(OUT, f"m7c_ctrl_{nm[:12]}.csv"))

    with open(os.path.join(OUT, "m7c_neut_lgb.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    print(f"[saved] {os.path.join(OUT,'m7c_neut_lgb.md')}")


if __name__ == "__main__":
    main()
