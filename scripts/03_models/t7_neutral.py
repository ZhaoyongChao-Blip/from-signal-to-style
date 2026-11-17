# -*- coding: utf-8 -*-
"""t7_neutral.py —— 风格中性化：剔掉「小盘/低价/低波动/低流动」之后，技术因子还剩多少？

背景（FINDINGS.md 第 4 节）：
  t5d 归因显示 LGB 组合 β_全市场等权 = 1.8，剔除市场 beta 后 α 的 t 值只有 0.03 / −0.63；
  D4 分十组显示 Q10 的中位成交额只有 0.37 亿、中位流通市值 27.8 亿。
  ⟹ 模型很可能只是「小盘+低价+低流动」这条风格的最优组合器。

本脚本要回答的问题：
  **把风格彻底剥掉之后，46 个技术因子还有没有预测力？**

做法（三层，逐层更严）：
  N1 「逐日截面回归取残差」当标签
     对每个调仓日 t，横截面回归  excess5 ~ log_amt + log_price + vol_20 + turn_ma_20
     取残差 z = 风格中性化后的超额收益，用 z 的符号（或连续值）当学习目标。
  N2 「特征中性化」：把 46 个因子各自对上述 4 个风格因子做截面回归取残差，
     再用原标签训练，看 AUC/IC 还剩多少。（这检验的是「因子里的风格成分有多少」）
  N3 「回测时风格中性」：TopK 选股后，在 4 个风格因子上做多空配对（组内再等权），
     看组合对全市场等权的超额是否还在。

产出：output/m7_neutral.md + output/m7_*.csv

⚠️ 口径纪律：
  · 风格因子必须**纯前定**：log_amt / log_price / vol_20 / turn_ma_20 全部是 t 日收盘可见的。
  · 残差的截面均值天然为 0，所以残差的「正样本率」≈50%，与 y_exc 43% 不可直接比。
"""
import os, sys
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
OUT = os.path.join(ROOT, "output")
L = []

STYLE = ["log_amt", "log_price", "vol_20", "turn_ma_20"]


def P(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    L.append(s)


def cross_resid(y, X, g, style=STYLE):
    """逐日（g 为日期 codes）截面回归 y ~ 1 + X，返回残差。向量化、按组做。
    X 的列数必须等于 len(style)（默认 4）。"""
    y = np.asarray(y, dtype=np.float64)
    X = np.asarray(X, dtype=np.float64)
    nmiss = X.shape[1]
    assert nmiss == len(style), f"X 有 {nmiss} 列，STYLE 是 {len(style)} 个"
    ok = np.isfinite(y) & np.isfinite(X).all(axis=1)
    g = np.asarray(g)
    n = int(g.max()) + 1
    k = 1 + nmiss
    A = np.zeros((n, k, k))
    b = np.zeros((n, k))
    okf = ok.astype(np.float64)
    one = np.ones(len(y))
    # 构造 Xa = [1, X]
    Xa = np.column_stack([one, X])
    Xa[~ok] = 0.0
    yy = np.where(ok, y, 0.0)
    for i in range(k):
        for j in range(k):
            A[:, i, j] = np.bincount(g, weights=Xa[:, i] * Xa[:, j], minlength=n)
        b[:, i] = np.bincount(g, weights=Xa[:, i] * yy, minlength=n)
    # 加微小岭项避免奇异
    A[:, np.arange(k), np.arange(k)] += 1e-8 * np.maximum(np.bincount(g, weights=okf, minlength=n), 1)[:, None]
    beta = np.linalg.solve(A, b[:, :, None])[:, :, 0]
    resid = yy - (Xa * beta[g]).sum(axis=1)
    resid[~ok] = np.nan
    return resid


def daily_ic(dates, pred, ret, min_n=20, stride=1):
    d = pd.DataFrame({"d": dates, "p": pred, "r": ret})
    d = d[np.isfinite(d["r"]) & np.isfinite(d["p"])]
    if len(d) == 0:
        return pd.Series(dtype=float)
    d["pr"] = d.groupby("d")["p"].rank()
    d["rr"] = d.groupby("d")["r"].rank()
    d["pr"] -= d.groupby("d")["pr"].transform("mean")
    d["rr"] -= d.groupby("d")["rr"].transform("mean")
    d["num"] = d["pr"] * d["rr"]
    g = d.groupby("d")
    num = g["num"].sum()
    den = np.sqrt(g["pr"].apply(lambda s: (s ** 2).sum()) * g["rr"].apply(lambda s: (s ** 2).sum()))
    ic = (num / den)[g.size() >= min_n].dropna()
    if stride > 1:
        idx = np.array(sorted(ic.index))
        ic = ic.loc[idx[::-1][::stride][::-1]]
    return ic


def main():
    import lightgbm as lgb
    from sklearn.metrics import roc_auc_score
    import t4_lgb

    panel = pd.read_parquet(os.path.join(ROOT, "data", "proc", "panel.parquet"))
    panel["date"] = pd.to_datetime(panel["date"])
    panel = panel[panel["has_fwd5"] == 1].reset_index(drop=True)
    df = t4_lgb.load()
    fc = t4_lgb.feat_cols(df)
    P(f"# 风格中性化\n\n- 面板 {df.shape}\n- 46 因子 + 4 个风格因子 {STYLE}\n")

    # ---------- 先看风格因子本身对 excess5 的解释力 ----------
    g = pd.factorize(df["date"])[0]
    P("## N0 风格因子对超额收益的解释力（逐日截面回归）\n")
    y = df["excess5"].to_numpy(dtype=np.float64)
    Xs = df[STYLE].to_numpy(dtype=np.float64)
    resid = cross_resid(y, Xs, g)
    r2 = 1 - np.nanvar(resid) / np.nanvar(y)
    P(f"- 4 个风格因子合计 R² = **{r2:.4f}**（单日截面平均，样本内）")
    P(f"- 残差标准差 {np.nanstd(resid):.6f} vs 原始 {np.nanstd(y):.6f}\n")

    # ---------- N2 特征中性化：46 个因子对风格取残差，再看单因子 IC ----------
    P("## N2 特征中性化后的单因子 IC\n")
    P("> 做法：每个因子逐日对 4 个风格因子做截面回归取残差，再算残差与 fwd5 的逐日 Spearman IC。\n")
    from d2_fast import factor_daily_ic
    raw = factor_daily_ic(df, fc).set_index("factor")

    # 中性化版本：一次算完（用 bincount 批量）
    neut_ic = {}
    # ⚠️ `df[["date","fwd5"] + fc + STYLE]` 会把 4 个风格因子取两次（它们本身在 fc 里），
    # 建成 (n, 8) 的重复列矩阵 → 回归矩阵奇异、断言失败。必须用 drop_duplicates 后去重。
    cols = list(dict.fromkeys(["date", "fwd5"] + fc + STYLE))
    dsub = df[cols].copy()
    assert dsub[STYLE].shape[1] == len(STYLE), dsub[STYLE].shape
    g2 = pd.factorize(dsub["date"])[0]
    y2 = dsub["fwd5"].to_numpy(dtype=np.float64)
    X2 = dsub[STYLE].to_numpy(dtype=np.float64)
    for c in fc:
        xv = dsub[c].to_numpy(dtype=np.float64)
        r = cross_resid(xv, X2, g2)      # 因子对风格取残差
        # 注意：残差本身与 X2 正交，所以对 fwd5 的 IC 等价于「剔除风格的因子」的 IC
        ic = daily_ic(dsub["date"].values, r, y2)
        if len(ic) > 100:
            neut_ic[c] = (ic.mean(), ic.mean() / ic.std(), len(ic))
    N = pd.DataFrame([{"factor": k, "IC_neutral": v[0], "ICIR_neutral": v[1], "n_days_n": v[2]}
                      for k, v in neut_ic.items()]).set_index("factor")
    # ⚠️ 两个表都有 n_days 列，join 会报 overlap → 先把右边的改名
    cmp = raw.join(N, how="inner")
    cmp["IC_raw"] = cmp["IC"]
    cmp["衰减%"] = (1 - cmp["IC_neutral"] / cmp["IC_raw"]) * 100
    # ⚠️ 残差与风格因子正交，但 IC 是 Spearman，不是 Pearson：正交不等于秩无关，
    # 所以「衰减%」可能 >100% 或为负，都是正常现象，不能机械解读。
    cmp = cmp[["IC_raw", "IC_neutral", "衰减%", "ICIR_neutral"]].sort_values("IC_neutral", key=abs, ascending=False)
    P(cmp.head(15).round(4).to_string())
    P(f"\n- 原始 |IC| 均值: {cmp['IC_raw'].abs().mean():.4f}")
    P(f"- 中性化后 |IC| 均值: {cmp['IC_neutral'].abs().mean():.4f}")
    P(f"- 中位衰减: **{cmp['衰减%'].median():.1f}%**")
    cmp.to_csv(os.path.join(OUT, "m7_factor_ic_neutral.csv"))

    # ---------- N1 用中性化标签重训 ----------
    P("\n## N1 用「风格中性化后的超额收益」当标签重训 LGBM\n")
    # 标签：残差 z 的符号（连续残差用二分类便于和 y_exc 对比）
    d = df.copy()
    d["ex_neut"] = cross_resid(y, Xs, g)
    d["y_neut"] = (d["ex_neut"] > 0).astype(int)
    P(f"- 正样本率: {d['y_neut'].mean():.4f}（y_exc 是 {d['y_exc'].mean():.4f}）")

    TRAIN, VALID, TEST = ("2018-01-01", "2022-12-31"), ("2023-01-01", "2023-12-31"), ("2024-01-01", "2026-12-31")
    tr = d[(d["date"] >= TRAIN[0]) & (d["date"] <= TRAIN[1])]
    va = d[(d["date"] >= VALID[0]) & (d["date"] <= VALID[1])]
    te = d[(d["date"] >= TEST[0]) & (d["date"] <= TEST[1])]
    pars = dict(objective="binary", metric="auc", learning_rate=0.03, num_leaves=63,
                min_data_in_leaf=500, feature_fraction=0.6, bagging_fraction=0.7, bagging_freq=5,
                lambda_l2=5.0, num_threads=4, verbosity=-1, seed=42)
    bst = lgb.train(pars, lgb.Dataset(tr[fc], tr["y_neut"]), num_boost_round=1500,
                    valid_sets=[lgb.Dataset(va[fc], va["y_neut"])],
                    callbacks=[lgb.early_stopping(80, verbose=False), lgb.log_evaluation(0)])
    bst.save_model(os.path.join(ROOT, "models", "lgb_y_neut.txt"), num_iteration=bst.best_iteration)
    P(f"- best_iter = {bst.best_iteration}\n")
    P("| 集合 | n | AUC | IC(对 ex_neut) | IC(对 excess5) | t(无重叠) |")
    P("|---|---|---|---|---|---|")
    for nm, s in (("train", tr), ("valid", va), ("test", te)):
        pr = bst.predict(s[fc], num_iteration=bst.best_iteration)
        auc = roc_auc_score(s["y_neut"], pr)
        ic1 = daily_ic(s["date"].values, pr, s["ex_neut"].to_numpy())
        ic2o = daily_ic(s["date"].values, pr, s["excess5"].to_numpy(), stride=5)
        P(f"| {nm} | {len(s):,} | {auc:.4f} | {ic1.mean():+.4f} | "
          f"{daily_ic(s['date'].values, pr, s['excess5'].to_numpy()).mean():+.4f} | "
          f"{ic2o.mean()/ic2o.std()*np.sqrt(len(ic2o)):+.2f} (n={len(ic2o)}) |")

    with open(os.path.join(OUT, "m7_neutral.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    print(f"[saved] {os.path.join(OUT,'m7_neutral.md')}")


if __name__ == "__main__":
    main()
