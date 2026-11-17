# -*- coding: utf-8 -*-
"""
t4_lgb.py —— LightGBM 选股

标签（三套，用来做「同一模型、不同标签」对照，这是本文最关键的设计）:
  A. y_raw   = fwd5 > 0                    （绝对上涨）
  B. y_exc   = fwd5 > 当日全市场等权 fwd5    （超额上涨）★ 用户要求的主标签
  C. y_exc_nzt = 同上，但剔除「未来5日内出现涨停」的样本（防涨停污染）

划分（滚动，不是一次性切）:
  train: 2018-01-01 ~ 2022-12-31
  valid: 2023-01-01 ~ 2023-12-31
  test : 2024-01-01 ~ 2025-11-10
  另外再做逐年 walk-forward（train ≤ T-2 年，test = T 年）用于稳健性

产出:
  models/lgb_<label>.txt
  output/m4_metrics.md      分类指标（AUC / 精度@TopK / IC）
  output/m4_importance.csv
"""
import os, json, argparse
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score

ROOT = os.path.dirname(os.path.abspath(__file__))
PROC = os.path.join(ROOT, "data", "proc")
OUT = os.path.join(ROOT, "output")
MOD = os.path.join(ROOT, "models")
for d in (OUT, MOD):
    os.makedirs(d, exist_ok=True)

TRAIN = ("2018-01-01", "2022-12-31")
VALID = ("2023-01-01", "2023-12-31")
TEST = ("2024-01-01", "2026-12-31")

DROP = {"date", "code", "board", "adj_close", "adj_open", "raw_open", "raw_close",
        "high", "low", "amount", "turnover", "volume", "factor",
        "fwd5", "fwd1", "fwd10", "mkt_fwd5", "excess5", "fwd5_has_zt", "has_fwd5",
        "lim", "is_zt", "is_dt"}
# 额外通过前缀屏蔽：指数未来收益(idx_*_fwd5) 与三个标签列都不能进特征
LEAK_PREFIX = ("idx_",)
LEAK_EXACT = {"y_raw", "y_exc", "y_exc_nzt", "y_perm"}


def daily_spearman_ic(dates, pred, ret, min_n=20):
    """逐日横截面 Spearman 相关，向量化（groupby.apply 在千万行上不可用）。"""
    d = pd.DataFrame({"d": dates, "p": pred, "r": ret})
    d = d[np.isfinite(d["r"]) & np.isfinite(d["p"])]
    if len(d) == 0:
        return pd.Series(dtype=float)
    # 组内排名 -> Pearson(rank) == Spearman
    g = d.groupby("d", sort=True)
    rp = g["p"].rank()
    rr = g["r"].rank()
    n = g["p"].transform("size")
    d = d.assign(rp=rp - rp.groupby(d["d"]).transform("mean"),
                 rr=rr - rr.groupby(d["d"]).transform("mean"),
                 n=n)
    num = (d["rp"] * d["rr"]).groupby(d["d"]).sum()
    den = np.sqrt((d["rp"] ** 2).groupby(d["d"]).sum() * (d["rr"] ** 2).groupby(d["d"]).sum())
    ic = (num / den).replace([np.inf, -np.inf], np.nan)
    cnt = d.groupby("d")["p"].size()
    ic = ic[cnt >= min_n].dropna()
    return ic


def load():
    p = os.path.join(PROC, "panel.parquet")
    df = pd.read_parquet(p)
    df["date"] = pd.to_datetime(df["date"])
    df = df[df["has_fwd5"] == 1].copy()

    df["y_raw"] = (df["fwd5"] > 0).astype(int)
    df["y_exc"] = (df["fwd5"] > df["mkt_fwd5"]).astype(int)
    # 剔涨停对照：样本层面剔除「未来5日含涨停」的观测
    df["y_exc_nzt"] = df["y_exc"]
    df.loc[df["fwd5_has_zt"] == 1, "y_exc_nzt"] = np.nan
    return df


def feat_cols(df):
    return [c for c in df.columns
            if c not in DROP and c not in LEAK_EXACT
            and not c.startswith(LEAK_PREFIX)
            and df[c].dtype.kind == "f"]


def train_one(df, label, params=None, verbose=True):
    fc = feat_cols(df)
    lab = df[label]
    # 只要求标签存在即可；因子缺失由 LightGBM 原生 NaN 处理（避免 844万×46 全量扫描）
    d = df[lab.notna()]
    tr = d[(d["date"] >= TRAIN[0]) & (d["date"] <= TRAIN[1])]
    va = d[(d["date"] >= VALID[0]) & (d["date"] <= VALID[1])]
    te = d[(d["date"] >= TEST[0]) & (d["date"] <= TEST[1])]

    p = dict(objective="binary", metric="auc", learning_rate=0.03, num_leaves=63,
             min_data_in_leaf=500, feature_fraction=0.6, bagging_fraction=0.7, bagging_freq=5,
             lambda_l2=5.0, num_threads=4, verbosity=-1, seed=42)
    if params:
        p.update(params)

    dtr = lgb.Dataset(tr[fc], tr[label], categorical_feature=[])
    dva = lgb.Dataset(va[fc], va[label], reference=dtr)
    evals = {}
    bst = lgb.train(p, dtr, num_boost_round=1500, valid_sets=[dtr, dva],
                    valid_names=["train", "valid"],
                    callbacks=[lgb.early_stopping(80, verbose=False),
                               lgb.record_evaluation(evals),
                               lgb.log_evaluation(0) if not verbose else lgb.log_evaluation(100)])
    bst.save_model(os.path.join(MOD, f"lgb_{label}.txt"), num_iteration=bst.best_iteration)

    res = {"label": label, "n_train": len(tr), "n_valid": len(va), "n_test": len(te),
           "best_iter": bst.best_iteration}
    for nm, s in (("train", tr), ("valid", va), ("test", te)):
        if len(s) == 0:
            continue
        pr = bst.predict(s[fc], num_iteration=bst.best_iteration)
        res[f"auc_{nm}"] = float(roc_auc_score(s[label], pr))
        res[f"pos_rate_{nm}"] = float(s[label].mean())
        # 每日横截面 IC（预测分 vs 未来收益），向量化 rank 相关
        ic = daily_spearman_ic(s["date"].values, pr, s["fwd5"].values)
        res[f"ic_{nm}"] = float(ic.mean())
        res[f"icir_{nm}"] = float(ic.mean() / ic.std()) if ic.std() > 0 else np.nan
        res[f"n_days_{nm}"] = int(len(ic))
    return bst, res, fc


def walk_forward(df, label, years=range(2021, 2026)):
    """逐年滚动：train ≤ Y-2, valid = Y-1, test = Y"""
    out = []
    for y in years:
        d = df[(df["date"] >= f"{y}-01-01") & (df["date"] <= f"{y}-12-31")]
        if len(d) == 0:
            continue
        sub = df[df["date"] <= f"{y}-12-31"].copy()
        res = {}
        orig = (TRAIN, VALID, TEST)
        globals()["TRAIN"] = (f"{y-5}-01-01", f"{y-2}-12-31")
        globals()["VALID"] = (f"{y-1}-01-01", f"{y-1}-12-31")
        globals()["TEST"] = (f"{y}-01-01", f"{y}-12-31")
        try:
            _, res, _ = train_one(sub, label, verbose=False)
        finally:
            globals()["TRAIN"], globals()["VALID"], globals()["TEST"] = orig
        res["walk_year"] = y
        out.append(res)
        print(f"  [WF {y}] AUC={res.get('auc_test', float('nan')):.4f} IC={res.get('ic_test', float('nan')):.4f}")
    return pd.DataFrame(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--walk", action="store_true")
    ap.add_argument("--label", default="all", choices=["all", "y_raw", "y_exc", "y_exc_nzt"])
    a = ap.parse_args()

    df = load()
    print(f"[lgb] panel {df.shape}  dates {df['date'].min().date()} ~ {df['date'].max().date()}")
    labels = ["y_raw", "y_exc", "y_exc_nzt"] if a.label == "all" else [a.label]

    lines = ["# LightGBM 训练结果\n"]
    all_res, imps = [], {}
    for lb in labels:
        print(f"\n[lgb] === {lb} ===")
        bst, res, fc = train_one(df, lb)
        all_res.append(res)
        imps[lb] = bst.feature_importance("gain")
        lines.append(f"\n## {lb}\n")
        for k, v in res.items():
            lines.append(f"- {k}: {v if not isinstance(v, float) else round(v, 4)}")

        if a.walk:
            wf = walk_forward(df, lb)
            wf.to_csv(os.path.join(OUT, f"m4_walkforward_{lb}.csv"), index=False)
            lines.append("\n逐年 walk-forward:\n")
            lines.append(wf[["walk_year", "n_test", "auc_test", "ic_test", "icir_test"]].round(4).to_string(index=False))

    R = pd.DataFrame(all_res)
    R.to_csv(os.path.join(OUT, "m4_metrics.csv"), index=False)
    print("\n" + R.round(4).to_string(index=False))

    fc = feat_cols(df)
    imp = pd.DataFrame(imps, index=fc)
    imp.index.name = "feature"
    imp.sort_values("y_exc", ascending=False).to_csv(os.path.join(OUT, "m4_importance.csv"))
    lines.append("\n## 特征重要性（gain, y_exc Top20）\n")
    top = imp["y_exc"].sort_values(ascending=False).head(20)
    for k, v in top.items():
        lines.append(f"- {k}: {v:,.0f}")

    with open(os.path.join(OUT, "m4_metrics.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"[saved] {os.path.join(OUT,'m4_metrics.md')}")


if __name__ == "__main__":
    main()
