# -*- coding: utf-8 -*-
"""t10_walkforward.py —— 扩展窗口逐年样本外，把 2 年样本外拉长到 7 年

设计（每一年都是真正的样本外）：
  for Y in 2019..2025:
      train  = 2014-01-01 ~ (Y-2)-12-31   (扩展窗口，逐年变长)
      valid  = (Y-1)-01-01 ~ (Y-1)-12-31  (早停)
      test   = Y-01-01 ~ Y-12-31          (样本外)

标签 y_neut 复用 t7c 的定义（逐日截面回归 excess5 ~ 4 风格因子的残差 > 0），
不重写，直接 import resid_block，避免两套实现给出不同的数。

★ 防泄漏：每次训练断言 train.max < valid.min <= valid.max < test.min。
"""
import gc
import os
import sys
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
OUT = os.path.join(ROOT, "output")
PANEL = os.path.join(ROOT, "data", "proc", "panel.parquet")

import t4_lgb            # noqa: E402
from t7c_neut_lgb import resid_block   # noqa: E402

STYLE = ["log_amt", "log_price", "vol_20", "turn_ma_20"]
LABEL = "y_neut"

PARAMS = dict(
    objective="binary", metric="auc", learning_rate=0.03, num_leaves=31,
    min_data_in_leaf=20, feature_fraction=0.8, bagging_fraction=0.8,
    bagging_freq=1, lambda_l1=0.1, lambda_l2=5.0, num_threads=4,
    verbosity=-1, seed=42,
)
CH = 300_000


def build_labels(panel):
    """逐块做截面回归，得到 ex_neut 与 y_neut"""
    ex = np.full(len(panel), np.nan)
    g_all = panel["date"].astype("category").cat.codes.to_numpy()
    for i in range(0, len(panel), CH):
        blk = panel.iloc[i:i + CH]
        g = g_all[i:i + CH]
        X = blk[STYLE].to_numpy(dtype=np.float64)
        y = blk["excess5"].to_numpy(dtype=np.float64)
        ex[i:i + len(blk)] = resid_block(y, X, g - g.min(), len(STYLE))
        print(f"  resid {i//CH+1}/{(len(panel)+CH-1)//CH}", flush=True)
        gc.collect()
    panel["ex_neut"] = ex
    panel["y_neut"] = (panel["ex_neut"] > 0).astype(int)
    return panel


def daily_ic(dates, pred, y):
    d = pd.DataFrame({"date": dates, "p": pred, "y": y}).dropna()
    if len(d) < 100:
        return np.nan, np.nan
    g = d.groupby("date", observed=True)
    ic = g.apply(lambda x: x["p"].corr(x["y"], method="spearman"), include_groups=False)
    return float(ic.mean()), ic


def main():
    t0 = time.time()
    print("[load] panel ...", flush=True)
    d = pd.read_parquet(PANEL)
    d = d[d["has_fwd5"] == 1].sort_values("date", kind="stable").reset_index(drop=True)
    print(f"[load] {len(d):,} rows", flush=True)

    print("[label] building y_neut ...", flush=True)
    d = build_labels(d)
    print(f"[label] pos-rate={d[LABEL].mean():.4f}  ({time.time()-t0:.0f}s)", flush=True)

    feat_cols = [c for c in t4_lgb.feat_cols(d) if c not in ("ex_neut", "y_neut")]
    # 防泄漏硬断言
    bad = [c for c in feat_cols if c in ("y_neut", "y_exc", "y_raw", "y_exc_nzt", "ex_neut", "excess5")
           or c.startswith("idx_") or c.startswith("fwd") or c.startswith("mkt_")]
    assert not bad, f"特征混入标签/未来信息: {bad}"
    print(f"[feat] {len(feat_cols)} features", flush=True)

    rows = []
    for Y in range(2019, 2026):
        tr = d[(d["date"] >= "2014-01-01") & (d["date"] <= f"{Y-2}-12-31")]
        va = d[(d["date"] >= f"{Y-1}-01-01") & (d["date"] <= f"{Y-1}-12-31")]
        te = d[(d["date"] >= f"{Y}-01-01") & (d["date"] <= f"{Y}-12-31")]

        assert tr["date"].max() < va["date"].min(), f"{Y}: train/valid 重叠"
        assert va["date"].max() < te["date"].min(), f"{Y}: valid/test 重叠"
        assert len(te) > 1000 and len(va) > 1000, f"{Y}: 样本不足"

        dtr = lgb.Dataset(tr[feat_cols].to_numpy(np.float32), tr[LABEL].to_numpy())
        dva = lgb.Dataset(va[feat_cols].to_numpy(np.float32), va[LABEL].to_numpy(), reference=dtr)
        bst = lgb.train(PARAMS, dtr, num_boost_round=1500, valid_sets=[dva],
                        valid_names=["valid"],
                        callbacks=[lgb.early_stopping(80, verbose=False)])
        best = bst.best_iteration or 500

        p_te = bst.predict(te[feat_cols].to_numpy(np.float32), num_iteration=best)
        p_va = bst.predict(va[feat_cols].to_numpy(np.float32), num_iteration=best)
        auc_te = roc_auc_score(te[LABEL].to_numpy(), p_te)
        auc_va = roc_auc_score(va[LABEL].to_numpy(), p_va)
        ic_te, _ = daily_ic(te["date"].values, p_te, te["fwd5"].values)

        # stride=5 无重叠
        ds = np.sort(te["date"].unique())
        sub = te[te["date"].isin(set(ds[::5]))]
        p_sub = bst.predict(sub[feat_cols].to_numpy(np.float32), num_iteration=best)
        ic_s5, icd = daily_ic(sub["date"].values, p_sub, sub["fwd5"].values)
        t_s5 = float(icd.mean() / (icd.std(ddof=1) / np.sqrt(len(icd)))) if len(icd) > 3 else np.nan

        rows.append(dict(
            year=Y, best_iter=best, n_train=len(tr), n_valid=len(va), n_test=len(te),
            test_days=int(te["date"].nunique()),
            auc_valid=round(float(auc_va), 4), auc_test=round(float(auc_te), 4),
            ic_test=round(ic_te, 4), ic_stride5=round(ic_s5, 4), t_stride5=round(t_s5, 2),
        ))
        print(f"[{Y}] train={len(tr):>9,} test={len(te):>7,} best={best:>4} "
              f"AUC_val={auc_va:.4f} AUC_te={auc_te:.4f} IC={ic_te:.4f} "
              f"IC_s5={ic_s5:.4f} t_s5={t_s5:.2f}  ({time.time()-t0:.0f}s)", flush=True)
        sys.stdout.flush()

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "m10_walkforward.csv"), index=False)

    print("\n" + "=" * 86)
    print("扩展窗口逐年样本外（y_neut，train 2014 ~ Y-2）")
    print("=" * 86)
    print(df.to_string(index=False))

    tot = int(df["n_test"].sum())
    w = float((df["ic_test"] * df["n_test"]).sum() / tot)
    print(f"\n合计样本外 {tot:,} 行 / {int(df['test_days'].sum())} 交易日 / {len(df)} 个独立年度")
    print(f"样本加权 IC = {w:.4f}")
    print(f"IC 均值 = {df['ic_test'].mean():.4f}  中位 = {df['ic_test'].median():.4f}")
    print(f"IC > 0 的年份 = {int((df['ic_test'] > 0).sum())}/{len(df)}")
    print(f"AUC 均值 = {df['auc_test'].mean():.4f}  范围 "
          f"[{df['auc_test'].min():.4f}, {df['auc_test'].max():.4f}]")
    print(f"AUC > 0.50 的年份 = {int((df['auc_test'] > 0.5).sum())}/{len(df)}")
    print(f"\n[saved] output/m10_walkforward.csv  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
