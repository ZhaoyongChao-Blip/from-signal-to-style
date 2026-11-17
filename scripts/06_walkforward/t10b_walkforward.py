# -*- coding: utf-8 -*-
"""t10b_walkforward.py —— 扩展窗口逐年样本外（内存友好版）

t10 跑到 2022 年时进程静默退出（训练集 450 万行 + 全表常驻，OOM）。
本版改动：
  1. 只保留 46 特征 + 4 风格 + 标签所需列，其余 drop
  2. 全部转 float32 后释放原始 float64
  3. 每年用完立刻 del 训练/验证集并 gc

设计同 t10：train 2014 ~ (Y-2)，valid Y-1，test Y。每一年都是真正样本外。
★ 支持断点续跑：已有年份从 m10_partial.json 读回，不重算。
"""
import gc
import json
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
PARAMS = dict(objective="binary", metric="auc", learning_rate=0.03, num_leaves=31,
              min_data_in_leaf=20, feature_fraction=0.8, bagging_fraction=0.8,
              bagging_freq=1, lambda_l1=0.1, lambda_l2=5.0, num_threads=4,
              verbosity=-1, seed=42)
CH = 300_000
PART = os.path.join(OUT, "m10_partial.json")


def daily_ic(dates, pred, y):
    d = pd.DataFrame({"date": dates, "p": pred, "y": y}).dropna()
    if len(d) < 100:
        return np.nan, np.nan
    g = d.groupby("date", observed=True)
    ic = g.apply(lambda x: x["p"].corr(x["y"], method="spearman"), include_groups=False)
    return float(ic.mean()), ic


def main():
    t0 = time.time()
    done = json.load(open(PART, encoding="utf-8")) if os.path.exists(PART) else []
    done_years = {r["year"] for r in done}
    print(f"[resume] 已有年份 {sorted(done_years)}", flush=True)

    print("[load] panel ...", flush=True)
    feat_cols = None
    d = pd.read_parquet(PANEL)
    d = d[d["has_fwd5"] == 1].sort_values("date", kind="stable").reset_index(drop=True)

    # ---- 先确定特征列，再只保留必要列，砍内存 ----
    feat_cols = [c for c in t4_lgb.feat_cols(d) if c not in ("ex_neut", "y_neut")]
    bad = [c for c in feat_cols if c in ("y_neut", "y_exc", "y_raw", "y_exc_nzt", "ex_neut", "excess5")
           or c.startswith("idx_") or c.startswith("fwd") or c.startswith("mkt_")]
    assert not bad, f"特征混入标签/未来信息: {bad}"

    keep = list(dict.fromkeys(feat_cols + STYLE + ["date", "code", "fwd5", "excess5"]))
    d = d[keep].copy()
    gc.collect()
    for c in feat_cols + STYLE + ["fwd5", "excess5"]:
        if d[c].dtype != np.float32:
            d[c] = d[c].astype(np.float32)
    gc.collect()
    print(f"[load] {len(d):,} rows, {len(feat_cols)} feats, "
          f"mem={d.memory_usage(deep=True).sum()/1e9:.2f}GB", flush=True)

    print("[label] building y_neut ...", flush=True)
    ex = np.full(len(d), np.nan, dtype=np.float32)
    g_all = d["date"].astype("category").cat.codes.to_numpy()
    for i in range(0, len(d), CH):
        blk = d.iloc[i:i + CH]
        g = g_all[i:i + CH]
        X = blk[STYLE].to_numpy(dtype=np.float64)
        y = blk["excess5"].to_numpy(dtype=np.float64)
        ex[i:i + len(blk)] = resid_block(y, X, g - g.min(), len(STYLE)).astype(np.float32)
        del blk, X, y
        gc.collect()
        if (i // CH) % 8 == 0:
            print(f"  resid {i//CH+1}/{(len(d)+CH-1)//CH}", flush=True)
    d["y_neut"] = (ex > 0).astype(np.int8)
    del ex, g_all
    gc.collect()
    print(f"[label] pos-rate={d[LABEL].mean():.4f}  ({time.time()-t0:.0f}s)", flush=True)

    Fc = np.ascontiguousarray(d[feat_cols].to_numpy(np.float32))
    dates = d["date"].to_numpy()
    yv = d[LABEL].to_numpy()
    fwd5 = d["fwd5"].to_numpy()
    del d
    gc.collect()
    print(f"[matrix] X={Fc.shape} {Fc.nbytes/1e9:.2f}GB", flush=True)

    for Y in range(2019, 2026):
        if Y in done_years:
            print(f"[{Y}] skip (已算)", flush=True)
            continue
        i_tr = np.where(dates <= np.datetime64(f"{Y-2}-12-31"))[0]
        i_va = np.where((dates >= np.datetime64(f"{Y-1}-01-01")) & (dates <= np.datetime64(f"{Y-1}-12-31")))[0]
        i_te = np.where((dates >= np.datetime64(f"{Y}-01-01")) & (dates <= np.datetime64(f"{Y}-12-31")))[0]
        assert dates[i_tr].max() < dates[i_va].min(), f"{Y}: train/valid 重叠"
        assert dates[i_va].max() < dates[i_te].min(), f"{Y}: valid/test 重叠"

        dtr = lgb.Dataset(Fc[i_tr], yv[i_tr], free_raw_data=True)
        dva = lgb.Dataset(Fc[i_va], yv[i_va], reference=dtr, free_raw_data=True)
        bst = lgb.train(PARAMS, dtr, num_boost_round=1500, valid_sets=[dva],
                        valid_names=["valid"],
                        callbacks=[lgb.early_stopping(80, verbose=False)])
        best = bst.best_iteration or 500

        p_te = bst.predict(Fc[i_te], num_iteration=best)
        p_va = bst.predict(Fc[i_va], num_iteration=best)
        auc_te = float(roc_auc_score(yv[i_te], p_te))
        auc_va = float(roc_auc_score(yv[i_va], p_va))
        ic_te, _ = daily_ic(dates[i_te], p_te, fwd5[i_te])

        ds = np.unique(dates[i_te])
        keepd = set(ds[::5])
        m = np.isin(dates[i_te], list(keepd))
        ic_s5, icd = daily_ic(dates[i_te][m], p_te[m], fwd5[i_te][m])
        t_s5 = float(icd.mean() / (icd.std(ddof=1) / np.sqrt(len(icd)))) if len(icd) > 3 else np.nan

        rec = dict(year=Y, best_iter=int(best), n_train=int(len(i_tr)), n_valid=int(len(i_va)),
                   n_test=int(len(i_te)), test_days=int(len(ds)),
                   auc_valid=round(auc_va, 4), auc_test=round(auc_te, 4),
                   ic_test=round(ic_te, 4), ic_stride5=round(ic_s5, 4),
                   t_stride5=round(t_s5, 2))
        done.append(rec)
        json.dump(done, open(PART, "w", encoding="utf-8"), indent=1)
        print(f"[{Y}] train={len(i_tr):>9,} test={len(i_te):>7,} best={best:>4} "
              f"AUC_val={auc_va:.4f} AUC_te={auc_te:.4f} IC={ic_te:.4f} "
              f"IC_s5={ic_s5:.4f} t_s5={t_s5:.2f}  ({time.time()-t0:.0f}s)", flush=True)
        del dtr, dva, bst, p_te, p_va
        gc.collect()

    df = pd.DataFrame(done).sort_values("year")
    df.to_csv(os.path.join(OUT, "m10_walkforward.csv"), index=False)
    print("\n" + "=" * 88)
    print("扩展窗口逐年样本外（y_neut，train 2014 ~ Y-2）")
    print("=" * 88)
    print(df.to_string(index=False))
    tot = int(df["n_test"].sum())
    print(f"\n合计样本外 {tot:,} 行 / {int(df['test_days'].sum())} 交易日 / {len(df)} 个独立年度")
    print(f"样本加权 IC = {(df['ic_test']*df['n_test']).sum()/tot:.4f}")
    print(f"IC 均值 = {df['ic_test'].mean():.4f}  中位 = {df['ic_test'].median():.4f}  "
          f"IC>0 年份 {int((df['ic_test']>0).sum())}/{len(df)}")
    print(f"AUC 均值 = {df['auc_test'].mean():.4f}  范围 "
          f"[{df['auc_test'].min():.4f}, {df['auc_test'].max():.4f}]  "
          f"AUC>0.50 年份 {int((df['auc_test']>0.5).sum())}/{len(df)}")
    print(f"\n[saved] output/m10_walkforward.csv  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
