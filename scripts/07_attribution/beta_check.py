# -*- coding: utf-8 -*-
"""beta_check.py —— 核实组合相对等权市场的 beta / alpha

背景：论文 7.3 说「这是低 beta 组合的特征」，但摘要说「beta 1.8」。
两者矛盾，必须用数据裁决。
"""
import os
import json
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "output")

p = pd.read_parquet(os.path.join(ROOT, "data", "proc", "panel.parquet"),
                    columns=["date", "code", "adj_close"])
p["date"] = pd.to_datetime(p["date"])
d = p[(p.date >= "2024-01-01") & (p.date <= "2025-11-10")].dropna(subset=["adj_close"])
d = d.sort_values(["code", "date"]).assign(r=d.groupby("code")["adj_close"].pct_change())
ew = d.groupby("date")["r"].mean().fillna(0)
ew.index = pd.to_datetime(ew.index)
mkt = ew.rename("mkt")

CONSTS = [("V3b", "m8d_nav_V3b.csv"),
          ("Smoothed", "m9_nav_R1_1.0.csv"),
          ("Baseline", "m8d_nav_V1a.csv")]

print(f"{'construct':10s} {'beta':>8s} {'alpha_d':>10s} {'alpha_ann':>10s} {'t(alpha)':>9s} {'corr':>7s}")
print("-" * 62)
res = {}
for name, f in CONSTS:
    s = pd.read_csv(os.path.join(OUT, f), index_col=0, parse_dates=True).iloc[:, 0].dropna()
    r = s.pct_change().dropna()
    j = pd.concat([r.rename("p"), mkt], axis=1, sort=True).dropna()
    y = j.p.values
    x = j.mkt.values
    X = np.column_stack([np.ones(len(x)), x])
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    alpha_d, beta = b
    resid = y - X @ b
    n, k = len(y), 2
    s2 = resid @ resid / (n - k)
    se = np.sqrt(np.diag(s2 * np.linalg.inv(X.T @ X)))
    t_alpha = alpha_d / se[0]
    alpha_ann = (1 + alpha_d) ** 252 - 1
    print(f"{name:10s} {beta:8.3f} {alpha_d*100:9.4f}% {alpha_ann*100:9.2f}% "
          f"{t_alpha:9.2f} {np.corrcoef(x,y)[0,1]:7.3f}")
    res[name] = dict(beta=float(beta), alpha_daily=float(alpha_d),
                     alpha_ann=float(alpha_ann), t_alpha=float(t_alpha), n=int(n))

print()
print("判据：")
v = res["V3b"]
if v["beta"] < 1.0:
    print("  beta < 1  -> 7.3 的「low-beta book」说法成立")
else:
    print(f"  beta = {v['beta']:.2f} >= 1  -> 7.3 的「low-beta book」说法不成立，"
          f"应改为「beta 偏离 1 但方向不是低」或重写机制")

json.dump(res, open(os.path.join(ROOT, "beta_result.json"), "w", encoding="utf-8"), indent=1)
