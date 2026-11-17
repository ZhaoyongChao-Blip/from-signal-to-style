# -*- coding: utf-8 -*-
"""attrib_check.py —— 核实 4.3 节的三因子归因数字

论文 4.3 写：三因子回归（等权市场 / CSI300 / CSI1000）得
  loading 1.8 on EW, -0.1 on CSI300, R2 0.75-0.81, alpha t = 0.03 / -0.63。
这与 beta_check.py 的双因子 beta 0.476 差很多，必须查清口径差异。
"""
import os
import json
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "output")

p = pd.read_parquet(os.path.join(ROOT, "data", "proc", "panel.parquet"),
                    columns=["date", "code", "adj_close", "mkt_fwd5",
                             "idx_000300_fwd5", "idx_000905_fwd5", "idx_000852_fwd5"])
p["date"] = pd.to_datetime(p["date"])
d = p[(p.date >= "2024-01-01") & (p.date <= "2025-11-10")]
d = d.sort_values(["code", "date"]).assign(r=d.groupby("code")["adj_close"].pct_change())
ew = d.groupby("date")["r"].mean().fillna(0)
ew.index = pd.to_datetime(ew.index)

# 指数日收益：从 fwd5 列反推不现实，改用净值列
# 这里直接用 panel 里的指数 fwd5 做 5 日重叠收益的近似不合适，
# 所以改为读取独立的指数日线（如果存在）
idx_file = os.path.join(ROOT, "data", "proc", "index_daily.parquet")
print("index_daily.parquet 存在:", os.path.exists(idx_file))

CONSTS = [("V1a (raw)", "m8d_nav_V1a.csv"),
          ("V3b (exc)", "m8d_nav_V3b.csv")]

# 只用能拿到的：EW 单因子 + 双因子
print(f"\n{'construct':12s} {'n':>6s} {'beta_EW':>8s} {'alpha_d':>10s} {'t(alpha)':>9s} {'R2':>6s}")
print("-" * 60)
for name, f in CONSTS:
    s = pd.read_csv(os.path.join(OUT, f), index_col=0, parse_dates=True).iloc[:, 0].dropna()
    r = s.pct_change().dropna()
    j = pd.concat([r.rename("p"), ew.rename("m")], axis=1, sort=True).dropna()
    y, x = j.p.values, j.m.values
    X = np.column_stack([np.ones(len(x)), x])
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ b
    n, k = len(y), 2
    s2 = resid @ resid / (n - k)
    se = np.sqrt(np.diag(s2 * np.linalg.inv(X.T @ X)))
    r2 = 1 - (resid @ resid) / ((y - y.mean()) @ (y - y.mean()))
    print(f"{name:12s} {n:6d} {b[1]:8.3f} {b[0]*100:9.4f}% {b[0]/se[0]:9.2f} {r2:6.3f}")
