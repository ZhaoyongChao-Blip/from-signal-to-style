# -*- coding: utf-8 -*-
"""sign_split.py —— 按基准涨跌分组的日均超额 + 加权平均。

数据源：panel.parquet 重算的等权市场 + 三个构造的净值 CSV。
产出写入 numbers.json，供 check_paper 断言。
"""
import json, os
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

CONSTS = [("v3b", "m8d_nav_V3b.csv"),
          ("smoothed", "m9_nav_R1_1.0.csv"),
          ("baseline", "m8d_nav_V1a.csv")]

res = {}
for key, f in CONSTS:
    s = pd.read_csv(os.path.join(OUT, f), index_col=0, parse_dates=True).iloc[:, 0].dropna()
    r = s.pct_change().dropna()
    j = pd.concat([r.rename("s"), ew.rename("m")], axis=1, sort=True).dropna()
    up, dn = j[j.m > 0], j[j.m <= 0]
    ex_up = float((up.s - up.m).mean())
    ex_dn = float((dn.s - dn.m).mean())
    wavg = float((len(up) * ex_up + len(dn) * ex_dn) / len(j))
    res[key] = dict(up=ex_up, down=ex_dn, wavg=wavg, n_up=len(up), n_dn=len(dn))

print(f"{'construct':10s} {'up':>9s} {'down':>9s} {'weighted':>10s}  n_up n_dn")
for k, v in res.items():
    print(f"{k:10s} {v['up']*100:+8.3f}% {v['down']*100:+8.3f}% {v['wavg']*100:+9.4f}%  {v['n_up']:4d} {v['n_dn']:4d}")

# 写入 numbers.json
pth = os.path.join(ROOT, "paper", "numbers.json")
nj = json.load(open(pth, encoding="utf-8"))
for k, v in res.items():
    for f, fld in (("up", "ex_up"), ("down", "ex_down"), ("wavg", "ex_wavg")):
        nj["numbers"][f"sign_{k}_{fld}"] = v[f]
        nj["source"][f"sign_{k}_{fld}"] = "panel.parquet + " + dict(CONSTS)[k]
    nj["numbers"][f"sign_{k}_n_up"] = v["n_up"]
    nj["source"][f"sign_{k}_n_up"] = "panel.parquet + " + dict(CONSTS)[k]
    nj["numbers"][f"sign_{k}_n_dn"] = v["n_dn"]
    nj["source"][f"sign_{k}_n_dn"] = "panel.parquet + " + dict(CONSTS)[k]
json.dump(nj, open(pth, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("\n[written] numbers.json")
