# -*- coding: utf-8 -*-
"""decision3_robust.py —— 「小盘不崩时段能否跑赢」的稳健性检验

季度只有 8 个、下跌季只有 2 个，样本太薄。改做三档更细的切分：
  R1 按日：基准日收益 > 0 / <= 0
  R2 按月：基准月收益 > 0 / <= 0
  R3 剔除 2024H1 后整段的表现
  R4 用不同构造（原版 V3b / 平滑 ma5）分别做，看结论是否一致
"""
import os
from math import sqrt
import pandas as pd, numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "output")
RF = 0.015


def load_nav(f):
    return pd.read_csv(os.path.join(OUT, f), index_col=0, parse_dates=True).iloc[:, 0].dropna()


p = pd.read_parquet(os.path.join(ROOT, "data", "proc", "panel.parquet"),
                    columns=["date", "code", "adj_close"])
p["date"] = pd.to_datetime(p["date"])
d = p[(p["date"] >= "2024-01-01") & (p["date"] <= "2025-11-10")].dropna(subset=["adj_close"])
d = d.sort_values(["code", "date"]).assign(r=d.groupby("code")["adj_close"].pct_change())
ew = d.groupby("date")["r"].mean().fillna(0)
ew.index = pd.to_datetime(ew.index)

STRATS = {
    "V3b 原版 (3亿,5d,15bp)": "m8d_nav_V3b.csv",
    "平滑 ma5 (3亿,5d,15bp)": "m9_nav_R1_1.0.csv",
    "V1a (2千万,5d)": "m8d_nav_V1a.csv",
}


def tstat(x):
    x = pd.Series(x).dropna()
    if len(x) < 3 or x.std(ddof=1) == 0:
        return np.nan, len(x)
    return x.mean() / (x.std(ddof=1) / sqrt(len(x))), len(x)


for sname, f in STRATS.items():
    nav = load_nav(f)
    rp = nav.pct_change().fillna(0)
    rb = ew.reindex(nav.index).fillna(0)
    print("=" * 72)
    print(f"策略：{sname}")
    print("=" * 72)

    # ---- R1 日频 ----
    up = (rp - rb)[rb > 0]
    dn = (rp - rb)[rb <= 0]
    print(f"  R1 日频：基准上涨 {len(up)} 天，日均超额 {up.mean()*100:+.4f}%  "
          f"t={tstat(up)[0]:+.2f}")
    print(f"           基准下跌 {len(dn)} 天，日均超额 {dn.mean()*100:+.4f}%  "
          f"t={tstat(dn)[0]:+.2f}")

    # ---- R2 月频 ----
    key = [nav.index.year, nav.index.month]
    m_p = (1 + rp).groupby(key).prod() - 1
    m_b = (1 + rb).groupby(key).prod() - 1
    m_e = m_p - m_b
    m_up = m_e[m_b > 0]
    m_dn = m_e[m_b <= 0]
    print(f"  R2 月频：基准上涨 {len(m_up)} 月，平均超额 {m_up.mean()*100:+.2f}pp  "
          f"t={tstat(m_up)[0]:+.2f}  胜 {int((m_up>0).sum())}/{len(m_up)}")
    print(f"           基准下跌 {len(m_dn)} 月，平均超额 {m_dn.mean()*100:+.2f}pp  "
          f"t={tstat(m_dn)[0]:+.2f}  胜 {int((m_dn>0).sum())}/{len(m_dn)}")

    # ---- R3 剔除 2024H1 ----
    mask = ~((nav.index.year == 2024) & (nav.index.month <= 6))
    for tag, m in (("全期", slice(None)), ("剔除2024H1", mask)):
        if tag == "全期":
            rp2, rb2 = rp, rb
        else:
            rp2, rb2 = rp[m], rb[m]
        a_p = (1 + rp2).prod() ** (252 / len(rp2)) - 1
        a_b = (1 + rb2).prod() ** (252 / len(rb2)) - 1
        v_p = rp2.std() * np.sqrt(252)
        print(f"  R3 {tag:12s}：组合年化 {a_p*100:6.2f}%  基准年化 {a_b*100:6.2f}%  "
              f"年化差 {(a_p-a_b)*100:+6.2f}pp  Sharpe {(a_p-RF)/v_p:+.3f}")
    print()
