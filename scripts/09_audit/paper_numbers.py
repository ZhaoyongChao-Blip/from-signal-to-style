# -*- coding: utf-8 -*-
"""paper_numbers.py —— 把论文里要引用的每个数字，从源文件重算一遍，落成 paper/numbers.json

规则（用户定）：论文里每个硬数字都必须能追溯到结果文件。
本脚本 = 断言表：key → 值 + 来源文件。论文生成脚本只允许从这里取数。
"""
import os, sys, json, re
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
OUT = os.path.join(ROOT, "output")
PAPER = os.path.join(ROOT, "paper")
os.makedirs(PAPER, exist_ok=True)
N = {}
SRC = {}


def put(k, v, src):
    N[k] = float(v) if isinstance(v, (int, float, np.floating, np.integer)) else v
    SRC[k] = src


def perf(nav):
    r = nav.pct_change().dropna()
    ann = (1 + r).prod() ** (252 / len(r)) - 1
    vol = r.std() * np.sqrt(252)
    return ann, vol, (ann - 0.015) / vol, (nav / nav.cummax() - 1).min(), len(r)


def main():
    # ---------- 数据与面板 ----------
    panel = pd.read_parquet(os.path.join(ROOT, "data", "proc", "panel.parquet"),
                            columns=["date", "code", "amount", "fwd5", "has_fwd5",
                                     "adj_close", "adj_open", "raw_close", "raw_open", "board"])
    panel["date"] = pd.to_datetime(panel["date"])
    put("n_rows", len(panel), "panel.parquet")
    put("n_stocks_panel", panel["code"].nunique(), "panel.parquet")
    put("date_min", str(panel["date"].min().date()), "panel.parquet")
    put("date_max", str(panel["date"].max().date()), "panel.parquet")
    put("n_days_total", panel["date"].nunique(), "panel.parquet")

    lst = pd.read_csv(os.path.join(ROOT, "data", "raw", "list.csv"))
    put("n_stocks_list", len(lst), "data/raw/list.csv")

    # ---------- 因子 IC ----------
    ic = pd.read_csv(os.path.join(OUT, "m6_factor_ic.csv"))
    put("ic_max_abs", ic["IC"].abs().max(), "m6_factor_ic.csv")
    put("ic_min_abs", ic["IC"].abs().min(), "m6_factor_ic.csv")
    put("ic_top_factor", ic.iloc[ic["IC"].abs().argmax()]["factor"], "m6_factor_ic.csv")
    put("ic_top_value", ic.iloc[ic["IC"].abs().argmax()]["IC"], "m6_factor_ic.csv")
    put("ic_n_factors", len(ic), "m6_factor_ic.csv")
    put("ic_n_above_008", int((ic["IC"].abs() >= 0.08).sum()), "m6_factor_ic.csv")

    # ---------- 无重叠 IC ----------
    b = pd.read_csv(os.path.join(OUT, "m4b_ic_nonoverlap.csv"))
    for _, r in b.iterrows():
        k = f"{r['label']}_{r['span']}"
        put(f"ic_{k}", r["ic_nonoverlap"], "m4b_ic_nonoverlap.csv")
        put(f"t_{k}", r["t_nonoverlap"], "m4b_ic_nonoverlap.csv")
        put(f"tnaive_{k}", r["t_overlap_naive"], "m4b_ic_nonoverlap.csv")
        put(f"nday_{k}", r["n_days_nonoverlap"], "m4b_ic_nonoverlap.csv")

    # ---------- 净值指标 ----------
    panels = {
        "y_exc_2kw_5d": "m8c_nav_y_exc.csv",
        "y_neut_2kw_5d": "m8c_nav_y_neut.csv",
        "v3b_3e_5d": "m8d_nav_V3b.csv",
        "ma5_3e_5d": "m9_nav_R1_1.0.csv",
        "v3a_1e_5d": "m8d_nav_V3a.csv",
    }
    for k, f in panels.items():
        p = os.path.join(OUT, f)
        if not os.path.exists(p):
            continue
        nav = pd.read_csv(p, index_col=0, parse_dates=True).iloc[:, 0].dropna()
        ann, vol, sh, mdd, n = perf(nav)
        put(f"ann_{k}", ann, f)
        put(f"vol_{k}", vol, f)
        put(f"sharpe_{k}", sh, f)
        put(f"mdd_{k}", mdd, f)
        put(f"ndays_{k}", n, f)

    # 等权基准
    d = panel[(panel["date"] >= "2024-01-01") & (panel["date"] <= "2025-11-10")]
    d = d.dropna(subset=["adj_close"]).sort_values(["code", "date"])
    d = d.assign(r=d.groupby("code")["adj_close"].pct_change())
    ew = (1 + d.groupby("date")["r"].mean().fillna(0)).cumprod()
    ann, vol, sh, mdd, n = perf(ew)
    put("ann_ew", ann, "panel recompute")
    put("vol_ew", vol, "panel recompute")
    put("sharpe_ew", sh, "panel recompute")
    put("mdd_ew", mdd, "panel recompute")

    # ---------- 从 md 解析的表 ----------
    def md_rows(path, kw):
        if not os.path.exists(path):
            return []
        lines = open(path, encoding="utf-8").read().splitlines()
        for i, l in enumerate(lines):
            if kw in l and l.strip().startswith("|"):
                out = []
                for l2 in lines[i + 2:]:
                    if not l2.strip().startswith("|"):
                        break
                    out.append([c.strip() for c in l2.strip().strip("|").split("|")])
                return out
        return []

    def nm(s):
        s = s.replace("**", "").replace(" ", "").replace(",", "")
        for t in ("%", "pp", "倍", "bp", "日", "亿"):
            s = s.replace(t, "")
        try:
            return float(s)
        except Exception:
            return np.nan

    # t7f 同格子超额（解析列位置：模型|交易日|picks均值|同格子基准|差|标准误|t）
    # 先补 AUC（来自 m7d_neut_eval.md 的 L2 表，test 行）
    for r in md_rows(os.path.join(OUT, "m7d_neut_eval.md"), "| 模型 | 集合 | n | AUC"):
        if len(r) < 5:
            continue
        model, split, auc = r[0], r[1], nm(r[3])
        if split == "test" and model in ("y_exc", "y_neut"):
            put(f"auc_{model.replace('y_', '')}", auc, "m7d_neut_eval.md")
        if split == "test" and model == "y_neut":
            put("ic_own_yneut_test", nm(r[4]), "m7d_neut_eval.md")
            tt = r[6] if len(r) > 6 else ""
            mt = re.search(r"([+-]?[0-9.]+)", tt)
            if mt:
                put("t_own_yneut_test", float(mt.group(1)), "m7d_neut_eval.md")

    for r in md_rows(os.path.join(OUT, "m7f_group.md"), "| 模型 | 交易日 |"):
        if r[0] == "LGB_y_exc":
            put("cell_picks_yexc", nm(r[2]) / 100, "m7f_group.md")
            put("cell_bench_yexc", nm(r[3]) / 100, "m7f_group.md")
            put("cell_exc_yexc", nm(r[4]) / 100, "m7f_group.md")
            put("cell_exc_yexc_t", nm(r[6].split("（")[0]), "m7f_group.md")
        elif r[0] == "LGB_y_neut":
            put("cell_picks_yneut", nm(r[2]) / 100, "m7f_group.md")
            put("cell_bench_yneut", nm(r[3]) / 100, "m7f_group.md")
            put("cell_exc_yneut", nm(r[4]) / 100, "m7f_group.md")
            put("cell_exc_yneut_t", nm(r[6].split("（")[0]), "m7f_group.md")

    # 风格 R²
    t = open(os.path.join(OUT, "m7b_neutral.md"), encoding="utf-8").read()
    m = re.search(r"逐日 R[²2] 均值\s*=\s*\*\*([0-9.]+)\*\*", t)
    if m:
        put("style_r2", float(m.group(1)), "m7b_neutral.md")
    m = re.search(r"中位数 ([0-9.]+)", t)
    if m:
        put("style_r2_median", float(m.group(1)), "m7b_neutral.md")

    # 成本敏感性
    for r in md_rows(os.path.join(OUT, "m9b_smooth.md"), "| 信号 |"):
        if len(r) < 3:
            continue
        key = {"s": "s", "s_ma5": "ma5", "s_lag5": "lag5"}.get(r[0])
        if not key:
            continue
        put(f"cost_ann_{key}_{int(nm(r[1]))}bp", nm(r[2]) / 100, "m9b_smooth.md")

    # 窗口敏感性
    for r in md_rows(os.path.join(OUT, "m9c_final.md"), "| 窗口 |"):
        if len(r) < 4:
            continue
        w = int(nm(r[0]))
        put(f"win_ann_{w}d", nm(r[1]) / 100, "m9c_final.md")
        put(f"win_exc_{w}d", nm(r[3]) / 100, "m9c_final.md")

    # 逐季度超额
    qs = md_rows(os.path.join(OUT, "m9d_excess.md"), "| 季度 | 组合收益")
    for r in qs:
        if "全期" in r[0]:
            put("q_total_port", nm(r[1]) / 100, "m9d_excess.md")
            put("q_total_bench", nm(r[2]) / 100, "m9d_excess.md")
            put("q_total_exc", nm(r[3]) / 100, "m9d_excess.md")
            continue
        put(f"q_{r[0]}_port", nm(r[1]) / 100, "m9d_excess.md")
        put(f"q_{r[0]}_bench", nm(r[2]) / 100, "m9d_excess.md")
        put(f"q_{r[0]}_exc", nm(r[3]) / 100, "m9d_excess.md")

    # ---------- 涨停 ----------
    fw = panel[panel["has_fwd5"] == 1].copy()
    zt = fw["fwd5"].notna()
    # 用 m6_diagnose 已算好的口径重算（fwd5_has_zt 不在裁剪列里，改从 csv 读）
    dg = os.path.join(OUT, "m6_diagnose.md")
    if os.path.exists(dg):
        t = open(dg, encoding="utf-8").read()
        for key, pat in (("zt_share", r"含涨停[^0-9]*([0-9.]+)%"),
                         ("zt_mean", r"含涨停[^\n]*?均值\s*([+-]?[0-9.]+)%")):
            m = re.search(pat, t)
            if m:
                put(key, float(m.group(1)) / 100, "m6_diagnose.md")

    # 容量
    cap = os.path.join(OUT, "m8g_poolfix.md")
    if os.path.exists(cap):
        t = open(cap, encoding="utf-8").read()
        m = re.search(r"picks 成交额合计中位 ([0-9.]+) 亿", t)
        if m:
            put("cap_amt_sum", float(m.group(1)), "m8g_poolfix.md")

    json.dump({"numbers": N, "source": SRC}, open(os.path.join(PAPER, "numbers.json"), "w",
              encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"[saved] paper/numbers.json  ({len(N)} keys)")
    for k in sorted(N):
        v = N[k]
        print(f"  {k:34s} {v if isinstance(v,str) else round(v,6)}")


if __name__ == "__main__":
    main()
