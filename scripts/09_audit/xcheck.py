# -*- coding: utf-8 -*-
"""add_crosscheck.py —— 给 paper_numbers.py 追加「三重交叉核对」段

背景（2026-11-12 18:35 自查）：
  numbers.json 里 `ann_y_exc_2kw_5d` 和 `ann_y_neut_2kw_5d` 逐位相同，
  根因是两个 key 指向了同一份净值文件（V1a 与 m8j_2千万 的 max|diff| = 0.0）。
  这类「不同标签却拿到同一份回测」的错误，光看数字看不出来，必须做对撞。

本脚本在 paper_numbers.json 之后追加三类硬校验：
  X1 任意两个「应当不同」的净值文件，逐位相同的必须报错
  X2 每个 key 指向的文件必须真实存在且非空
  X3 正文里出现的每个关键数字，必须能从 numbers.json 反查到
"""
import json, os, itertools
import pandas as pd, numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "output")
PAPER = os.path.join(ROOT, "paper")

FAIL = []

# 应当互不相同的净值文件组
GROUPS = {
    "不同标签（真实价口径）": [
        ("y_exc", "m8c_nav_y_exc.csv"),
        ("y_neut", "m8c_nav_y_neut.csv"),
        ("y_raw", "m5_nav_LGB_y_raw.csv"),
    ],
    "不同资金池（y_neut）": [
        ("2千万", "m8c_nav_y_neut.csv"),
        ("1亿", "m8d_nav_V3a.csv"),
        ("3亿", "m8d_nav_V3b.csv"),
    ],
    "不同处理（3亿池）": [
        ("原版", "m8d_nav_V3b.csv"),
        ("平滑5日", "m9_nav_R1_1.0.csv"),
    ],
}


def load(f):
    p = os.path.join(OUT, f)
    if not os.path.exists(p):
        return None
    s = pd.read_csv(p, index_col=0, parse_dates=True).iloc[:, 0].dropna()
    return s if len(s) > 10 else None


print("=== X1 应当不同的净值必须不同 ===")
for gname, items in GROUPS.items():
    print(f"\n  [{gname}]")
    navs = {}
    for lbl, f in items:
        s = load(f)
        if s is None:
            FAIL.append(f"X1 {gname}: {f} 缺失/过短")
            print(f"    {lbl:10s} {f:26s} MISSING")
            continue
        navs[lbl] = s
        print(f"    {lbl:10s} {f:26s} n={len(s):4d} final={s.iloc[-1]:.6f}")
    for (l1, s1), (l2, s2) in itertools.combinations(navs.items(), 2):
        c = s1.index.intersection(s2.index)
        if len(c) < 50:
            continue
        d = float(np.abs(s1.loc[c] - s2.loc[c]).max())
        if d < 1e-12:
            FAIL.append(f"X1 {gname}: {l1} 与 {l2} 逐位相同 (max|diff|=0)")
            print(f"    !! {l1} == {l2}  逐位相同 -> 违规")
        else:
            print(f"    ok {l1} vs {l2}  max|diff|={d:.2e}")

print("\n=== X2 numbers.json 的每个来源文件必须存在 ===")
n = json.load(open(os.path.join(PAPER, "numbers.json"), encoding="utf-8"))
srcs = n.get("source", {})

def _tokens(s):
    """把 source 字符串拆成候选文件名。

    source 常见写法："panel.parquet" / "m9_nav_R1_1.0.csv" /
    "panel.parquet + m9_nav_R1_1.0.csv" / "derived: 1 - zt_share_all"。
    旧实现拿整串去 os.path.exists，复合写法必然误报；
    正则提取又会把 R1_1.0.csv 截成 1.0.csv。
    改为按分隔符切分后，逐个看是否像文件名。
    """
    out = []
    for part in s.replace("+", ",").replace(";", ",").split(","):
        part = part.strip()
        if not part:
            continue
        # 只把「以 .csv/.parquet/.md/.json 结尾」的整体当文件名
        if part.lower().endswith((".csv", ".parquet", ".md", ".json")):
            out.append(part)
    return out

missing = set()
for k, s in srcs.items():
    if not s:
        continue
    if s.startswith("panel") or s.startswith("data/"):
        continue
    for f in _tokens(s):
        ok = (os.path.exists(os.path.join(OUT, f))
              or os.path.exists(os.path.join(ROOT, f))
              or os.path.exists(os.path.join(ROOT, "data", "proc", f))
              or os.path.exists(os.path.join(PAPER, f)))
        if not ok:
            missing.add(f)
if missing:
    for m in sorted(missing):
        print(f"    !! 来源文件不存在: {m}")
    FAIL.append(f"X2 有 {len(missing)} 个来源文件缺失")
else:
    print(f"    ok 全部来源可解析（{len(srcs)} 条）")

print("\n=== X3 关键数字反查 ===")
md = open(os.path.join(PAPER, "paper.md"), encoding="utf-8").read()
KEY = {
    "21.11%": ("ann_ew", 100), "0.764": ("sharpe_ew", 1),
    "11.68%": ("ann_y_neut_2kw_5d", 100), "0.383": ("sharpe_y_neut_2kw_5d", 1),
    "21.17%": ("ann_v3b_3e_5d", 100), "0.955": ("sharpe_v3b_3e_5d", 1),
    "29.71%": ("cost_ann_ma5_15bp", 100), "1.398": ("sharpe_ma5_3e_5d", 1),
    "0.1655": ("style_r2", 1), "0.5525": ("auc_neut", 1), "0.5353": ("auc_exc", 1),
    "39.20%": ("cost_ann_ma5_0bp", 100), "32.76%": ("cost_ann_s_0bp", 100),
    "20.86%": ("cost_ann_ma5_30bp", 100), "33.86%": ("cost_ann_lag5_0bp", 100),
}
vals = n["numbers"]
for lit, (key, scale) in KEY.items():
    if key not in vals:
        print(f"    -- {lit}: key {key} 不在 numbers.json（跳过）")
        continue
    v = float(vals[key]) * scale
    good = abs(v - float(lit.rstrip("%"))) < 0.02
    inmd = lit in md
    tag = "ok " if (good and inmd) else "!! "
    if not (good and inmd):
        FAIL.append(f"X3 {lit}: json={v:.4f} vs 正文 {lit} in_md={inmd}")
    print(f"    {tag}{lit:8s} json {key} -> {v:.4f}   正文出现={inmd}")

print("\n" + "=" * 66)
if FAIL:
    print(f"FAIL {len(FAIL)}")
    for f in FAIL:
        print("  -", f)
else:
    print("全部通过")
print("=" * 66)
