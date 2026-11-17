# -*- coding: utf-8 -*-
"""verify_numbers.py —— 对 numbers.json 的每个数字，回到源文件独立重算并比对。

MEMORY.md 2026-11-15 新规则：
「数字必须在 numbers.json 里有源文件路径；没有路径当错处理」。
本脚本再进一步：**有路径也要验证路径里的数确实等于该数**。

做法：每个源 CSV 读进来，列出全部数值列的可选值集合，
检查 numbers.json 的值能否在某一列中找到（容差 1e-6 相对）。
找不到 → 报告「无法在源文件复现」，人工核查。
"""
import json, os, sys
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "output")
PAPER = os.path.join(ROOT, "paper")
DATA = os.path.join(ROOT, "data")

nj = json.load(open(os.path.join(PAPER, "numbers.json"), encoding="utf-8"))
nums = nj["numbers"]
src = nj["source"]

_cache = {}


def load(rel):
    """按 source 字段解析到实际文件路径（CSV / parquet）。"""
    if rel in _cache:
        return _cache[rel]
    cands = [os.path.join(OUT, rel), os.path.join(ROOT, rel),
             os.path.join(DATA, rel), os.path.join(DATA, "proc", rel),
             os.path.join(DATA, "raw", rel), os.path.join(PAPER, rel)]
    for c in cands:
        if os.path.exists(c):
            try:
                df = pd.read_parquet(c) if c.endswith(".parquet") else pd.read_csv(c)
            except Exception:
                # 非标准 CSV（如 markdown 表格）：用宽松解析，取所有能转数的列
                try:
                    df = pd.read_csv(c, sep=None, engine="python", on_bad_lines="skip")
                except Exception:
                    _cache[rel] = None
                    return None
            _cache[rel] = df
            return df
    _cache[rel] = None
    return None


def all_numeric_values(df):
    """收集 DataFrame 里所有数值（含 index），返回 np.array。"""
    vals = []
    for c in df.columns:
        s = df[c]
        if pd.api.types.is_numeric_dtype(s):
            vals.append(pd.to_numeric(s, errors="coerce").dropna().values)
    out = np.concatenate(vals) if vals else np.array([])
    return out


ok = miss_file = cannot_repro = nonnum = 0
problems = []
skipped_files = set()

for k, v in nums.items():
    rel = src.get(k)
    if rel is None:
        problems.append((k, v, "NO_SOURCE", ""))
        miss_file += 1
        continue
    if not isinstance(v, (int, float)):
        nonnum += 1          # 字符串型（如因子名、日期），跳过数值比对
        continue
    df = load(rel)
    if df is None:
        problems.append((k, v, "FILE_NOT_FOUND", rel))
        miss_file += 1
        skipped_files.add(rel)
        continue
    arr = all_numeric_values(df)
    if arr.size == 0:
        problems.append((k, v, "NO_NUMERIC_COL", rel))
        cannot_repro += 1
        continue
    # 相对容差比对
    tol = 1e-6 * max(1.0, abs(v))
    if np.any(np.abs(arr - v) <= tol):
        ok += 1
    else:
        # 尝试常见变换：×100 / ÷100（百分比存成小数）
        hit = False
        for f in (100.0, 0.01, -1.0):
            if np.any(np.abs(arr * f - v) <= tol * max(1.0, abs(v))):
                hit = True
                break
        if hit:
            ok += 1
        else:
            problems.append((k, v, "NOT_FOUND_IN_SOURCE", rel))
            cannot_repro += 1

print("=" * 70)
print(f"numbers.json 共 {len(nums)} 项")
print(f"  在源文件中复现      : {ok}")
print(f"  非数值（跳过比对）  : {nonnum}")
print(f"  源文件缺失          : {miss_file}")
print(f"  无法在源文件中复现  : {cannot_repro}")
print("=" * 70)

if problems:
    print("\n=== 问题清单 ===")
    for k, v, why, rel in problems:
        print(f"  [{why}] {k} = {v}   <- {rel}")

if skipped_files:
    print("\n=== 找不到的源文件 ===")
    for f in sorted(skipped_files):
        print("  ", f)

sys.exit(0 if (miss_file == 0 and cannot_repro == 0) else 1)
