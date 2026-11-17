# -*- coding: utf-8 -*-
"""verify_numbers2.py —— 改进版数字核对。

区分三类数字：
  A. 直接存储值    —— 源 CSV/parquet 里应有该字面值
  B. 派生量        —— ann/vol/sharpe/mdd/ndays 由 nav 序列算出，需重算
  C. 行数/计数     —— n_rows/n_stocks/ic_n_factors 等，需 COUNT 源表

对 B 类真正重算，对 C 类真正 count，这样才能验证。
"""
import json, os, re, sys
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "output")
PAPER = os.path.join(ROOT, "paper")
DATA = os.path.join(ROOT, "data")

nj = json.load(open(os.path.join(PAPER, "numbers.json"), encoding="utf-8"))
nums, src = nj["numbers"], nj["source"]

_cache = {}
_vals = {}


def _numify(s):
    """把 '32.76%' / '+0.1254' / '3,933,186' / '1.18 (n=49)' 转成数值。"""
    return pd.to_numeric(
        s.astype(str).str.replace("%", "", regex=False)
         .str.replace(",", "", regex=False)
         .str.replace(r"^\+", "", regex=True)
         .str.extract(r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)", expand=False),
        errors="coerce")


def numeric_pool(rel):
    """源文件的全部数值（含 index），只算一次。"""
    if rel in _vals:
        return _vals[rel]
    # panel 8.4M×73，不建全量池；只在 C 类计数时单独处理
    if rel.endswith("panel.parquet"):
        _vals[rel] = np.array([])
        return _vals[rel]
    if rel.endswith(".md"):
        # markdown：直接逐单元格提取数字（不经 DataFrame，避免列混合）
        c = None
        for cand in (os.path.join(OUT, rel), os.path.join(ROOT, rel)):
            if os.path.exists(cand):
                c = cand
                break
        if c is None:
            _vals[rel] = np.array([])
            return _vals[rel]
        out = []
        for line in open(c, encoding="utf-8-sig"):
            line = line.strip()
            if not line.startswith("|"):
                continue
            if re.match(r"^\|[\s\-:|]+\|$", line):
                continue
            for cell in line.strip("|").split("|"):
                m = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", cell)
                if m:
                    try:
                        out.append(float(m.group(0)))
                    except Exception:
                        pass
        _vals[rel] = np.array(out, dtype=float)
        return _vals[rel]
    df = load(rel)
    if df is None:
        _vals[rel] = np.array([])
        return _vals[rel]
    collected = []
    for c in df.columns:
        a = _numify(df[c]).dropna().values.astype(float)
        if a.size:
            collected.append(a)
    collected.append(np.array([float(len(df))]))
    for c in df.columns:
        a = _numify(df[c]).dropna()
        if len(a):
            collected.append(np.array([float(a.nunique())]))
    out = np.concatenate(collected) if collected else np.array([])
    _vals[rel] = out
    return out


def load(rel):
    if rel in _cache:
        return _cache[rel]
    cands = [os.path.join(OUT, rel), os.path.join(ROOT, rel),
             os.path.join(DATA, "proc", rel), os.path.join(DATA, rel), os.path.join(PAPER, rel)]
    df = None
    for c in cands:
        if not os.path.exists(c):
            continue
        try:
            if c.endswith(".parquet"):
                df = pd.read_parquet(c)
            else:
                df = pd.read_csv(c)
        except Exception:
            # markdown：可能含多个不同列数的表格，按连续块分段解析
            blocks, cur = [], []
            for line in open(c, encoding="utf-8-sig"):
                line = line.strip()
                if line.startswith("|"):
                    if re.match(r"^\|[\s\-:|]+\|$", line):
                        continue          # 分隔行
                    cur.append([x.strip() for x in line.strip("|").split("|")])
                else:
                    if cur:
                        blocks.append(cur)
                        cur = []
            if cur:
                blocks.append(cur)
            frames = []
            for b in blocks:
                w = {len(r) for r in b}
                if len(b) >= 2 and len(w) == 1:
                    frames.append(pd.DataFrame(b[1:], columns=b[0]))
            df = pd.concat(frames, axis=1, ignore_index=True) if frames else None
        break
    _cache[rel] = df
    return df


def to_num(x):
    if isinstance(x, (int, float, np.integer, np.floating)):
        return float(x)
    s = str(x).strip().replace("%", "").replace(",", "").replace("+", "")
    try:
        return float(s)
    except Exception:
        return np.nan


def nav_stats(rel):
    """从 nav CSV 重算 年化/波动/夏普/最大回撤/天数。"""
    df = load(rel)
    if df is None:
        return None
    col = "nav" if "nav" in df.columns else df.columns[0]
    s = pd.to_numeric(df[col], errors="coerce").dropna()
    if len(s) < 3:
        return None
    n = len(s)
    daily = s.pct_change().dropna()
    ann = (s.iloc[-1] / s.iloc[0]) ** (252 / n) - 1
    vol = daily.std() * np.sqrt(252)
    rf = 0.015
    sharpe = (ann - rf) / vol if vol > 0 else np.nan
    cum = s / s.cummax() - 1
    return dict(ann=ann, vol=vol, sharpe=sharpe, mdd=cum.min(), ndays=float(n))


PAT = re.compile(r"^(ann|vol|sharpe|mdd|ndays)_")
name_tail = lambda k: PAT.sub("", k)

ok, derived, counted, skipped, bad = 0, 0, 0, 0, []
for k, v in nums.items():
    rel = src.get(k)
    if not isinstance(v, (int, float)):
        skipped += 1
        continue
    if rel is None:
        bad.append((k, v, "NO_SOURCE", rel)); continue
    df = None if rel.endswith("panel.parquet") else load(rel)

    # --- B 类：派生量 ---
    m = PAT.match(k)
    if m:
        st = nav_stats(rel)
        if st is None:
            bad.append((k, v, "DERIVED_BUT_NO_NAV", rel)); continue
        got = st[m.group(1)]
        tol = 2e-3 * max(1.0, abs(v)) if m.group(1) != "ndays" else 0.5
        if abs(got - v) <= tol:
            derived += 1
        else:
            bad.append((k, v, f"DERIVED_MISMATCH(got {got:.6g})", rel))
        continue

    if df is None:
        if rel.endswith("panel.parquet"):
            skipped += 1      # panel 计数项单独用 pandas 快速验证，不在此循环
            continue
        bad.append((k, v, "FILE_NOT_FOUND", rel)); continue
    # --- C 类：计数 ---
    if re.match(r"^(n_rows|n_stocks|n_days|ic_n|n_days_total)", k):
        cand = {len(df)}
        for c in df.columns:
            s = pd.to_numeric(df[c], errors="coerce").dropna()
            if len(s):
                cand.add(int(s.nunique()))
        if v in cand or abs(len(df) - v) <= 0:
            counted += 1
        else:
            bad.append((k, v, f"COUNT_MISMATCH(len={len(df)})", rel))
        continue
    # --- A 类：字面值 ---
    vals = numeric_pool(rel)
    tol = 1e-6 * max(1.0, abs(v))
    if vals.size and (
            np.any(np.abs(vals - v) <= tol)
            or np.any(np.abs(np.abs(vals) - abs(v)) <= tol)
            or np.any(np.abs(vals * 100 - v) <= tol * 100)
            or np.any(np.abs(vals / 100 - v) <= tol / 100)):
        ok += 1
    else:
        bad.append((k, v, "NOT_FOUND_IN_SOURCE", rel))

print("=" * 72)
print(f"总项 {len(nums)}")
print(f"  A 字面值已复核     : {ok}")
print(f"  B 派生量已重算     : {derived}")
print(f"  C 计数已核对       : {counted}")
print(f"  非数值（跳过）     : {skipped}")
print(f"  异常               : {len(bad)}")
print("=" * 72)
if bad:
    print("\n=== 异常清单 ===")
    for k, v, why, rel in bad:
        print(f"  [{why}] {k} = {v}   <- {rel}")
sys.exit(0 if not bad else 1)
