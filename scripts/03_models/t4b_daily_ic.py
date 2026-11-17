# -*- coding: utf-8 -*-
"""t4b_daily_ic.py —— 把 t4 的「5 日重叠标签」IC 换算成「无重叠」口径

问题：标签 fwd5 在相邻 5 天里高度重叠（同一段未来收益被算 5 次），
      于是 daily IC 序列的 t 统计量被重复计数抬高约 √5 倍。
做法：只取每 5 个交易日中的 1 天（stride=5），得到真正独立的样本外 IC 序列，
      再算 mean / std / t。对模型无任何改动，只改评估口径。
产出：output/m4b_ic_nonoverlap.csv + 打印表
"""
import os, sys
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
OUT = os.path.join(ROOT, "output")

import t4_lgb


def stratify(dates_sorted, stride=5):
    """每 stride 个交易日取一天（从最后一天往前取，保证最近一天一定入选）"""
    d = np.array(sorted(pd.unique(dates_sorted)))
    keep = d[::-1][::stride][::-1]
    return set(pd.DatetimeIndex(keep))


def daily_ic(dates, pred, ret, min_n=20):
    d = pd.DataFrame({"d": dates, "p": pred, "r": ret})
    d = d[np.isfinite(d["r"]) & np.isfinite(d["p"])]
    if len(d) == 0:
        return pd.Series(dtype=float, index=pd.DatetimeIndex([]))
    d["pr"] = d.groupby("d")["p"].rank()
    d["rr"] = d.groupby("d")["r"].rank()
    d["pr"] = d["pr"] - d.groupby("d")["pr"].transform("mean")
    d["rr"] = d["rr"] - d.groupby("d")["rr"].transform("mean")
    d["num"] = d["pr"] * d["rr"]
    g = d.groupby("d")
    num = g["num"].sum()
    den = np.sqrt(g["pr"].apply(lambda s: (s ** 2).sum()) * g["rr"].apply(lambda s: (s ** 2).sum()))
    n = g.size()
    ic = (num / den)[n >= min_n]
    return ic.dropna()


def main():
    import lightgbm as lgb
    df = t4_lgb.load()
    fc = t4_lgb.feat_cols(df)
    print(f"[t4b] panel {df.shape}  feats {len(fc)}")
    rows = []
    for lb in ("y_raw", "y_exc", "y_exc_nzt"):
        mp = os.path.join(ROOT, "models", f"lgb_{lb}.txt")
        if not os.path.exists(mp):
            print(f"  [skip] {lb} 模型不存在")
            continue
        bst = lgb.Booster(model_file=mp)
        d = df[df[lb].notna()]
        for span, (a, b) in {"valid2023": ("2023-01-01", "2023-12-31"),
                             "test2024-25": ("2024-01-01", "2026-12-31")}.items():
            s = d[(d["date"] >= a) & (d["date"] <= b)]
            if len(s) == 0:
                continue
            pr = bst.predict(s[fc])
            ic5 = daily_ic(s["date"].values, pr, s["fwd5"].values)
            # 无重叠：按日排序后每 5 天取 1 天
            keep = stratify(s["date"].values, 5)
            ico = ic5[ic5.index.isin(keep)]
            base = ic5.mean()
            rows.append({
                "label": lb, "span": span,
                "ic_overlap": base, "icir_overlap": base / ic5.std(), "n_days_overlap": len(ic5),
                "t_overlap_naive": base / ic5.std() * np.sqrt(len(ic5)),
                "ic_nonoverlap": ico.mean(), "icir_nonoverlap": ico.mean() / ico.std(),
                "n_days_nonoverlap": len(ico),
                "t_nonoverlap": ico.mean() / ico.std() * np.sqrt(len(ico)),
                "p_two_sided": None,  # 用 scipy 另算，避免 math.erf 拼写问题
            })
            print(f"  {lb:10s} {span:12s} IC={base:+.4f} (n={len(ic5)}d, 重叠)  "
                  f"→ 无重叠 n={len(ico)}d IC={ico.mean():+.4f} t={rows[-1]['t_nonoverlap']:.2f}")
    from scipy import stats as st
    for r in rows:
        r["p_two_sided"] = float(2 * (1 - st.norm.cdf(abs(r["t_nonoverlap"]))))
    R = pd.DataFrame(rows)
    R.to_csv(os.path.join(OUT, "m4b_ic_nonoverlap.csv"), index=False)
    print("\n" + R.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
