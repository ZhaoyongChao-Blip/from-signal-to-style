# -*- coding: utf-8 -*-
"""t7b_neut_fast.py —— 风格中性化（快速版，分块避免 OOM）

与 t7_neutral.py 同一问题，但改成**按日期分块**处理：
  1. 逐块（按日期区间）读面板 → 对块内每一天算截面残差 → 立刻算当天的 IC → 丢弃
  2. 全程只持有一个块（约 50 万行），峰值内存 < 1GB

三个产出：
  N0  4 个风格因子对未来 5 日超额收益的截面 R²（逐日平均）
  N2  46 个因子「对风格取残差后」的单因子 IC（对比原始 IC）
  N1  用「风格中性化后的超额」当标签重训 LGBM（这一项仍需全量内存，单独跑 t7c）

口径（写死）：
  · 风格因子纯前定：log_amt / log_price / vol_20 / turn_ma_20，都是 t 日收盘可见
  · 残差 = 逐日 OLS 残差（含截距）
  · IC 用逐日 Spearman；另给 stride=5 的无重叠 t 值
"""
import os, sys, gc
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
OUT = os.path.join(ROOT, "output")
STYLE = ["log_amt", "log_price", "vol_20", "turn_ma_20"]
L = []


def P(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    L.append(s)


def resid_block(y, X, g, ncol):
    """块内逐日截面回归 y ~ 1 + X（X 已保证 ncol 列）→ 残差。"""
    k = 1 + ncol
    n = int(g.max()) + 1
    ok = np.isfinite(y) & np.isfinite(X).all(axis=1)
    one = np.ones(len(y))
    Xa = np.empty((len(y), k))
    Xa[:, 0] = one
    Xa[:, 1:] = X
    Xa[~ok] = 0.0
    yy = np.where(ok, y, 0.0)
    A = np.zeros((n, k, k))
    b = np.zeros((n, k))
    for i in range(k):
        for j in range(i, k):
            v = np.bincount(g, weights=Xa[:, i] * Xa[:, j], minlength=n)
            A[:, i, j] = v
            A[:, j, i] = v
        b[:, i] = np.bincount(g, weights=Xa[:, i] * yy, minlength=n)
    cnt = np.bincount(g, weights=ok.astype(np.float64), minlength=n)
    A[:, np.arange(k), np.arange(k)] += 1e-6 * np.maximum(cnt, 1)[:, None]
    beta = np.linalg.solve(A, b[:, :, None])[:, :, 0]
    r = yy - (Xa * beta[g]).sum(axis=1)
    r[~ok] = np.nan
    return r


def daily_ic_arrays(d_codes, pred, ret, min_n=20):
    """返回 (日期 codes, ic 数组)，用 numpy 逐组算 Spearman"""
    ok = np.isfinite(pred) & np.isfinite(ret)
    d = d_codes[ok]
    p = pred[ok]
    r = ret[ok]
    df = pd.DataFrame({"d": d, "p": p, "r": r})
    g = df.groupby("d")
    n = g.size()
    rr = df.groupby("d")[["p", "r"]].rank()
    rr["p"] -= rr.groupby(df["d"])["p"].transform("mean")
    rr["r"] -= rr.groupby(df["d"])["r"].transform("mean")
    rr["num"] = rr["p"] * rr["r"]
    den = np.sqrt(rr.groupby(df["d"])["p"].apply(lambda s: (s ** 2).sum())
                  * rr.groupby(df["d"])["r"].apply(lambda s: (s ** 2).sum()))
    ic = (rr.groupby(df["d"])["num"].sum() / den)[n >= min_n]
    return ic.dropna()


def main():
    import t4_lgb
    cols = ["date", "code", "fwd5", "excess5", "has_fwd5"]
    base = pd.read_parquet(os.path.join(ROOT, "data", "proc", "panel.parquet"))
    fc = [c for c in t4_lgb.feat_cols(base) if c not in STYLE]   # 不含风格因子的 42 个
    all_fc = fc + STYLE
    dates = np.sort(base["date"].unique())
    P(f"# 风格中性化（快速版）\n")
    P(f"- 总行数 {len(base):,}，日期 {len(dates)} 天")
    P(f"- 检验的因子：46 个 = 42 个非风格因子 + 4 个风格因子本身")
    del base
    gc.collect()

    # ---------- 按块处理 ----------
    use = cols + all_fc
    CH = 250000
    srt = pd.read_parquet(os.path.join(ROOT, "data", "proc", "panel.parquet"), columns=use)
    srt = srt[srt["has_fwd5"] == 1].sort_values("date", kind="stable").reset_index(drop=True)
    P(f"- 有效行数（有 fwd5）{len(srt):,}")

    r2_list = []
    ic_acc = {c: [] for c in all_fc}
    ic_idx = {c: [] for c in all_fc}
    for i in range(0, len(srt), CH):
        blk = srt.iloc[i:i + CH]
        g = pd.factorize(blk["date"])[0]
        k = int(g.max()) + 1
        X = blk[STYLE].to_numpy(dtype=np.float64)
        y = blk["excess5"].to_numpy(dtype=np.float64)
        f5 = blk["fwd5"].to_numpy(dtype=np.float64)
        rc = resid_block(y, X, g, len(STYLE))
        # 逐日 R²
        gc_ = pd.Series(g)
        ss_res = pd.Series(rc ** 2).groupby(gc_).sum()
        ss_tot = pd.Series((y - pd.Series(y).groupby(gc_).transform("mean")) ** 2).groupby(gc_).sum()
        r2 = (1 - ss_res / ss_tot).replace([np.inf, -np.inf], np.nan).dropna()
        r2_list.append(pd.Series(r2.values, index=pd.Index(np.unique(g), name="g") + i))
        for c in all_fc:
            xv = blk[c].to_numpy(dtype=np.float64)
            r = resid_block(xv, X, g, len(STYLE))
            ic = daily_ic_arrays(g, r, f5)
            ic_acc[c].append(ic.values)
            ic_idx[c].append(ic.index.values)
        del blk, X, y, f5, rc
        gc.collect()
        print(f"  block {i//CH+1}/{(len(srt)+CH-1)//CH} done", flush=True)

    r2_all = pd.concat(r2_list).groupby(level=0).first()
    P(f"\n## N0 4 个风格因子对未来 5 日超额收益的截面 R²\n")
    P(f"- 逐日 R² 均值 = **{r2_all.mean():.4f}**，中位数 {r2_all.median():.4f}，"
      f"分位 10%/90% = {r2_all.quantile(.1):.3f} / {r2_all.quantile(.9):.3f}")
    P(f"- 覆盖 {len(r2_all)} 个交易日\n")

    # 汇总 IC
    rows = []
    for c in all_fc:
        v = np.concatenate(ic_acc[c])
        idx = np.concatenate(ic_idx[c])
        s = pd.Series(v, index=idx).sort_index()
        n_stride = s.iloc[::-1][::5]
        rows.append({"factor": c, "IC_neut": s.mean(), "ICIR_neut": s.mean() / s.std(),
                     "n_days": len(s), "IC_stride5": n_stride.mean(),
                     "t_stride5": n_stride.mean() / n_stride.std() * np.sqrt(len(n_stride))})
    N = pd.DataFrame(rows).set_index("factor")

    from d2_fast import factor_daily_ic
    sub = srt[["date", "fwd5"] + all_fc] if False else None
    # 原始 IC 直接读上轮结果（同一口径 fwd5）——但那是 46 因子含风格，列名 IC
    raw_p = os.path.join(OUT, "m6_factor_ic.csv")
    if os.path.exists(raw_p):
        raw = pd.read_csv(raw_p).set_index("factor")[["IC", "ICIR"]]
        N = N.join(raw, how="left")
        N["风格成分占比"] = 1 - N["IC_neut"] / N["IC"]

    P("\n## N2 特征中性化后的单因子 IC\n")
    P("> `IC_neut` = 因子对 4 个风格因子逐日截面回归取残差后，与 fwd5 的 Spearman IC。\n")
    show = N[["IC", "IC_neut", "风格成分占比", "IC_stride5", "t_stride5"]].sort_values(
        "IC_neut", key=abs, ascending=False)
    P(show.head(20).round(4).to_string())
    P(f"\n- 原始 |IC| 均值（46 因子）：{N['IC'].abs().mean():.4f}")
    P(f"- 中性化后 |IC| 均值：{N['IC_neut'].abs().mean():.4f}")
    P(f"- 中性化后 |IC| > 0.02 的因子：{(N['IC_neut'].abs() > 0.02).sum()} 个 / {len(N)}")
    P(f"- 中性化后 |t_stride5| > 2 的因子：{(N['t_stride5'].abs() > 2).sum()} 个")
    N.to_csv(os.path.join(OUT, "m7b_factor_ic_neutral.csv"))
    P(f"- 风格因子本身中性化后 IC：{N.loc[STYLE,'IC_neut'].round(4).to_dict()}")

    with open(os.path.join(OUT, "m7b_neutral.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    print(f"[saved] {os.path.join(OUT,'m7b_neutral.md')}")


if __name__ == "__main__":
    main()
