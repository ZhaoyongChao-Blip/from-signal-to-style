# -*- coding: utf-8 -*-
"""
t3_features.py —— 因子构造 + 标签，输出 panel.parquet（内存优化版）

数据来源（两层，分工明确）:
  data/raw/daily/<code>.parquet  fqt=0 真实成交价 → 回测按此成交、用于「涨停不可买」判断
  data/raw/adj/<code>.parquet    fqt=2 后复权价   → adj_factor = adj_close / raw_close
                                 → 计算收益用 adj_close，避免除权跳空被当成暴跌

质量过滤（每档记录被剔除的股票数）:
  A. 有效行数 < 400                          → 剔除
  B. 单日 |涨跌幅| > 涨跌停 + 2pp 的行 > 2 条 → 剔除该股
  C. 交易日覆盖率 < 60%                      → 剔除

⚠️ 2026-11-11 修：最初版全 float64 一次 concat 800 万行 × 60 列 → 需 ~4GB/份，
   裁剪 + clip + concat 各复制一份，峰值 5.6GB 后 OOM（`Unable to allocate 64.4 MiB`）。
   现改为：feature 一律 float32，分块 concat，再一次性写盘；峰值 < 2GB。
"""
import os, glob, sys, json, gc
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(ROOT, "data", "raw")
D_DAILY = os.path.join(RAW, "daily")
D_ADJ = os.path.join(RAW, "adj")
D_IDX = os.path.join(RAW, "index")
OUTD = os.path.join(ROOT, "data", "proc")
os.makedirs(OUTD, exist_ok=True)

MIN_ROWS = 400
MIN_COVER = 0.60
F32 = np.float32


def board_of(code):
    if code[:3] in ("600", "601", "603", "605"):
        return "SH_MAIN"
    if code[:3] in ("000", "001", "002", "003"):
        return "SZ_MAIN"
    if code[:3] in ("300", "301"):
        return "GEM"
    if code[:2] == "68":
        return "STAR"
    return "BJ"


def lim_of(board):
    return 0.20 if board in ("GEM", "STAR") else 0.10


def rsi(close, n=14):
    d = close.diff()
    up = d.clip(lower=0).rolling(n).mean()
    dn = (-d.clip(upper=0)).rolling(n).mean()
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))


def build_one(code, cal_dates_all):
    n_calendar = len(cal_dates_all)
    fp = os.path.join(D_DAILY, f"{code}.parquet")
    ap = os.path.join(D_ADJ, f"{code}.parquet")
    if not (os.path.exists(fp) and os.path.exists(ap)):
        return None, "missing"
    df = pd.read_parquet(fp)
    if len(df) < MIN_ROWS:
        return None, "short"
    adj = pd.read_parquet(ap)
    df["date"] = pd.to_datetime(df["date"])
    adj["date"] = pd.to_datetime(adj["date"])
    d = df.merge(adj, on="date", how="left").sort_values("date").reset_index(drop=True)
    d["adj_close"] = d["adj_close"].ffill()
    d["factor"] = d["adj_close"] / d["close"]

    board = board_of(code)
    lim = lim_of(board)
    if d["date"].nunique() / max(n_calendar, 1) < MIN_COVER:
        return None, "low_coverage"

    chg = d["adj_close"].pct_change()
    bad = (chg.abs() > lim + 0.02) & (d.index > 0)
    if bad.sum() > 2:
        return None, "abnormal"
    if bad.sum() > 0:
        d = d.loc[~bad].reset_index(drop=True)

    c = d["adj_close"].astype(float)
    fac = d["factor"].astype(float)
    o = d["open"].astype(float) * fac
    h = d["high"].astype(float) * fac
    l = d["low"].astype(float) * fac
    v = d["volume"].astype(float)
    amt = d["amount"].astype(float)
    to = d["turnover"].astype(float)
    r = c.pct_change()

    cols = {}
    cols["date"] = d["date"]
    cols["code"] = code
    cols["board"] = board
    cols["raw_open"] = d["open"].astype(F32)
    cols["raw_close"] = d["close"].astype(F32)
    cols["adj_open"] = o.astype(F32)
    cols["adj_close"] = c.astype(F32)
    cols["high"] = h.astype(F32)
    cols["low"] = l.astype(F32)
    cols["amount"] = amt.astype(F32)
    cols["turnover"] = to.astype(F32)
    cols["volume"] = v.astype(F32)

    for n in (1, 5, 10, 20, 60, 120):
        cols[f"ret_{n}"] = (c / c.shift(n) - 1).astype(F32)
    for n in (5, 10, 20, 60):
        ma = c.rolling(n).mean()
        cols[f"ma_dev_{n}"] = (c / ma - 1).astype(F32)
        cols[f"ma_slope_{n}"] = (ma / ma.shift(5) - 1).astype(F32)
    cols["ma_cross_5_20"] = (c.rolling(5).mean() > c.rolling(20).mean()).astype(F32)
    cols["ma_cross_20_60"] = (c.rolling(20).mean() > c.rolling(60).mean()).astype(F32)

    for n in (5, 20, 60):
        cols[f"vol_{n}"] = r.rolling(n).std().astype(F32)
    cols["vol_ratio_5_60"] = (r.rolling(5).std() / r.rolling(60).std().replace(0, np.nan)).astype(F32)
    cols["hl_range"] = ((h - l) / c).astype(F32)
    cols["hl_range_20"] = ((h - l) / c).rolling(20).mean().astype(F32)
    cols["upper_shadow"] = ((h - np.maximum(o, c)) / c).astype(F32)
    cols["lower_shadow"] = ((np.minimum(o, c) - l) / c).astype(F32)
    cols["dd_20"] = (c / c.rolling(20).max() - 1).astype(F32)
    cols["dd_60"] = (c / c.rolling(60).max() - 1).astype(F32)
    cols["skew_20"] = r.rolling(20).skew().astype(F32)
    cols["kurt_20"] = r.rolling(20).kurt().astype(F32)

    vmean5 = v.rolling(5).mean().replace(0, np.nan)
    vmean20 = v.rolling(20).mean().replace(0, np.nan)
    cols["vol_ma_ratio_5"] = (v / vmean5).astype(F32)
    cols["vol_ma_ratio_20"] = (v / vmean20).astype(F32)
    cols["turn_ma_5"] = to.rolling(5).mean().astype(F32)
    cols["turn_ma_20"] = to.rolling(20).mean().astype(F32)
    cols["turn_ratio"] = (to / to.rolling(20).mean().replace(0, np.nan)).astype(F32)
    cols["corr_pv_20"] = r.rolling(20).corr(v.pct_change()).astype(F32)
    cols["up_vol_ratio_20"] = (v.where(r > 0, 0).rolling(20).sum()
                               / v.rolling(20).sum().replace(0, np.nan)).astype(F32)
    cols["amt_ma_20"] = amt.rolling(20).mean().astype(F32)

    cols["rsi_14"] = rsi(c, 14).astype(F32)
    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    dif = ema12 - ema26
    dea = dif.ewm(span=9, adjust=False).mean()
    cols["macd_dif"] = (dif / c).astype(F32)
    cols["macd_hist"] = ((dif - dea) * 2 / c).astype(F32)
    low9, high9 = l.rolling(9).min(), h.rolling(9).max()
    rsv = (c - low9) / (high9 - low9).replace(0, np.nan) * 100
    k = rsv.ewm(com=2, adjust=False).mean()
    cols["kdj_k"] = k.astype(F32)
    cols["kdj_d"] = k.ewm(com=2, adjust=False).mean().astype(F32)
    mid, std = c.rolling(20).mean(), c.rolling(20).std()
    cols["boll_pos"] = ((c - mid) / (2 * std).replace(0, np.nan)).astype(F32)
    cols["boll_width"] = ((4 * std) / mid).astype(F32)

    cols["log_price"] = np.log(c).astype(F32)
    cols["log_amt"] = np.log1p(amt).astype(F32)
    cols["log_turn"] = np.log1p(to).astype(F32)

    cols["lim"] = F32(lim)
    cols["is_zt"] = (d["pct_chg"] >= lim * 100 - 0.6).astype(np.int8)
    cols["is_dt"] = (d["pct_chg"] <= -(lim * 100 - 0.6)).astype(np.int8)

    # 标签：先对齐到全市场交易日历再 shift，否则上方局部剔除异常行会让 shift(-5) 跳期
    dti = pd.DatetimeIndex(d["date"])
    cal_i = pd.DatetimeIndex(cal_dates_all)
    c_al = pd.Series(c.values, index=dti).reindex(cal_i)
    f1 = (c_al.shift(-1) / c_al - 1)
    f5 = (c_al.shift(-5) / c_al - 1)
    f10 = (c_al.shift(-10) / c_al - 1)
    zt_al = pd.Series(cols["is_zt"].to_numpy(), index=dti).reindex(cal_i)
    ztf = pd.concat([zt_al.shift(-kk) for kk in range(1, 6)], axis=1).fillna(0).max(axis=1)
    cols["fwd1"] = f1.reindex(dti).values.astype(F32)
    cols["fwd5"] = f5.reindex(dti).values.astype(F32)
    cols["fwd10"] = f10.reindex(dti).values.astype(F32)
    cols["fwd5_has_zt"] = ztf.reindex(dti).fillna(0).values.astype(np.int8)
    cols["has_fwd5"] = f5.reindex(dti).notna().values.astype(np.int8)

    f = pd.DataFrame(cols)
    return f, "ok"


FEAT_START = "ret_1"


def main():
    codes = [os.path.basename(p)[:-8] for p in sorted(glob.glob(os.path.join(D_DAILY, "*.parquet")))]
    cal = pd.to_datetime(pd.read_parquet(os.path.join(D_IDX, "000300.parquet"))["date"])
    n_calendar = len(cal)
    print(f"[feat] {len(codes)} stocks, calendar={n_calendar} days", flush=True)

    parts, stats = [], {}
    for i, c in enumerate(codes, 1):
        f, why = build_one(c, cal)
        stats[why] = stats.get(why, 0) + 1
        if f is not None:
            parts.append(f)
        if i % 800 == 0:
            print(f"  {i}/{len(codes)} {stats} mem={sum(p.memory_usage(deep=True).sum() for p in parts)/1e9:.2f}GB", flush=True)
    print(f"[feat] build stats: {stats}", flush=True)

    feat = pd.concat(parts, ignore_index=True, copy=False)
    del parts
    gc.collect()
    print(f"[feat] concat done shape={feat.shape} mem={feat.memory_usage(deep=True).sum()/1e9:.2f}GB", flush=True)

    feat = feat[feat["date"].isin(set(cal))].copy()
    feat["code"] = feat["code"].astype("category")
    feat["board"] = feat["board"].astype("category")

    mkt = feat.groupby("date", observed=True)["fwd5"].mean().rename("mkt_fwd5")
    feat["mkt_fwd5"] = feat["date"].map(mkt).astype(F32)
    feat["excess5"] = (feat["fwd5"] - feat["mkt_fwd5"]).astype(F32)

    for cname in ("000300", "000905", "000852"):
        p = os.path.join(D_IDX, f"{cname}.parquet")
        if os.path.exists(p):
            d = pd.read_parquet(p).sort_values("date")
            d["date"] = pd.to_datetime(d["date"])
            d = d.assign(**{f"idx_{cname}_fwd5": (d["close"].shift(-5) / d["close"] - 1).astype(F32)})[
                ["date", f"idx_{cname}_fwd5"]]
            feat[f"idx_{cname}_fwd5"] = feat["date"].map(d.set_index("date")[f"idx_{cname}_fwd5"]).astype(F32)

    skip = {"fwd1", "fwd5", "fwd10", "excess5", "mkt_fwd5", "adj_close", "adj_open",
            "raw_open", "raw_close", "high", "low", "amount", "turnover", "volume",
            "lim", "is_zt", "is_dt", "fwd5_has_zt", "has_fwd5"}
    num_cols = [c for c in feat.columns
                if feat[c].dtype.kind == "f" and c not in skip and not c.startswith("idx_")]
    print(f"[feat] normalizing {len(num_cols)} factors", flush=True)

    # 逐列截面 z-score（用 numpy bincount，避免 groupby.transform 的巨量中间复制）
    codes_dt, _ = pd.factorize(feat["date"], sort=False)
    cnt = np.bincount(codes_dt).astype(np.float64)
    for c in num_cols:
        x = feat[c].to_numpy(dtype=np.float64, copy=False)
        ok = ~np.isnan(x)
        xz = np.where(ok, x, 0.0)
        cs = np.bincount(codes_dt, weights=xz)
        cq = np.bincount(codes_dt, weights=xz * xz)
        n = np.maximum(cnt, 1)
        mu = cs / n
        var = np.maximum(cq / n - mu * mu, 0)
        sd = np.sqrt(var)
        sd[sd == 0] = np.nan
        z = (x - mu[codes_dt]) / sd[codes_dt]
        z[~ok] = np.nan
        feat[c] = np.clip(z, -10, 10).astype(F32)
        del x, xz, z
    gc.collect()

    out = os.path.join(OUTD, "panel.parquet")
    feat.to_parquet(out, index=False)
    print(f"[feat] saved {out} shape={feat.shape} factors={len(num_cols)}", flush=True)
    print(f"[feat] dates {feat['date'].min().date()} ~ {feat['date'].max().date()}, "
          f"stocks={feat['code'].nunique()}", flush=True)
    with open(os.path.join(OUTD, "feat_meta.json"), "w", encoding="utf-8") as fh:
        json.dump({"stats": stats, "n_factors": len(num_cols), "factors": num_cols,
                   "shape": list(feat.shape)}, fh, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
