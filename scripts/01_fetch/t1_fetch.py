# -*- coding: utf-8 -*-
"""
t1_fetch.py —— 全市场日线 + 基准指数，带缓存，断点续跑。

用法:
    python t1_fetch.py list                # 只拉股票列表
    python t1_fetch.py daily --workers 8   # 拉全部日线（断点续跑，已存在则跳过）
    python t1_fetch.py index               # 拉基准指数
    python t1_fetch.py check               # 统计覆盖率

目录:
    data/raw/list.csv
    data/raw/daily/<code>.parquet
    data/raw/index/<code>.parquet
"""
import os, sys, time, argparse, traceback
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(ROOT, "data", "raw")
D_DAILY = os.path.join(RAW, "daily")
D_IDX = os.path.join(RAW, "index")
for d in (RAW, D_DAILY, D_IDX):
    os.makedirs(d, exist_ok=True)

START = "20140101"
END = "20251110"

COLS = {"日期": "date", "股票代码": "code", "开盘": "open", "收盘": "close",
        "最高": "high", "最低": "low", "成交量": "volume", "成交额": "amount",
        "振幅": "amplitude", "涨跌幅": "pct_chg", "涨跌额": "chg", "换手率": "turnover"}


def get_list():
    import akshare as ak
    p = os.path.join(RAW, "list.csv")
    df = ak.stock_info_a_code_name()
    df.columns = ["code", "name"]
    df["code"] = df["code"].astype(str).str.zfill(6)
    df = df.drop_duplicates("code").sort_values("code").reset_index(drop=True)
    df.to_csv(p, index=False, encoding="utf-8-sig")
    print(f"[list] {len(df)} stocks -> {p}")
    return df


def load_list():
    p = os.path.join(RAW, "list.csv")
    if not os.path.exists(p):
        return get_list()
    return pd.read_csv(p, dtype={"code": str})


def fetch_one(code):
    import akshare as ak
    out = os.path.join(D_DAILY, f"{code}.parquet")
    if os.path.exists(out) and os.path.getsize(out) > 200:
        return "skip"
    for attempt in range(3):
        try:
            df = ak.stock_zh_a_hist(symbol=code, period="daily",
                                    start_date=START, end_date=END, adjust="qfq")
            if df is None or len(df) == 0:
                return "empty"
            df = df.rename(columns=COLS)
            df["code"] = code
            df["date"] = pd.to_datetime(df["date"])
            df = df[["date", "code", "open", "high", "low", "close",
                     "volume", "amount", "amplitude", "pct_chg", "turnover"]]
            df = df.sort_values("date").reset_index(drop=True)
            df.to_parquet(out, index=False)
            return "ok"
        except Exception as e:
            if attempt == 2:
                return f"err:{type(e).__name__}:{str(e)[:60]}"
            time.sleep(1.5 * (attempt + 1))
    return "err"


def fetch_daily(workers=8):
    from concurrent.futures import ThreadPoolExecutor, as_completed
    lst = load_list()
    codes = lst["code"].tolist()
    todo = [c for c in codes
            if not (os.path.exists(os.path.join(D_DAILY, f"{c}.parquet"))
                    and os.path.getsize(os.path.join(D_DAILY, f"{c}.parquet")) > 200)]
    print(f"[daily] total={len(codes)} todo={len(todo)} workers={workers}", flush=True)
    stats = {}
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(fetch_one, c): c for c in todo}
        for i, f in enumerate(as_completed(futs), 1):
            r = f.result()
            key = r.split(":")[0]
            stats[key] = stats.get(key, 0) + 1
            if i % 200 == 0 or i == len(todo):
                el = time.time() - t0
                rate = i / max(el, 1e-6)
                eta = (len(todo) - i) / max(rate, 1e-6)
                print(f"  {i}/{len(todo)}  {stats}  {rate:.1f}/s  ETA {eta/60:.1f}min", flush=True)
    print("[daily] done", stats)


def fetch_index():
    import akshare as ak
    idx = {"000300": "sh000300", "000905": "sh000905", "000852": "sh000852",
           "000001": "sh000001", "399006": "sz399006"}
    for code, sym in idx.items():
        out = os.path.join(D_IDX, f"{code}.parquet")
        try:
            df = ak.stock_zh_index_daily_em(symbol=sym)
            df["date"] = pd.to_datetime(df["date"])
            df = df[(df["date"] >= pd.Timestamp("2014-01-01")) & (df["date"] <= pd.Timestamp("2025-11-10"))]
            df[["date", "open", "close", "high", "low", "volume", "amount"]].sort_values("date").to_parquet(out, index=False)
            print(f"[idx] {code} {len(df)} rows -> {out}")
        except Exception as e:
            print(f"[idx] {code} ERR {repr(e)[:80]}")


def check():
    lst = load_list()
    files = [f for f in os.listdir(D_DAILY) if f.endswith(".parquet")]
    print(f"list={len(lst)} files={len(files)} coverage={len(files)/len(lst):.1%}")
    rows = 0
    for f in files[:50]:
        rows += len(pd.read_parquet(os.path.join(D_DAILY, f)))
    print(f"avg rows (first 50) = {rows/50:.0f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["list", "daily", "index", "check"])
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    if a.cmd == "list":
        get_list()
    elif a.cmd == "daily":
        fetch_daily(a.workers)
    elif a.cmd == "index":
        fetch_index()
    else:
        check()
