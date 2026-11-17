# -*- coding: utf-8 -*-
"""
t1c_fetch_adj.py —— 拉后复权序列（fqt=2），用于计算本地复权因子。

为什么：
  fqt=0（不复权）价格在除权日会跳空（000002 2026 年 3.19 vs 2014 年 7.99），
  直接用会把除权当暴跌；fqt=1（前复权）在东财接口上跨除权日出现错乱（见 t1b 注释）。
  → 落盘用 fqt=0（真实成交价，回测按此成交才对），复权因子 = hfq_close / raw_close。
     这个因子天然只含「已发生」的除权信息，不含未来。

产出: data/raw/adj/<code>.parquet  列: date, adj_close, factor
"""
import os, time, random
import pandas as pd
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

ROOT = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(ROOT, "data", "raw")
D_DAILY = os.path.join(RAW, "daily")
D_ADJ = os.path.join(RAW, "adj")
os.makedirs(D_ADJ, exist_ok=True)

URL = "http://push2his.eastmoney.com/api/qt/stock/kline/get"
FIELDS = ["date", "open", "close", "high", "low", "volume", "amount", "amplitude", "pct_chg", "chg", "turnover"]

SESS = requests.Session()
SESS.headers.update({
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120.0 Safari/537.36",
    "Referer": "https://quote.eastmoney.com/",
})


def secid(code):
    return ("1." if code[:3] in ("600", "601", "603", "605", "688", "689") else "0.") + code


def fetch_one(code, retries=3):
    out = os.path.join(D_ADJ, f"{code}.parquet")
    if os.path.exists(out) and os.path.getsize(out) > 100:
        return "skip"
    p = {"fields1": "f1,f2,f3,f4,f5,f6",
         "fields2": "f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61",
         "klt": "101", "fqt": "2", "beg": "20140101", "end": "20991231", "lmt": "100000",
         "secid": secid(code)}
    for k in range(retries):
        try:
            j = SESS.get(URL, params=p, timeout=12).json()
            kl = (j.get("data") or {}).get("klines") or []
            if not kl:
                return "empty"
            df = pd.DataFrame([x.split(",") for x in kl], columns=FIELDS)
            df["date"] = pd.to_datetime(df["date"])
            df["adj_close"] = pd.to_numeric(df["close"], errors="coerce")
            d = df[["date", "adj_close"]].sort_values("date").reset_index(drop=True)
            tmp = out + ".tmp"
            d.to_parquet(tmp, index=False)
            os.replace(tmp, out)
            return "ok"
        except Exception:
            if k == retries - 1:
                return "err"
            time.sleep(1.0 + random.random())
    return "err"


def main(workers=24):
    codes = [f[:-8] for f in sorted(os.listdir(D_DAILY)) if f.endswith(".parquet")]
    todo = [c for c in codes if not os.path.exists(os.path.join(D_ADJ, f"{c}.parquet"))]
    print(f"[adj] total={len(codes)} todo={len(todo)}", flush=True)
    stats, t0 = {}, time.time()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(fetch_one, c): c for c in todo}
        for i, fu in enumerate(as_completed(futs), 1):
            r = fu.result()
            stats[r] = stats.get(r, 0) + 1
            if i % 1000 == 0 or i == len(todo):
                print(f"  {i}/{len(todo)} {stats} {i/(time.time()-t0):.1f}/s", flush=True)
    print("[adj] done", stats)


if __name__ == "__main__":
    main()
