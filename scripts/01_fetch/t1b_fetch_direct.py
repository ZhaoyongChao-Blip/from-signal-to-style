# -*- coding: utf-8 -*-
"""
t1b_fetch_direct.py —— 东财 push2his 直连抓取（比 akshare 封装快 5-10 倍）

接口:
  https://push2his.eastmoney.com/api/qt/stock/kline/get
    secid = 1.600519 (沪) / 0.000001 (深) / 0.920xxx,0.83xxxx (北交所也走 0.)
    klt=101(日)  fqt=1(前复权)

用法:
  python t1b_fetch_direct.py --workers 24
  python t1b_fetch_direct.py --check
"""
import os, sys, time, argparse, json, random
import pandas as pd
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

ROOT = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(ROOT, "data", "raw")
D_DAILY = os.path.join(RAW, "daily")
os.makedirs(D_DAILY, exist_ok=True)

URL = "http://push2his.eastmoney.com/api/qt/stock/kline/get"
PARAMS = {
    "fields1": "f1,f2,f3,f4,f5,f6",
    "fields2": "f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61",
    "klt": "101", "fqt": "0", "beg": "20140101", "end": "20991231", "lmt": "100000",
}
FIELDS = ["date", "open", "close", "high", "low", "volume", "amount", "amplitude", "pct_chg", "chg", "turnover"]

# ⚠️ 2026-11-11 踩坑：起初用 fqt=1（前复权），东财在「前复权」模式下，凡是**分红除权日**
#    会把除权日前后的价格序列拼接得极不稳定（负价、单日 +/-2600%），万科的 2014-2016 全烂。
#    → 改为 fqt=0（不复权）落盘，复权统一在本地按 adj_factor 算（见 t3_features.py）。
#    验证：fqt=0 的 close 与 fqt=1 在同日期可完全对上（000002 max|diff|=0），说明差异只由复权方式引起。

SESS = requests.Session()
SESS.headers.update({
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120.0 Safari/537.36",
    "Referer": "https://quote.eastmoney.com/",
})


def secid(code):
    return ("1." if code[:3] in ("600", "601", "603", "605", "688", "689") else "0.") + code


def fetch_one(code, retries=3):
    out = os.path.join(D_DAILY, f"{code}.parquet")
    if os.path.exists(out) and os.path.getsize(out) > 200:
        return "skip"
    p = dict(PARAMS); p["secid"] = secid(code)
    for k in range(retries):
        try:
            r = SESS.get(URL, params=p, timeout=12)
            j = r.json()
            kl = (j.get("data") or {}).get("klines") or []
            if not kl:
                return "empty"
            rows = [x.split(",") for x in kl]
            df = pd.DataFrame(rows, columns=FIELDS)
            df["date"] = pd.to_datetime(df["date"])
            for c in FIELDS[1:]:
                df[c] = pd.to_numeric(df[c], errors="coerce")
            df["code"] = code
            df = df[["date", "code", "open", "high", "low", "close", "volume", "amount",
                     "amplitude", "pct_chg", "turnover"]].sort_values("date")
            tmp = out + ".tmp"
            df.to_parquet(tmp, index=False)
            os.replace(tmp, out)
            return "ok"
        except Exception as e:
            if k == retries - 1:
                return f"err:{type(e).__name__}"
            time.sleep(1.2 + random.random())
    return "err"


def main(workers=24):
    lst = pd.read_csv(os.path.join(RAW, "list.csv"), dtype={"code": str})
    codes = lst["code"].astype(str).str.zfill(6).tolist()
    todo = [c for c in codes if not (os.path.exists(os.path.join(D_DAILY, f"{c}.parquet"))
                                     and os.path.getsize(os.path.join(D_DAILY, f"{c}.parquet")) > 200)]
    print(f"[direct] total={len(codes)} todo={len(todo)} workers={workers}", flush=True)
    stats, t0 = {}, time.time()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(fetch_one, c): c for c in todo}
        for i, fu in enumerate(as_completed(futs), 1):
            rr = fu.result()
            k = rr.split(":")[0]
            stats[k] = stats.get(k, 0) + 1
            if i % 500 == 0 or i == len(todo):
                el = time.time() - t0
                print(f"  {i}/{len(todo)} {stats} {i/el:.1f}/s ETA {(len(todo)-i)/(i/el)/60:.1f}min", flush=True)
    print("[direct] done", stats)


def check():
    lst = pd.read_csv(os.path.join(RAW, "list.csv"), dtype={"code": str})
    files = [f for f in os.listdir(D_DAILY) if f.endswith(".parquet")]
    ok = sum(1 for f in files if os.path.getsize(os.path.join(D_DAILY, f)) > 200)
    print(f"list={len(lst)} files={len(files)} ok={ok} coverage={ok/len(lst):.1%}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    if a.check:
        check()
    else:
        main(a.workers)
