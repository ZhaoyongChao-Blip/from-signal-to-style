# -*- coding: utf-8 -*-
"""
t5_backtest.py —— 可执行回测（这是与韩磊论文最大的分歧点）

规则（每一条都写死，便于审阅）:
  · 信号：t 日收盘后得到 score；t+1 日**开盘价**买入（不是收盘价）。
  · 持仓：等权持有 Top-K，每 N 个交易日调仓一次（默认 N=5，与标签周期一致）。
  · 涨停不可买：t+1 开盘价 ≥ 前收 × (1+limit-0.5%) 视为买不进，跳过，用次优补齐。
  · 跌停不可卖：t+1 开盘价 ≤ 前收 × (1-limit+0.5%) 视为卖不出，顺延。
  · 成本：单边 c（默认 0.15% = 佣金 0.025% + 印花税 0.1%（卖出）+ 冲击 0.03%×2），
          按「换手率 × c」计，即只对实际换掉的仓位收费。
  · 基准：中证1000 / 中证500 / 沪深300 / 全市场等权。
  · 安慰剂：同一批股票、同样的调仓频率、随机打分，跑 200 次，得到 Sharpe 分布。

指标：年化收益、年化波动、Sharpe(rf=1.5%)、最大回撤、胜率、换手率、超额 vs 基准。

产出: output/m5_bt_<tag>.csv, output/m5_summary.md, output/m5_placebo.npy
"""
import os, json, argparse
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
PROC = os.path.join(ROOT, "data", "proc")
IDX = os.path.join(ROOT, "data", "raw", "index")
OUT = os.path.join(ROOT, "output")
os.makedirs(OUT, exist_ok=True)

RF = 0.015  # 年化无风险利率，2024 年 1 年期国债约 1.1%，10 年期约 2.0%，取 1.5%（不是论文的 3.0%）


def limit_of(board):
    return np.where(np.isin(board, ["GEM", "STAR"]), 0.20, 0.10)


def load_panel():
    df = pd.read_parquet(os.path.join(PROC, "panel.parquet"))
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["date", "code"]).reset_index(drop=True)
    df["lim"] = limit_of(df["board"].values)
    return df


def perf(nav: pd.Series, bench: pd.Series = None, freq=252):
    r = nav.pct_change().dropna()
    if len(r) < 20:
        return {}
    ann = (1 + r).prod() ** (freq / len(r)) - 1 if len(r) > 0 else np.nan
    vol = r.std() * np.sqrt(freq)
    shp = (ann - RF) / vol if vol > 0 else np.nan
    dd = (nav / nav.cummax() - 1).min()
    out = {"ann_ret": ann, "ann_vol": vol, "sharpe": shp, "mdd": dd,
           "win_rate": (r > 0).mean(), "n_days": len(r)}
    if bench is not None:
        br = bench.pct_change().reindex(r.index).fillna(0)
        out["bench_ann"] = (1 + br).prod() ** (freq / len(r)) - 1
        out["excess_ann"] = ann - out["bench_ann"]
        out["ir"] = (r - br).mean() / (r - br).std() * np.sqrt(freq) if (r - br).std() > 0 else np.nan
    return out


def backtest(panel, score_col, topk=30, hold=5, cost=0.0015, start="2024-01-01", end="2026-12-31",
             filter_col=None, seed=None):
    """
    返回 daily nav Series 与交易记录。
    panel 需含: date, code, board, open, close, lim, <score_col>
    """
    d = panel[(panel["date"] >= start) & (panel["date"] <= end)].copy()
    start = pd.Timestamp(start)
    end = pd.Timestamp(end)
    dates = np.array(sorted(d["date"].unique()))

    #  预处理成 {date: DataFrame}
    by_date = {dt: g.set_index("code") for dt, g in d.groupby("date")}
    # 前一日收盘价（用于判断涨停/跌停）
    prev_close = {}
    for dt in dates:
        prev_close[dt] = by_date[dt]["adj_close"]
    close_map = {dt: by_date[dt]["adj_close"] for dt in dates}
    open_map = {dt: by_date[dt]["adj_open"] for dt in dates}
    lim_map = {dt: by_date[dt]["lim"] for dt in dates}

    rng = np.random.default_rng(seed)
    cash = 1.0
    # code -> 上一次估值的价格（建仓当日用开盘价，之后用前一日收盘价）
    hold_price = {}
    weights = {}
    nav_list, rebal_dates, turnover_list = [], [], []

    for i, dt in enumerate(dates):
        # ---- 估值：用当日收盘价 / 持仓上次估值价 ----
        if weights:
            px_now = close_map[dt]
            rets = {}
            for c, w in weights.items():
                p0 = hold_price.get(c, np.nan)
                p1 = px_now.get(c, np.nan)
                # 停牌（无价格）视为 0 收益，不填 1.0 以外的值
                rets[c] = (p1 / p0 - 1.0) if (p0 and np.isfinite(p0) and p0 > 0 and
                                             np.isfinite(p1)) else 0.0
            port_ret = float(sum(w * rets[c] for c, w in weights.items()))
        else:
            port_ret = 0.0
        cash *= (1 + port_ret)
        nav_list.append((dt, cash))
        # 估值后把持仓基准价更新为当日收盘价
        if weights:
            px_now = close_map[dt]
            for c in weights:
                p = px_now.get(c, np.nan)
                if np.isfinite(p) and p > 0:
                    hold_price[c] = p

        # ---- 是否调仓 ----
        if (i % hold) != 0 or i == len(dates) - 1:
            continue

        nxt = dates[i + 1]

        sc = by_date[dt].copy()
        if filter_col is not None:
            sc = sc[sc[filter_col] == 0] if filter_col in sc.columns else sc
        if seed is not None:
            sc["_s"] = rng.random(len(sc))
            score_col_use = "_s"
        else:
            score_col_use = score_col

        # 剔除：ST 无法判断（数据里没有），但可剔除「当日停牌」（无收盘价）
        cand = sc.dropna(subset=[score_col_use, "adj_close"])
        # 只在流动性够的池子里选：成交额 20 日均值（已在因子里的 log_amt 是标准化的，用原始 amount 更好）
        cand = cand[cand["amount"] > 2e7] if "amount" in cand.columns else cand
        if len(cand) < topk:
            continue
        picks = cand.nlargest(topk, score_col_use).index.tolist()

        # ---- 次日开盘建仓，剔除买不进的 ----
        op = open_map[nxt]
        pc = prev_close[dt]
        lm = lim_map[nxt]
        buyable = []
        for c in picks:
            if c not in op.index or pd.isna(op.get(c)):
                continue
            if pc.get(c, np.nan) > 0 and op[c] >= pc[c] * (1 + lm.get(c, 0.1) - 0.005):
                continue  # 涨停开盘，买不进
            buyable.append(c)
        if not buyable:
            continue
        new_w = {c: 1.0 / len(buyable) for c in buyable}

        # 建仓基准价 = 次日开盘价（次日估值时从开盘价算起）
        for c in buyable:
            p = open_map[nxt].get(c, np.nan)
            if np.isfinite(p) and p > 0:
                hold_price[c] = p

        # ---- 换手与成本 ----
        old = set(weights)
        new = set(new_w)
        turnover = 1.0 - len(old & new) / max(len(new), 1) if old else 1.0
        cash *= (1 - cost * turnover * 2)   # 双边近似：买卖各一次
        weights = new_w
        turnover_list.append((dt, turnover))

    # 已被上面的 for 循环完整收录（nav_list 每天都有）
    nav = pd.Series(dict(nav_list)).sort_index()
    return nav, pd.DataFrame(turnover_list, columns=["date", "turnover"])


def load_bench(name, start, end):
    p = os.path.join(IDX, f"{name}.parquet")
    if not os.path.exists(p):
        return None
    d = pd.read_parquet(p).sort_values("date")
    d["date"] = pd.to_datetime(d["date"])
    s = d.set_index("date")["close"]
    s = s[(s.index >= start) & (s.index <= end)]
    return s / s.iloc[0]


def run_all(panel, scores: dict, topk=30, hold=5, cost=0.0015,
            start="2024-01-01", end="2026-12-31"):
    lines, rows, navs = [], [], {}
    bm = {nm: load_bench(nm, start, end) for nm in ("000300", "000905", "000852")}

    # 全市场等权基准：逐日横截面等权持有全部股票，按 adj_close 计算日收益
    eq = panel[(panel["date"] >= start) & (panel["date"] <= end)][["date", "code", "adj_close"]]
    eq = eq.sort_values(["code", "date"])
    eq["r"] = eq.groupby("code")["adj_close"].pct_change()
    ew_r = eq.groupby("date")["r"].mean().fillna(0)
    bm["EW_ALL"] = (1 + ew_r).cumprod()

    for name, col in scores.items():
        nav, to = backtest(panel, col, topk=topk, hold=hold, cost=cost, start=start, end=end)
        if len(nav) < 20:
            continue
        navs[name] = nav
        for bnm, bs in bm.items():
            if bs is None or len(bs) == 0:
                continue
            r = perf(nav, bs.reindex(nav.index).ffill())
            r.update({"model": name, "bench": bnm, "topk": topk, "hold": hold, "cost": cost})
            rows.append(r)
        nav.to_frame("nav").to_csv(os.path.join(OUT, f"m5_nav_{name}.csv"))

    R = pd.DataFrame(rows)
    if len(R):
        R.to_csv(os.path.join(OUT, f"m5_bt_k{topk}_h{hold}.csv"), index=False)
        piv = R.pivot_table(index="model", columns="bench",
                            values=["ann_ret", "sharpe", "excess_ann", "ir", "mdd"])
        lines.append(f"\n## Top{topk} / 持有{hold}日 / 成本{cost*100:.2f}%/边\n")
        lines.append(piv.round(4).to_string())
        print(piv.round(4).to_string())
    return R, navs, bm


def placebo(panel, topk=30, hold=5, cost=0.0015, start="2024-01-01", end="2026-12-31", n=200):
    """随机打分，得到 Sharpe 的零分布。"""
    res = []
    for k in range(n):
        nav, _ = backtest(panel, "fwd5", topk=topk, hold=hold, cost=cost,
                          start=start, end=end, seed=1000 + k)
        p = perf(nav)
        if p:
            res.append({"seed": 1000 + k, **p})
        if (k + 1) % 50 == 0:
            print(f"  placebo {k+1}/{n}", flush=True)
    df = pd.DataFrame(res)
    df.to_csv(os.path.join(OUT, "m5_placebo.csv"), index=False)
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--topk", type=int, nargs="+", default=[30])
    ap.add_argument("--hold", type=int, nargs="+", default=[5])
    ap.add_argument("--cost", type=float, default=0.0015)
    ap.add_argument("--placebo", action="store_true")
    ap.add_argument("--start", default="2024-01-01")
    ap.add_argument("--end", default="2026-12-31")
    a = ap.parse_args()

    panel = load_panel()
    print(f"[bt] panel {panel.shape}")
    # 打分：为了让回测脚本独立可跑，这里直接用 t4 保存的模型重打分
    import lightgbm as lgb
    MOD = os.path.join(ROOT, "models")
    scores, drop = {}, None
    import t4_lgb
    df4 = t4_lgb.load()
    fc = t4_lgb.feat_cols(df4)
    for lb in ("y_raw", "y_exc"):
        mp = os.path.join(MOD, f"lgb_{lb}.txt")
        if not os.path.exists(mp):
            continue
        bst = lgb.Booster(model_file=mp)
        pr = bst.predict(panel[fc])
        panel[f"score_{lb}"] = pr
        scores[f"LGB_{lb}"] = f"score_{lb}"

    all_rows = []
    for tk in a.topk:
        for h in a.hold:
            R, navs, bm = run_all(panel, scores, topk=tk, hold=h, cost=a.cost, start=a.start, end=a.end)
            if len(R):
                all_rows.append(R)

    if all_rows:
        R = pd.concat(all_rows, ignore_index=True)
        R.to_csv(os.path.join(OUT, "m5_all.csv"), index=False)

        # 与随机基准比较
        lines = ["# 回测结果\n", f"- 区间: {a.start} ~ {a.end}",
                 f"- 成本: 单边 {a.cost*100:.2f}%", f"- rf: {RF*100:.1f}%\n"]
        for (tk, h), g in R.groupby(["topk", "hold"]):
            lines.append(f"\n## Top{tk} 持有{h}日\n")
            piv = g.pivot_table(index="model", columns="bench", values=["ann_ret", "sharpe", "excess_ann"])
            lines.append(piv.round(4).to_string())
        with open(os.path.join(OUT, "m5_summary.md"), "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        print(f"[saved] {os.path.join(OUT,'m5_summary.md')}")

    if a.placebo:
        print("\n[placebo] 开始 200 次随机打分...")
        df = placebo(panel, topk=a.topk[0], hold=a.hold[0], cost=a.cost, start=a.start, end=a.end)
        print(df["sharpe"].describe().round(4).to_string())


if __name__ == "__main__":
    main()
