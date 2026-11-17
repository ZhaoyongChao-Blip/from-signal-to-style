# -*- coding: utf-8 -*-
"""t9_reduce_turnover.py —— 降低换手率：唯一可能把成本平衡点从 10bp 抬到 15bp 以上的方向

现状：3亿池 / 5日 / 15bp → 年化 21.17%、相对等权 −3.41%；0bp 时 32.76%（+5.82%）。
      alpha 存在（同风格格子内 +0.713%/5日，t=3.47），但年化换手 28.56 倍、成本 8.57%/年。

三个降换手手段（每轮调仓时决定买什么）：

  R1 **分数平滑**：用过去 S 日模型分数的均值排序（S=1 即原版）。平滑会让排名更稳 → 重合度更高。
  R2 **缓冲区（buffer）**：上期持仓中仍在 TopK×(1+b) 内的继续持有，只把掉出缓冲区的换掉。
  R3 **排名滞后带**：只替换「新进前 K 名」且「上期排名在 K×(1+b) 之外」的股票。

三者的共同点：**不改变选股方向，只减少无谓换手**。
如果 alpha 是真的且不集中在「最新进入 TopK 的那几只」，降换手应该能显著改善净收益。

口径与 t8l 完全一致（已验证）：
  dt 日筛池 → t+1 开盘买入（原始价判涨停）→ 复权价算收益 → 逐日收益累乘

产出：output/m9_turnover.md
"""
import os, sys
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
OUT = os.path.join(ROOT, "output")
L = []


def P(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    L.append(s)


def perf(nav):
    r = nav.pct_change().dropna()
    if len(r) < 20:
        return {}
    ann = (1 + r).prod() ** (252 / len(r)) - 1
    vol = r.std() * np.sqrt(252)
    return {"ann": ann, "vol": vol, "sharpe": (ann - 0.015) / vol,
            "mdd": (nav / nav.cummax() - 1).min()}


def main():
    import lightgbm as lgb
    import t4_lgb
    panel = pd.read_parquet(os.path.join(ROOT, "data", "proc", "panel.parquet"))
    panel["date"] = pd.to_datetime(panel["date"])
    fc = t4_lgb.feat_cols(panel)
    bt = panel[(panel["date"] >= "2024-01-01") & (panel["date"] <= "2025-11-10")].copy()
    bt = bt.dropna(subset=["adj_close", "adj_open", "raw_close", "raw_open", "amount"])
    bt["lim"] = np.where(bt["board"].isin(["GEM", "STAR"]), 0.20, 0.10)
    bt["s"] = lgb.Booster(model_file=os.path.join(ROOT, "models", "lgb_y_neut.txt")).predict(bt[fc])
    bt = bt.sort_values(["date", "code"]).reset_index(drop=True)

    # R1 需要的「分数平滑」：用过去 S 日的日频分数均值。
    # panel 是 5 日频率吗？不是——panel 是逐日，has_fwd5 只要求未来 5 日存在。
    # 所以日期是连续交易日，可以直接按 code 做 rolling。
    if "s" in bt.columns:
        bt = bt.sort_values(["code", "date"])
        for S in (3, 5):
            bt[f"s_ma{S}"] = (bt.groupby("code")["s"]
                              .transform(lambda x: x.rolling(S, min_periods=1).mean()))
        bt = bt.sort_values(["date", "code"]).reset_index(drop=True)

    dates = np.array(sorted(bt["date"].unique()))
    by = {d: g.set_index("code") for d, g in bt.groupby("date")}
    dd = bt[["date", "code", "adj_close"]].sort_values(["code", "date"])
    dd["r"] = dd.groupby("code")["adj_close"].pct_change()
    ew = (1 + dd.groupby("date")["r"].mean().fillna(0)).cumprod()
    ew_r = ew.pct_change()

    P("# 降换手实验\n")
    pe = perf(ew)
    P(f"- 基准（全市场等权）年化 {pe['ann']:.2%}，Sharpe {pe['sharpe']:.3f}\n")

    def run(col, min_amt=3e8, hold=5, cost=0.0015, K=30, buffer=1.0, keep_prev=False):
        """
        buffer > 1：上期持仓中排名仍在 TopK*buffer 内的继续持有（缓冲区）
        keep_prev=False 时完全按新排名选（原版）
        """
        cash = 1.0
        hold_px, weights = {}, {}
        out, to_list = [], []
        for i, dt in enumerate(dates):
            g = by[dt]
            if weights:
                s_ = 0.0
                for c, w in weights.items():
                    p0 = hold_px.get(c, np.nan)
                    p1 = g["adj_close"].get(c, np.nan)
                    if np.isfinite(p0) and np.isfinite(p1) and p0 > 0:
                        s_ += w * (p1 / p0 - 1)
                cash *= (1 + s_)
            out.append((dt, cash))
            for c in weights:
                p_ = g["adj_close"].get(c, np.nan)
                if np.isfinite(p_) and p_ > 0:
                    hold_px[c] = p_
            if i % hold != 0 or i == len(dates) - 1:
                continue
            nxt = dates[i + 1]
            g2 = by[nxt]
            cand = g.dropna(subset=[col, "adj_close"])
            cand = cand[cand["amount"] > min_amt]
            if len(cand) < K:
                continue
            ranked = cand[col].rank(ascending=False)
            rank_map = ranked.to_dict()

            if keep_prev and weights:
                # 缓冲区：上期持仓中排名仍在 K*buffer 内的保留
                keep = [c for c in weights if rank_map.get(c, 1e9) <= K * buffer]
                keep = keep[:K]
                n_new = K - len(keep)
                pool = cand.drop(index=[c for c in keep if c in cand.index], errors="ignore")
                new_add = pool.nlargest(n_new, col).index.tolist() if n_new > 0 else []
                picks = keep + new_add
            else:
                picks = cand.nlargest(K, col).index.tolist()

            buy = []
            for c in picks:
                if c not in g2.index:
                    continue
                ro, rcp, lm = g2.at[c, "raw_open"], g.at[c, "raw_close"], g2.at[c, "lim"]
                if not (np.isfinite(ro) and np.isfinite(rcp) and rcp > 0):
                    continue
                if ro >= rcp * (1 + lm - 0.005):
                    continue
                buy.append(c)
            if not buy:
                continue
            new = {c: 1.0 / len(buy) for c in buy}
            for c in buy:
                hold_px[c] = g2.at[c, "adj_open"]
            old = set(weights)
            to = 1.0 - len(old & set(new)) / max(len(new), 1) if old else 1.0
            to_list.append(to)
            cash *= (1 - cost * to * 2)
            weights = new
        return pd.Series(dict(out)).sort_index(), np.array(to_list)

    rows = []
    P("| 变体 | 信号 | K | 缓冲区 | 单边成本 | 年化 | Sharpe | 相对等权 | 单次换手 | 年化换手 |")
    P("|---|---|---|---|---|---|---|---|---|---|")
    tests = [
        ("原版", "s", 30, 1.0),
        ("R1 平滑3日", "s_ma3", 30, 1.0),
        ("R1 平滑5日", "s_ma5", 30, 1.0),
        ("R2 缓冲1.3x", "s", 30, 1.3),
        ("R2 缓冲1.5x", "s", 30, 1.5),
        ("R2 缓冲2.0x", "s", 30, 2.0),
        ("R1+R2 平滑5日+缓冲1.5x", "s_ma5", 30, 1.5),
    ]
    for nm, col, K, buf in tests:
        if col not in bt.columns:
            continue
        for cost in (0.0015,):
            nv, to = run(col, K=K, cost=cost, buffer=buf, keep_prev=(buf > 1.0))
            pf = perf(nv)
            if not pf:
                continue
            rr = nv.pct_change().fillna(0)
            rb = ew_r.reindex(nv.index).fillna(0)
            exc = (1 + (rr - rb)).prod() ** (252 / len(rr)) - 1
            t = np.median(to) if len(to) else np.nan
            ann_to = t * 50.4
            P(f"| {nm} | {col} | {K} | {buf:.1f} | {cost*1e4:.0f}bp | {pf['ann']:.2%} | "
              f"{pf['sharpe']:.3f} | {exc:+.2%} | {t:.1%} | {ann_to:.1f}倍 |")
            nv.to_frame("nav").to_csv(os.path.join(OUT, f"m9_nav_{nm.split()[0]}_{buf}.csv"))

    with open(os.path.join(OUT, "m9_turnover.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    print(f"[saved] {os.path.join(OUT,'m9_turnover.md')}")


if __name__ == "__main__":
    main()
