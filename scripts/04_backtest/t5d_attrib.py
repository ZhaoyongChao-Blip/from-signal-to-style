# -*- coding: utf-8 -*-
"""t5d_slice.py —— 把回测拆成「等权市场 beta + 风格暴露」两块

问题：LGB 组合在 2024H1 亏 30~37%，而全市场等权亏 32.6%，指数只亏 1~5%。
→ 组合的风险来源不是「选股」，而是「持有一篮子小盘股 = 承担微盘 beta」。

做法（不改模型，只做回归归因）：
  R1 用沪深300 / 中证1000 / 微盘代理(全市场等权) 三因子，对策略日收益做 OLS，看 α 和 beta
  R2 同样回归「超额收益（策略 − 全市场等权）」，这才是选股能力的直接检验
  R3 按时间拆成 2024H1 / 2024H2-2025 分别回归，看 α 是否稳定

产出：output/m5d_attrib.md
"""
import os, sys
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
OUT = os.path.join(ROOT, "output")
IDX = os.path.join(ROOT, "data", "raw", "index")
import t5_backtest as T5

L = []


def P(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    L.append(s)


def bench_r(name, start, end):
    d = pd.read_parquet(os.path.join(IDX, f"{name}.parquet")).sort_values("date")
    d["date"] = pd.to_datetime(d["date"])
    s = d.set_index("date")["close"]
    s = s[(s.index >= start) & (s.index <= end)].ffill()
    return s.pct_change()


def main():
    panel = T5.load_panel()
    import lightgbm as lgb
    import t4_lgb
    fc = t4_lgb.feat_cols(panel)

    S, E = "2024-01-01", "2025-11-10"
    d = panel[(panel["date"] >= S) & (panel["date"] <= E)][["date", "code", "adj_close"]]
    d = d.sort_values(["code", "date"])
    d["r"] = d.groupby("code")["adj_close"].pct_change()
    ew = d.groupby("date")["r"].mean().fillna(0)
    ew.name = "EW"

    X = pd.DataFrame({"HS300": bench_r("000300", S, E), "CSI1000": bench_r("000852", S, E), "EW": ew})
    # 微盘代理：全市场等权 − 中证1000 的残差部分没有直接数据，用 EW 与 CSI1000 的线性组合即可覆盖

    navs = {}
    for lb in ("y_raw", "y_exc"):
        mp = os.path.join(ROOT, "models", f"lgb_{lb}.txt")
        if not os.path.exists(mp):
            continue
        bst = lgb.Booster(model_file=mp)
        panel[f"score_{lb}"] = bst.predict(panel[fc])
        nav, _ = T5.backtest(panel, f"score_{lb}", topk=30, hold=5, cost=0.0015, start=S, end=E)
        navs[f"LGB_{lb}"] = nav
    # 朴素对照：低波动
    panel["neg_vol_20"] = -panel["vol_20"]
    nav, _ = T5.backtest(panel, "neg_vol_20", topk=30, hold=5, cost=0.0015, start=S, end=E)
    navs["NAIVE_negvol"] = nav

    def ols(y, Xm):
        Xm = Xm.copy()
        Xm.insert(0, "const", 1.0)
        beta, *_ = np.linalg.lstsq(Xm.values, y.values, rcond=None)
        resid = y.values - Xm.values @ beta
        n, k = Xm.shape
        s2 = (resid ** 2).sum() / (n - k)
        cov = s2 * np.linalg.pinv(Xm.values.T @ Xm.values)
        se = np.sqrt(np.diag(cov))
        names = list(Xm.columns)
        out = {nm: (beta[i], se[i], beta[i] / se[i]) for i, nm in enumerate(names)}
        r2 = 1 - (resid ** 2).sum() / ((y.values - y.mean()) ** 2).sum()
        return out, r2, n

    for title, (a, b) in {"全期": (S, E), "2024H1": ("2024-01-01", "2024-06-30"),
                          "2024H2-2025": ("2024-07-01", E)}.items():
        P(f"\n## {title}（{a} ~ {b}）\n")
        P("| 策略 | 因变量 | 年化α | t(α) | β_HS300 | β_CSI1000 | β_EW | R² | n |")
        P("|---|---|---|---|---|---|---|---|---|")
        for nm, nav in navs.items():
            r = nav.pct_change().dropna()
            r = r[(r.index >= a) & (r.index <= b)]
            Xs = X.reindex(r.index).dropna()
            r = r.reindex(Xs.index)
            if len(r) < 30:
                continue
            res, r2, n = ols(r, Xs)
            alpha_ann = res["const"][0] * 252
            P(f"| {nm} | 原始收益 | {alpha_ann:.2%} | {res['const'][2]:.2f} | "
              f"{res['HS300'][0]:.2f} | {res['CSI1000'][0]:.2f} | {res['EW'][0]:.2f} | {r2:.3f} | {n} |")
            # R2: 对超额收益回归
            ex = r - Xs["EW"]
            res2, r2b, n2 = ols(ex, Xs[["HS300", "CSI1000"]])
            P(f"| {nm} | **超额 vs EW** | {res2['const'][0]*252:.2%} | {res2['const'][2]:.2f} | "
              f"{res2['HS300'][0]:.2f} | {res2['CSI1000'][0]:.2f} | — | {r2b:.3f} | {n2} |")

    P("\n> 读法：`原始收益` 的 α 若为正但 β_EW≈1，说明收益来自「持有小盘股」而不是选股；")
    P("> `超额 vs EW` 那一行的 α/t 才是选股能力。t<2 即不能拒绝「没有选股能力」。")

    with open(os.path.join(OUT, "m5d_attrib.md"), "w", encoding="utf-8") as f:
        f.write("# 收益归因（OLS）\n" + "\n".join(L))
    print(f"[saved] {os.path.join(OUT,'m5d_attrib.md')}")


if __name__ == "__main__":
    main()
