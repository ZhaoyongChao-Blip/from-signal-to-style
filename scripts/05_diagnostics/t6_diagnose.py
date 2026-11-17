# -*- coding: utf-8 -*-
"""
t6_diagnose.py —— 诊断：这些坑不查，回测数字就没意义

依次检查（每一项都输出数字，不是论述）:
  D1 链路自检：从原始 parquet 独立重算 fwd5，与 panel 里比对，要求 max|diff| < 1e-10
  D2 因子是否含未来信息：把因子与「未来收益」做单变量 IC，若某个基础因子 IC 异常高（>0.15）就报警
  D3 标签是否被涨停绑架：分别统计 y_raw / y_exc 的正样本里，未来5日含涨停的比例
  D4 模型是否真的在学「超额」：按模型分数分 10 组，看各组 fwd5 与 excess5 的均值
  D5 预测分数的分布漂移：训练期 vs 测试期的分数均值/标准差
  D6 逐年 AUC：如果只有某一年有效，就是过拟合

产出: output/m6_diagnose.md
"""
import os, sys
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
PROC = os.path.join(ROOT, "data", "proc")
RAW = os.path.join(ROOT, "data", "raw")
OUT = os.path.join(ROOT, "output")
os.makedirs(OUT, exist_ok=True)

L = []
def P(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    L.append(s)


def d1_chain_check(panel):
    P("## D1 链路自检（重算 fwd5）\n")
    # 抽 50 只，用与 t3 **完全相同的口径**重算 fwd5：
    #   ① 复权价 adj_close（ffill） ② 剔除异常报价行 ③ 对齐到全市场日历后 shift(-5)
    cal = pd.to_datetime(pd.read_parquet(os.path.join(RAW, "index", "000300.parquet"))["date"])
    cal_i = pd.DatetimeIndex(cal)
    codes = panel["code"].drop_duplicates().sample(50, random_state=1).tolist()
    maxdiff = 0.0
    n = 0
    bad = []
    for c in codes:
        p = os.path.join(RAW, "daily", f"{c}.parquet")
        if not os.path.exists(p):
            continue
        raw = pd.read_parquet(p).sort_values("date").reset_index(drop=True)
        raw["date"] = pd.to_datetime(raw["date"])
        ap = os.path.join(RAW, "adj", f"{c}.parquet")
        if not os.path.exists(ap):
            continue
        adj = pd.read_parquet(ap)
        adj["date"] = pd.to_datetime(adj["date"])
        d = raw.merge(adj, on="date", how="left").sort_values("date").reset_index(drop=True)
        d["adj_close"] = d["adj_close"].ffill()

        # 复刻 t3 的异常行剔除
        board_lim = 0.20 if c[:3] in ("300", "301") or c[:2] == "68" else 0.10
        chg = d["adj_close"].pct_change()
        badrow = (chg.abs() > board_lim + 0.02) & (d.index > 0)
        if badrow.sum() > 0:
            d = d.loc[~badrow].reset_index(drop=True)

        # 对齐日历后 shift(-5)
        c_al = pd.Series(d["adj_close"].astype(float).values, index=pd.DatetimeIndex(d["date"])).reindex(cal_i)
        f5 = (c_al.shift(-5) / c_al - 1)
        mine = pd.DataFrame({"date": pd.DatetimeIndex(d["date"]), "fwd5_calc": f5.reindex(pd.DatetimeIndex(d["date"])).values})

        sub = panel[panel["code"] == c][["date", "fwd5"]].copy()
        sub["date"] = pd.to_datetime(sub["date"])
        m = mine.merge(sub, on="date", how="inner")
        if len(m) == 0:
            continue
        diff = (m["fwd5"] - m["fwd5_calc"]).abs()
        dd = diff.max()
        maxdiff = max(maxdiff, dd)
        n += len(m)
        if dd > 1e-6:
            bad.append((c, float(dd)))
    P(f"- 抽样股票: {len(codes)}, 比对行数: {n:,}")
    P(f"- **max|diff| = {maxdiff:.3e}**  {'✅ 通过' if maxdiff < 1e-6 else '❌ 未通过'}")
    if bad:
        P(f"- 异常: {bad[:5]}")
    return maxdiff < 1e-6


def d2_factor_ic(panel):
    P("\n## D2 因子单变量 IC（对 fwd5）\n")
    P("> 若某个「应当很弱的」因子 IC > 0.15，说明它偷偷用了未来信息。\n")
    import t4_lgb
    from d2_fast import factor_daily_ic
    fc = t4_lgb.feat_cols(panel)
    # 向量化实现（原 groupby.apply + corr(spearman) 要跑数小时，结果已对拍一致 2.8e-17）
    R = factor_daily_ic(panel, fc).sort_values("IC", key=abs, ascending=False)
    R.to_csv(os.path.join(OUT, "m6_factor_ic.csv"), index=False)
    P(R.head(15).round(4).to_string(index=False))
    flag = R[abs(R["IC"]) > 0.15]
    if len(flag):
        P(f"\n⚠️ **{len(flag)} 个因子 |IC| > 0.15**: {flag['factor'].tolist()}")
    else:
        P("\n✅ 没有因子 |IC| > 0.15")


def d3_label_zt(panel):
    P("\n## D3 标签是否被涨停绑架\n")
    d = panel.dropna(subset=["fwd5", "mkt_fwd5"])
    P(f"- 全样本 y_raw 正样本率: {(d['fwd5']>0).mean():.2%}")
    P(f"- 全样本 y_exc 正样本率: {(d['fwd5']>d['mkt_fwd5']).mean():.2%}")
    a = d[d["fwd5"] > 0]
    b = d[d["fwd5"] > d["mkt_fwd5"]]
    P(f"- y_raw=1 中「未来5日含涨停」占比: **{a['fwd5_has_zt'].mean():.2%}**")
    P(f"- y_exc=1 中「未来5日含涨停」占比: **{b['fwd5_has_zt'].mean():.2%}**")
    P(f"- 全样本中「未来5日含涨停」占比: {d['fwd5_has_zt'].mean():.2%}")
    # 分组收益
    P("\n### 按是否含未来涨停分组的 5 日收益\n")
    for k, g in d.groupby("fwd5_has_zt", observed=True):
        nm = "含涨停" if k == 1 else "不含涨停"
        P(f"- {nm}: n={len(g):,} 均值 {g['fwd5'].mean():.2%} 中位数 {g['fwd5'].median():.2%} 胜率 {(g['fwd5']>0).mean():.1%}")


def d4_decile(panel, label="y_exc"):
    P(f"\n## D4 按模型分数分 10 组（模型 {label}）\n")
    import lightgbm as lgb
    import t4_lgb
    mp = os.path.join(ROOT, "models", f"lgb_{label}.txt")
    if not os.path.exists(mp):
        P("- 模型文件不存在，跳过")
        return
    bst = lgb.Booster(model_file=mp)
    fc = t4_lgb.feat_cols(panel)
    d = panel.dropna(subset=["fwd5", "mkt_fwd5"]).copy()
    d = d[(d["date"] >= "2024-01-01")]
    d["pred"] = bst.predict(d[fc])
    d["q"] = d.groupby("date")["pred"].transform(lambda s: pd.qcut(s.rank(method="first"), 10, labels=False))
    g = d.groupby("q").agg(n=("fwd5", "size"), fwd5_mean=("fwd5", "mean"),
                           fwd5_med=("fwd5", "median"), exc_mean=("excess5", "mean"),
                           win=("fwd5", lambda s: (s > 0).mean()),
                           zt_rate=("fwd5_has_zt", "mean"))
    g.index = [f"Q{i+1}" for i in g.index]
    P(g.round(4).to_string())
    P("\n> 若 Q10 的 exc_mean 没有明显高于 Q1，说明模型**在超额收益上无效**，只在绝对收益上有效 = 只是「选涨得多的」。")
    spread = g["exc_mean"].iloc[-1] - g["exc_mean"].iloc[0]
    P(f"\n- Q10 − Q1 超额差: **{spread:.2%}**")


def d5_drift(panel):
    P("\n## D5 分数漂移\n")
    import lightgbm as lgb
    import t4_lgb
    for lb in ("y_raw", "y_exc"):
        mp = os.path.join(ROOT, "models", f"lgb_{lb}.txt")
        if not os.path.exists(mp):
            continue
        bst = lgb.Booster(model_file=mp)
        fc = t4_lgb.feat_cols(panel)
        d = panel.dropna(subset=fc[:3])
        d = d.assign(pred=bst.predict(d[fc]))
        for nm, (a, b) in {"train18-22": ("2018-01-01", "2022-12-31"),
                           "valid23": ("2023-01-01", "2023-12-31"),
                           "test24-25": ("2024-01-01", "2026-12-31")}.items():
            s = d[(d["date"] >= a) & (d["date"] <= b)]["pred"]
            P(f"- {lb} {nm}: mean={s.mean():.4f} std={s.std():.4f} q50={s.quantile(.5):.4f} q90={s.quantile(.9):.4f}")


def d6_yearly_auc(panel):
    P("\n## D6 逐年样本外 AUC（train 用之前 5 年）\n")
    import lightgbm as lgb
    import t4_lgb
    from sklearn.metrics import roc_auc_score
    fc = t4_lgb.feat_cols(panel)
    lb = "y_exc"
    d = panel.dropna(subset=[lb])
    P("| 年份 | 训练集 | n_test | AUC | 正样本率 |")
    P("|---|---|---|---|---|")
    for y in range(2019, 2026):
        tr = d[(d["date"] >= f"{y-5}-01-01") & (d["date"] <= f"{y-2}-12-31")]
        va = d[(d["date"] >= f"{y-1}-01-01") & (d["date"] <= f"{y-1}-12-31")]
        te = d[(d["date"] >= f"{y}-01-01") & (d["date"] <= f"{y}-12-31")]
        if len(te) < 1000 or len(tr) < 50000:
            continue
        p = dict(objective="binary", metric="auc", learning_rate=0.05, num_leaves=31,
                 min_data_in_leaf=500, feature_fraction=0.6, bagging_fraction=0.7, bagging_freq=5,
                 lambda_l2=5.0, num_threads=4, verbosity=-1, seed=42)
        bst = lgb.train(p, lgb.Dataset(tr[fc], tr[lb]), num_boost_round=400,
                        valid_sets=[lgb.Dataset(va[fc], va[lb])],
                        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)])
        auc = roc_auc_score(te[lb], bst.predict(te[fc], num_iteration=bst.best_iteration))
        P(f"| {y} | {y-5}~{y-2} | {len(te):,} | **{auc:.4f}** | {te[lb].mean():.3f} |")


def main():
    panel = pd.read_parquet(os.path.join(PROC, "panel.parquet"))
    panel["date"] = pd.to_datetime(panel["date"])
    # 与 t4_lgb.load() 同口径的标签（panel 里不存标签，避免误入特征）
    panel = panel[panel["has_fwd5"] == 1].copy()
    panel["y_raw"] = (panel["fwd5"] > 0).astype(int)
    panel["y_exc"] = (panel["fwd5"] > panel["mkt_fwd5"]).astype(int)
    P(f"# 诊断报告\n\n- panel: {panel.shape}\n- 日期: {panel['date'].min().date()} ~ {panel['date'].max().date()}\n")
    d1_chain_check(panel)
    d2_factor_ic(panel)
    d3_label_zt(panel)
    d4_decile(panel, "y_exc")
    d5_drift(panel)
    d6_yearly_auc(panel)
    with open(os.path.join(OUT, "m6_diagnose.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    print(f"\n[saved] {os.path.join(OUT,'m6_diagnose.md')}")


if __name__ == "__main__":
    main()
