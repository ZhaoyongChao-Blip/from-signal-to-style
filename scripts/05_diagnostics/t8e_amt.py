# -*- coding: utf-8 -*-
"""t8e_amt.py —— V3b 是唯一「追平等权」的变体，必须查清它的收益是不是又一次「容量幻觉」

V3b = y_neut 分数 + 只买成交额 ≥ 3 亿的股票 → 年化 21.17%，Sharpe 0.955，相对等权 +0.05%。
看起来「找到了可以落地的版本」。但必须排除三件事：
  A. 是不是「大票 beta」？→ 用成交额分层看各层未来 5 日收益，看 3 亿以上那层本身是否就跑赢
  B. 是不是「涨停股」？→ 看 picks 里未来 5 日含涨停的比例
  C. 是不是「样本太小」？→ 每天满足 amount≥3亿 的股票数

产出：output/m8e_amt.md
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


def main():
    panel = pd.read_parquet(os.path.join(ROOT, "data", "proc", "panel.parquet"),
                            columns=["date", "code", "board", "amount", "fwd5", "excess5",
                                     "fwd5_has_zt", "has_fwd5", "log_amt", "raw_close"])
    panel["date"] = pd.to_datetime(panel["date"])
    panel = panel[panel["has_fwd5"] == 1]
    d = panel[(panel["date"] >= "2024-01-01") & (panel["date"] <= "2025-11-10")]

    P("# V3b「成交额 ≥ 3 亿」的收益来源排查\n")
    P(f"- 区间 2024-01-02 ~ {d['date'].max().date()}，样本 {len(d):,} 行\n")

    # A 成交额分层本身的收益
    P("## A. 按成交额分层的未来 5 日收益（不选股，全样本）\n")
    P("| 成交额档 | n | 中位成交额 | 未来5日均值 | 中位数 | 含涨停率 |")
    P("|---|---|---|---|---|---|")
    for lo, hi, nm in [(2e7, 1e8, "2千万~1亿"), (1e8, 3e8, "1亿~3亿"), (3e8, 1e11, "3亿以上")]:
        s = d[(d["amount"] >= lo) & (d["amount"] < hi)]
        if len(s) == 0:
            continue
        P(f"| {nm} | {len(s):,} | {s['amount'].median()/1e8:.2f}亿 | **{s['fwd5'].mean():.3%}** | "
          f"{s['fwd5'].median():.3%} | {s['fwd5_has_zt'].mean():.1%} |")
    P("\n> 若「3 亿以上」本身就跑赢，那 V3b 的收益可能只是「买大票」，不是选股。\n")

    # 每日可选池大小
    P("## C. 每日满足 amount ≥ 3 亿 的股票数\n")
    cnt = d[d["amount"] >= 3e8].groupby("date").size()
    cnt_all = d.groupby("date").size()
    P(f"- 中位 {cnt.median():.0f} 只 / 全市场 {cnt_all.median():.0f} 只"
      f"（占比 {cnt.median()/cnt_all.median():.1%}）")
    P(f"- 最少 {cnt.min()} 只，最多 {cnt.max()} 只")
    P(f"- 低于 30 只的交易日：{(cnt < 30).sum()} 天 / {len(cnt)} 天")
    P("\n> ⚠️ 若可选池经常少于 30 只，Top30 就等于「全买」，模型根本没有选择权。\n")

    # B 涨停
    P("## B. 分层里的涨停情况\n")
    for lo, hi, nm in [(2e7, 1e8, "2千万~1亿"), (1e8, 3e8, "1亿~3亿"), (3e8, 1e11, "3亿以上")]:
        s = d[(d["amount"] >= lo) & (d["amount"] < hi)]
        if len(s) == 0:
            continue
        z = s[s["fwd5_has_zt"] == 1]
        nz = s[s["fwd5_has_zt"] == 0]
        P(f"- {nm}: 含涨停 {z['fwd5'].mean():.3%}(n={len(z):,}) / 不含 {nz['fwd5'].mean():.3%}(n={len(nz):,})")

    with open(os.path.join(OUT, "m8e_amt.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    print(f"[saved] {os.path.join(OUT,'m8e_amt.md')}")


if __name__ == "__main__":
    main()
