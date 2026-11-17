# -*- coding: utf-8 -*-
"""fix43.py —— 修正 4.3 节「两行拼一句」的错误

问题：论文写 beta 1.8 / R2 0.75-0.81（来自 m5d_attrib.md 的「原始收益」行）
      + t = 0.03/-0.63（来自「超额 vs EW」行）。两行不能拼在一句话里。
"""
import os

ROOT = os.path.dirname(os.path.abspath(__file__))
MD = os.path.join(ROOT, "paper", "paper.md")

old = ("Regressing the strategy's daily return on three factors, the equal-weighted market, "
       "the CSI 300 and the CSI 1000, yields a loading of 1.8 on the equal-weighted market and "
       "-0.1 on the CSI 300, with R-squared between 0.75 and 0.81. The intercept, tested against "
       "Newey-West standard errors, has t-statistics of 0.03 and -0.63 for the two label "
       "constructions. Whatever the model earned is explained by market and style exposure.")

new = ("Two regressions are needed here, and they answer different questions. Regressing the "
       "strategy's daily return on the equal-weighted market, the CSI 300 and the CSI 1000 "
       "explains 75 to 81% of its variance, with a loading of 1.84 on the equal-weighted market "
       "and -0.11 on the CSI 300: the book is, mechanically, a levered small-cap portfolio. That "
       "regression's own intercept is negative, with t-statistics of -0.72 and -1.39 for the two "
       "label constructions. Regressing instead on the *excess* return over the equal-weighted "
       "market removes that exposure by construction, so the intercept there measures selection "
       "alone: t-statistics of 0.03 and -0.63 over 447 trading days, tested against Newey-West "
       "standard errors. Both regressions point the same way. The first establishes that the "
       "returns are market and size exposure rather than selection; the second shows that once "
       "that exposure is subtracted there is nothing left to find.")

txt = open(MD, encoding="utf-8").read()
if old not in txt:
    raise SystemExit("原文未找到（可能已改过）")
txt = txt.replace(old, new)
open(MD, "w", encoding="utf-8").write(txt)
print("4.3 已修正：原 %d 字符 -> 新 %d 字符（净 +%d）" % (len(old), len(new), len(new) - len(old)))
