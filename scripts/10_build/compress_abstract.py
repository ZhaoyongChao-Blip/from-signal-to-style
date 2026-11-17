# -*- coding: utf-8 -*-
"""compress_abstract.py — 把摘要压到 <=250 词（Word 口径），并核对每个数字在正文中存在。

计数口径（沿用 rewrite_abstract.py 的教训）：
  期刊用 Word/ScholarOne 数空白分隔 token，连字符按 1 个词算，但 PDF 换行会把
  "nearest-neighbour" 断成两个 token。定案：len(s.replace("-"," ").split())，
  最后必须回 PDF 实测。

新摘要删除的数字全部在正文可查，且每个保留数字都用 assert 回查 paper.md。
Run: python compress_abstract.py            dry run
     python compress_abstract.py --apply    写回 paper.md（留 .bak）
"""
import os
import re
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
MD = os.path.join(HERE, "paper", "paper.md")

NEW = (
 "Technical factors are widely used in quantitative equity selection, yet their predictive "
 "power often fails in trading. We audit the whole chain on the Chinese A-share market: 46 "
 "factors, a LightGBM ranker and twelve years on 3,317 stocks. Overlapping forward 5-day "
 "labels inflate the daily IC t-statistic about sqrt(5); non-overlapping striding leaves it "
 "significant (t = 8.30), although every factor has absolute IC below 0.10. Most of the edge "
 "is style: four style factors explain an R-squared of 0.1655 of forward excess return, and "
 "the long book is a low-beta portfolio (beta 0.48, CAPM alpha 11.2% with t = 0.85). "
 "Neutralizing the label against those styles raises test AUC from 0.5353 to 0.5525, and the "
 "score earns 0.713% per five days within style cells (t = 3.47, positive in all eight "
 "quarters); a randomized placebo shows those cells unbiased, whereas nearest-neighbour "
 "matching is biased by 3.3 points. The alpha still does not monetize: 5-day smoothing lifts "
 "the break-even cost from 10 to 24 basis points, but the entire +18.17 percentage point "
 "excess arises in 2024H1, when the benchmark fell 7.33% and 10.00%, and the other six "
 "quarters sum to -13.3. Conditioning on market direction makes the edge signed, losing 0.50% "
 "on up days and gaining 0.66% on down days (t = -6.30, +8.10). The excess is avoidance of a "
 "small-cap crash, not skill. Style-cell benchmarking, randomized placebos and quarterly "
 "decomposition should be mandatory."
)

# 摘要里出现的每个数字都必须在正文里出现（防编造）
NUMS = ["46", "3,317", "5", "8.30", "0.10", "0.1655", "0.48", "11.2%", "0.85",
        "0.5353", "0.5525", "0.713%", "3.47", "3.3", "10", "24", "18.17",
        "7.33%", "10.00%", "-13.3", "0.50%", "0.66%", "-6.30", "+8.10"]


def wc(s):
    return len(s.replace("-", " ").split())


def main():
    apply = "--apply" in sys.argv
    t = open(MD, encoding="utf-8").read()
    i = t.find("## Abstract")
    j = t.find("**Keywords**")
    old = t[i + len("## Abstract"):j].strip()

    print(f"old abstract: {wc(old)} words (Word convention)")
    print(f"new abstract: {wc(NEW)} words (Word convention)")

    body = t[j:]
    missing = [n for n in NUMS if n not in body]
    if missing:
        print(f"[FAIL] numbers not found in body: {missing}")
        return 1
    print(f"[ok] all {len(NUMS)} numbers traced to body")

    if wc(NEW) > 250:
        print("[FAIL] still over 250")
        return 1
    if wc(NEW) < 150:
        print("[WARN] under 150")

    if apply:
        shutil.copy(MD, MD + ".bak_abs250")
        t2 = t[:i] + "## Abstract\n\n" + NEW + "\n\n" + t[j:]
        open(MD, "w", encoding="utf-8").write(t2)
        print(f"[written] {MD}  (backup .bak_abs250)")
    else:
        print("(dry run; pass --apply to write)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
