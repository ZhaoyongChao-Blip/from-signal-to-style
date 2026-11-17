# -*- coding: utf-8 -*-
"""compress_abstract_qf.py —— Quantitative Finance 摘要压到 <=200 词。

QF 官方要求（2026-11-16 抓自 tandfonline Instructions for Authors）：
    "Should contain an unstructured abstract of 200 words."

原 244 词 -> 200 词。原则：
  · 保留全部关键数字（NUMS 逐个 assert 存在于正文）
  · 只删连接词与重复表述
  · 词数用 Word / ScholarOne 口径：连字符拆成空格后按空白切分

用法：
    python compress_abstract_qf.py            # 只报字数与数字核对
    python compress_abstract_qf.py --apply    # 写入 paper/paper.md
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
MD = os.path.join(HERE, "paper", "paper.md")

PARTS = [
    "Technical factors are widely used in equity selection, yet they often fail.",
    "We audit the chain on the Chinese A-share market with 46 factors, LightGBM and "
    "3,317 stocks over twelve years.",
    "Overlapping 5-day labels inflate the daily IC t-statistic about sqrt(5); "
    "non-overlapping striding leaves it significant (t = 8.30), though all factors "
    "have absolute IC below 0.10.",
    "Most of the edge is style: four factors explain 16.6% of forward excess return, "
    "and the long book is low-beta (beta 0.48, CAPM alpha 11.2%, t = 0.85).",
    "Neutralizing against them raises test AUC from 0.5353 to 0.5525, and the score "
    "earns 0.713% per five days within style cells (t = 3.47); a randomized placebo "
    "shows those cells unbiased, while nearest-neighbour matching is biased by 3.3 "
    "points.",
    "The alpha does not monetize: smoothing lifts the break-even cost from 10 to 24 "
    "basis points, but the +18.17 point excess arises entirely in 2024H1, when the "
    "benchmark fell 7.33% and 10.00%, and the six other quarters sum to -13.3.",
    "The edge is signed by market direction: -0.50% on up days, +0.66% on down days "
    "(t = -6.30, +8.10).",
    "It is avoidance of a small-cap crash, not skill.",
]

NEW = " ".join(PARTS)

NUMS = ["46", "3,317", "8.30", "0.10", "16.6%", "0.48", "11.2%", "0.85",
        "0.5353", "0.5525", "0.713%", "3.47", "3.3", "10", "24", "18.17",
        "7.33%", "10.00%", "13.3", "0.50%", "0.66%", "6.30", "8.10"]


def wc(s):
    return len(s.replace("-", " ").split())


def main():
    src = open(MD, encoding="utf-8").read()
    body = src[:src.find("## References")]

    n_new = wc(NEW)
    print(f"new abstract: {n_new} words (limit 200, headroom {200 - n_new})")
    bad = [x for x in NUMS if x not in body]
    print(f"numbers checked: {len(NUMS)}, missing in body: {bad if bad else 'none'}")
    ok = n_new <= 200 and not bad
    print("VERDICT:", "PASS" if ok else "FAIL")

    if "--apply" in sys.argv and ok:
        i = src.find("## Abstract")
        j = src.find("**Keywords**")
        src2 = src[:i] + "## Abstract\n\n" + NEW + "\n\n" + src[j:]
        open(MD, "w", encoding="utf-8").write(src2)
        chk = open(MD, encoding="utf-8").read()
        i2 = chk.find("## Abstract")
        j2 = chk.find("**Keywords**")
        print(f"[applied] verify: {wc(chk[i2:j2].replace('## Abstract', ''))} words")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
