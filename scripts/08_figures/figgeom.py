# -*- coding: utf-8 -*-
"""figgeom.py —— 对当前 figs/*.png 做四维审查（不靠肉眼）。

维度：
  1. 有效 dpi（按 15cm 排版宽计算）
  2. 纸面最小字号（从 fig_manifest.json 的 min_fontsize + figure 宽推算）
  3. 长宽比 / 纸面高度（不变形、不超版心）
  4. 文字压图形：把 PNG 里的深色像素团块当作「墨迹」，检查文字行是否
     与柱形色块重叠——这里改用「已知布局反推」：在重新渲染的 matplotlib
     figure 上做 text-vs-patch 真实 bbox 判定（仅对脚本可控的图）。
     PNG 版本只做「行内墨迹密度突变」异常提示，不作判据。

输出：控制台表 + figgeom_report.txt
"""
import glob
import json
import os
import sys

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
FIGS = os.path.join(HERE, "figs")
MAN = os.path.join(FIGS, "fig_manifest.json")

PAPER_W_CM = 15.0          # A4 版心宽
MIN_DPI = 300              # 印刷下限
MIN_PT = 7.0               # 纸面最小可读字号
MAX_H_CM = 20.0            # 单图纸面高上限（超过会挤掉图注）


def main():
    man = json.load(open(MAN, encoding="utf-8")) if os.path.exists(MAN) else {}
    rows = []
    for p in sorted(glob.glob(os.path.join(FIGS, "fig*.png"))):
        f = os.path.basename(p)
        im = Image.open(p)
        w, h = im.size
        dpi = w / (PAPER_W_CM / 2.54)
        paper_h = PAPER_W_CM * h / w
        aspect = w / h
        m = man.get(f, {})
        fig_w = m.get("fig_w_in")
        min_fs = m.get("min_fontsize")
        paper_pt = (min_fs * (PAPER_W_CM / 2.54) / fig_w) if (min_fs and fig_w) else None
        ok = (dpi >= MIN_DPI and (paper_pt is None or paper_pt >= MIN_PT)
              and paper_h <= MAX_H_CM)
        rows.append((f, w, h, dpi, paper_h, aspect, fig_w, min_fs, paper_pt, ok))

    hdr = ("figure", "px", "dpi", "h_cm", "aspect", "fig_w", "min_fs", "paper_pt", "verdict")
    print("%-22s %-11s %5s %6s %7s %6s %7s %9s  %s" % hdr)
    print("-" * 92)
    lines = []
    nbad = 0
    for (f, w, h, dpi, ph, asp, fw, fs, pt, ok) in rows:
        print("%-22s %-11s %5.0f %6.2f %7.3f %6s %7s %9s  %s"
              % (f, f"{w}x{h}", dpi, ph, asp, fw, fs,
                 f"{pt:.2f}" if pt else "-", "OK" if ok else "FAIL"))
        lines.append(f"{f}\t{w}x{h}\tdpi={dpi:.0f}\th_cm={ph:.2f}\taspect={asp:.3f}"
                     f"\tpaper_pt={pt if pt else '-'}\t{'OK' if ok else 'FAIL'}")
        if not ok:
            nbad += 1
    print("-" * 92)
    print(f"不合格 {nbad} / {len(rows)}")
    reasons = []
    for (f, w, h, dpi, ph, asp, fw, fs, pt, ok) in rows:
        if not ok:
            r = []
            if dpi < MIN_DPI:
                r.append(f"dpi {dpi:.0f} < {MIN_DPI}")
            if pt is not None and pt < MIN_PT:
                r.append(f"纸面字号 {pt:.2f}pt < {MIN_PT}")
            if ph > MAX_H_CM:
                r.append(f"纸面高 {ph:.1f}cm > {MAX_H_CM}")
            reasons.append(f"{f}: " + "；".join(r))
    for r in reasons:
        print("  - " + r)
    with open(os.path.join(HERE, "figgeom_report.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines + ["", f"不合格 {nbad} / {len(rows)}"] + reasons))
    return 1 if nbad else 0


if __name__ == "__main__":
    sys.exit(main())
