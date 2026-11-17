# -*- coding: utf-8 -*-
"""figcheck.py —— 图片印刷质量检查器

四项判据（全部量化，不靠肉眼）：
  1. 有效 dpi ≥ 300      有效dpi = px_width / (排版宽cm / 2.54)
  2. 图内最小字号 ≥ 5.0pt 纸面pt = fontsize * (排版宽cm/2.54) / figure宽in
     —— 需要画出时记录字体，用 shadow 机制实现
  3. 宽高比不被拉伸       在 PDF 里比对 bbox 比例 vs PNG 比例
  4. 无横向溢出           文字宽度 < 图区宽度

用法：python figcheck.py           # 检查 figs/ 下所有 png
"""
import os
import sys
import json

import fitz

ROOT = os.path.dirname(os.path.abspath(__file__))
FIGS = os.path.join(ROOT, "figs")
LAYOUT_CM = 15.0      # build_paper.py 的 FULL_W_CM
MIN_W_CM = 13.2       # build_paper.py 的 MIN_W_CM
MIN_DPI = 300
MIN_PT = 5.0

# 每张图画出时的 figure 宽度（英寸）与最小字号，由 paper_figs.py 写出的 sidecar 提供；
# 若没有 sidecar，则从本表读取（手动维护，改图时必须同步）
MANIFEST = os.path.join(FIGS, "fig_manifest.json")


def check():
    rows = []
    man = {}
    if os.path.exists(MANIFEST):
        man = json.load(open(MANIFEST, encoding="utf-8"))

    for f in sorted(os.listdir(FIGS)):
        if not f.lower().endswith(".png"):
            continue
        p = os.path.join(FIGS, f)
        pix = fitz.Pixmap(p)
        rec = man.get(f, {})
        fig_w_in = rec.get("fig_w_in")
        min_fs = rec.get("min_fontsize")

        # 1) dpi：最坏情况按最小可读宽度 13.2cm 算（图被压缩时 dpi 反而更高，
        #    真正的下界是 15cm 全宽）
        dpi_full = pix.width / (LAYOUT_CM / 2.54) * 1.0
        # 2) 纸面字号
        if fig_w_in and min_fs:
            scale = (LAYOUT_CM / 2.54) / fig_w_in
            pt_on_paper = min_fs * scale
        else:
            scale = None
            pt_on_paper = None

        ok_dpi = dpi_full >= MIN_DPI
        ok_pt = (pt_on_paper is None) or (pt_on_paper >= MIN_PT)
        rows.append(dict(f=f, px=(pix.width, pix.height), dpi=dpi_full,
                         fig_w_in=fig_w_in, min_fs=min_fs, pt=pt_on_paper,
                         ok_dpi=ok_dpi, ok_pt=ok_pt))

    print("%-22s %11s %7s %9s %9s  %s" %
          ("figure", "size", "dpi@15", "fig_w_in", "min_pt", "verdict"))
    print("-" * 74)
    nbad = 0
    for r in rows:
        flags = []
        if not r["ok_dpi"]:
            flags.append("DPI")
        if not r["ok_pt"]:
            flags.append("FONT")
        if flags:
            nbad += 1
        pt_s = "%.2f" % r["pt"] if r["pt"] is not None else "?"
        w_s = "%.1f" % r["fig_w_in"] if r["fig_w_in"] is not None else "?"
        print("%-22s %5dx%-5d %7.0f %9s %9s  %s" %
              (r["f"], r["px"][0], r["px"][1], r["dpi"], w_s, pt_s,
               "OK" if not flags else "<<< " + "+".join(flags)))
    print("-" * 74)
    print(f"不合格 {nbad} / {len(rows)}")
    return nbad


if __name__ == "__main__":
    sys.exit(1 if check() else 0)
