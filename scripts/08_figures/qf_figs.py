# -*- coding: utf-8 -*-
"""qf_figs.py —— 按 Quantitative Finance 要求重新整理图件。

QF 官方要求（2026-11-16 抓自 tandfonline Instructions for Authors）：
  · "Figures should be saved separately from the text."
  · "Figures should be supplied in one of our preferred file formats: PS, JPEG,
     TIFF, or Microsoft Word (DOC or DOCX)"   ← **不接受 PNG**
  · "1200 dpi for line art, 600 dpi for grayscale and 300 dpi for colour"
  · "figure captions (as a list)"  ← 所有图注汇总成单独清单

本脚本做三件事：
  1. figs/ 的 PNG -> submission/qf_figs/Figure_N.tif（LZW 无损压缩，不损清晰度）
  2. 图注清单 -> submission/qf_figs/Figure_Captions.txt（从 paper.md 抽取）
  3. 校验每张 TIFF 的像素宽度与有效 dpi

用法：python qf_figs.py
"""
import os
import re
import sys

from PIL import Image
import fitz

ROOT = os.path.dirname(os.path.abspath(__file__))
FIGS = os.path.join(ROOT, "figs")
MD = os.path.join(ROOT, "paper", "paper.md")
OUT = os.path.join(ROOT, "submission", "qf_figs")

LAYOUT_CM = 15.0
MIN_DPI = 1200          # line art


def export_tiffs():
    os.makedirs(OUT, exist_ok=True)
    names = sorted(f for f in os.listdir(FIGS) if f.lower().endswith(".png"))
    rows = []
    for i, f in enumerate(names, 1):
        src = os.path.join(FIGS, f)
        dst = os.path.join(OUT, f"Figure_{i}.tif")
        im = Image.open(src).convert("RGB")
        im.save(dst, format="TIFF", compression="tiff_lzw", dpi=(1200, 1200))
        pix = fitz.Pixmap(src)
        dpi = pix.width / (LAYOUT_CM / 2.54)
        rows.append((i, f, pix.width, pix.height, dpi,
                     os.path.getsize(dst) / 1024))
    return rows


def extract_captions():
    """从 paper.md 抽取图注。图注形如 '**Fig. 3** ...' 或 'Fig. 3 ...' 开头的段。

    坑：最后一条图注可能后面直接跟标题（如 '## 7. Discussion'），中间没有空行，
    用 (?=\n\n) 做前瞻会把它当成“没结束”而漏掉。所以先按行收集，
    遇到新的 '#' 标题或下一张图注就截断。
    """
    t = open(MD, encoding="utf-8").read()
    lines = t.split("\n")
    caps = []
    cur_n, cur_buf = None, []

    fig_re = re.compile(r"^\*{0,2}Fig\.\s*(\d)\*{0,2}\s+(.*)$")

    def flush():
        if cur_n is not None and cur_buf:
            caps.append((cur_n, re.sub(r"\s+", " ", " ".join(cur_buf)).strip()))

    for ln in lines:
        m = fig_re.match(ln.strip())
        if m:
            flush()
            cur_n, cur_buf = int(m.group(1)), [m.group(2)]
            continue
        if cur_n is not None:
            s = ln.strip()
            # 遇到标题 / 空行之后再出现表格或新节 => 结束当前图注
            if s.startswith("#") or s.startswith("|") or s.startswith("**Table"):
                flush()
                cur_n, cur_buf = None, []
            elif s == "":
                flush()
                cur_n, cur_buf = None, []
            else:
                cur_buf.append(s)
    flush()

    seen, out = set(), []
    for n, c in caps:
        if n not in seen:
            seen.add(n)
            out.append((n, c))
    out.sort()
    return out


def main():
    rows = export_tiffs()
    print(f"{'#':>2} {'source':22s} {'px':>13s} {'dpi@15cm':>9s} {'TIFF KB':>8s}  verdict")
    print("-" * 74)
    bad = 0
    for i, f, w, h, dpi, kb in rows:
        ok = dpi >= MIN_DPI
        if not ok:
            bad += 1
        print(f"{i:>2} {f:22s} {w:6d}x{h:<6d} {dpi:9.0f} {kb:8.0f}  "
              f"{'OK' if ok else 'LOW-DPI'}")
    print("-" * 74)
    print(f"TIFF 数 {len(rows)}，dpi 不达标 {bad}")

    caps = extract_captions()
    print(f"\n图注抽取：{len(caps)} 条")
    for n, c in caps:
        print(f"  Fig. {n}: {c[:90]}{'...' if len(c) > 90 else ''}")

    cpath = os.path.join(OUT, "Figure_Captions.txt")
    with open(cpath, "w", encoding="utf-8") as fh:
        fh.write("Figure captions (Quantitative Finance requires captions as a list,\n"
                 "separate from the figure files)\n\n")
        for n, c in caps:
            fh.write(f"Fig. {n}. {c}\n\n")
    print(f"\n[written] {OUT}")
    print(f"[written] {cpath}")
    return 0 if bad == 0 and len(caps) == len(rows) else 1


if __name__ == "__main__":
    sys.exit(main())
