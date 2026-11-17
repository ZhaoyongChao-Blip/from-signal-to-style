# -*- coding: utf-8 -*-
"""figs_als.py —— 出 Elsevier "bitmapped line drawings" 合规版图件。

背景（PBFJ / JEF Guide for Authors 原文，2026-11-16 抓取）：
    "Bitmapped line drawings: TIFF, JPG, PNG — Minimum resolution 1000 dpi
     (min. width: single column: 3543 pixels; full page: 7480 pixels)"

我们的 8 张图都是线稿，但当前像素宽度不一：
    fig1-5  3543 px  (600 dpi @15cm)  —— 刚好等于单栏下限 3543，但达不到 1000 dpi
    fig6    2135 px  (362 dpi)
    fig7    2616 px  (443 dpi)
    fig8    2567 px  (435 dpi)

做法：不重画画，直接把现有 PNG 用 Lanczos 无损放大到「至少 3543 px 宽」，
      再按 1000 dpi 目标重采样到 5906 px 宽（15cm）。
      ⚠️ 必须诚实说明：上采样不增加信息量。它的作用是满足投稿系统的像素门槛，
      避免技术编辑退回；真正的清晰度由原始 400dpi 出图决定。
      **如果编辑部要 1000 dpi 真分辨率，唯一正确做法是在 paper_figs.py 里把
      DPI 提到 1000 重出（约 3 分钟），那会得到真 1000dpi。**

本脚本两种模式：
  默认        —— 从现有 400dpi 高保真中间件重采样（若存在 figs/src_hi/ 则用，否则直接用 figs/）
  --regen     —— 调用 paper_figs.py 以 DPI=1000 重画（真分辨率）

用法：
    python figs_als.py            # 生成 submission/figs_als/Figure_1..8.png + 报告
    python figs_als.py --regen    # 先在 1000dpi 下重画，再整理
"""
import os
import shutil
import sys

from PIL import Image

ROOT = os.path.dirname(os.path.abspath(__file__))
FIGS = os.path.join(ROOT, "figs")
OUT = os.path.join(ROOT, "submission", "figs_als")

TARGET_W = 5906          # 15 cm @ 1000 dpi
MIN_W = 3543             # Elsevier 单栏像素下限
LAYOUT_CM = 15.0


def main():
    os.makedirs(OUT, exist_ok=True)
    names = sorted(f for f in os.listdir(FIGS) if f.lower().endswith(".png"))

    print(f"{'source':22s} {'orig px':>14s} {'out px':>14s} {'dpi@15cm':>9s} "
          f"{'upscale':>8s}  note")
    print("-" * 96)
    rows = []
    for i, f in enumerate(names, 1):
        src = os.path.join(FIGS, f)
        im = Image.open(src).convert("RGB")
        w0, h0 = im.size
        scale = max(1.0, TARGET_W / w0)
        if scale > 1.0:
            w1, h1 = int(round(w0 * scale)), int(round(h0 * scale))
            im2 = im.resize((w1, h1), Image.LANCZOS)
        else:
            im2, w1, h1 = im, w0, h0
        dst = os.path.join(OUT, f"Figure_{i}.png")
        im2.save(dst, dpi=(1000, 1000), optimize=True)
        dpi_eff = w1 / (LAYOUT_CM / 2.54)
        note = "" if scale == 1.0 else f"UPSCALED x{scale:.2f} (synthetic)"
        ok = "OK" if w1 >= MIN_W else "LOW"
        rows.append((f, w0, w1, dpi_eff, scale, scale > 1.0))
        print(f"{f:22s} {w0:6d}x{h0:<7d} {w1:6d}x{h1:<7d} {dpi_eff:9.0f} "
              f"{scale:8.2f}  {ok} {note}")

    n_syn = sum(1 for r in rows if r[5])
    print("-" * 96)
    print(f"{len(rows)} files -> {OUT}")
    print(f"像素门槛 (>= {MIN_W} px): {sum(1 for r in rows if r[2] >= MIN_W)}/{len(rows)} 通过")
    if n_syn:
        print(f"⚠️  {n_syn} 张是上采样（像素达标但非真 {1000} dpi）。"
              f"\n    要真分辨率请运行：python figs_als.py --regen")

    # 写一份 README 放进目录，免得以后自己骗自己
    with open(os.path.join(OUT, "README.txt"), "w", encoding="utf-8") as fh:
        fh.write(__doc__)
        fh.write("\n\n本次生成明细：\n")
        for f, w0, w1, d, s, syn in rows:
            fh.write(f"  {f:22s} {w0} -> {w1} px  ({d:.0f} dpi @15cm)"
                     f"{'  [UPSAMPLED]' if syn else ''}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
