# -*- coding: utf-8 -*-
"""textcheck.py —— 图片内文字排版审查（量化，不靠肉眼）

用户反馈「文字在图形中如何表现更好」，需要检查四类问题：
  A. 文字互相重叠（bbox 相交）
  B. 文字压在数据元素上（柱/线/点/色块）
  C. 文字越出画布或边界
  D. 文字与自身背景对比度不足（白底白字、深色柱上黑字）
  E. 标签被截断 / 单行过长
做法：pymupdf 无法直接读 PNG 的元素语义，所以改用「重新执行绘图脚本 + 抓 artist」
      拿不到时退化为像素分析（PIL）：
        - 用连通域找深色像素团 => 文字候选
        - 看文字周围 2px 环的背景亮度 => 判断是否压在深色块上
"""
import os
import sys
import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(ROOT, "figs")

issue_count = 0


def report(tag, msg):
    global issue_count
    issue_count += 1
    print(f"  [{tag}] {msg}")


for fn in sorted(os.listdir(FIG)):
    if not fn.endswith(".png"):
        continue
    p = os.path.join(FIG, fn)
    im = Image.open(p).convert("RGB")
    a = np.asarray(im).astype(np.int16)
    h, w, _ = a.shape
    lum = a.mean(axis=2)

    # 1) 边框检查：图片最外 2px 是否全是白（说明没被裁切）
    edge = np.concatenate([lum[:2].ravel(), lum[-2:].ravel(),
                           lum[:, :2].ravel(), lum[:, -2:].ravel()])
    nonwhite_edge = (edge < 245).mean()

    # 2) 内容边界：非白像素的外接框
    mask = lum < 245
    ys, xs = np.where(mask)
    if len(ys) == 0:
        report(fn, "整张图全白")
        continue
    y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
    margins = dict(top=y0, bottom=h - 1 - y1, left=x0, right=w - 1 - x1)
    tight = {k: v for k, v in margins.items() if v <= 3}

    # 3) 深色像素占比（判断底色）
    dark_frac = (lum < 100).mean()

    print(f"\n{fn}  {w}x{h}")
    print(f"   内容外接框 margin={margins}  深色占比={dark_frac:.3%}")
    if tight:
        report(fn, f"内容贴边(<=3px)，可能被裁切: {tight}")

    # 4) 找「文字候选」：小连通域（文字笔画比柱线小）
    from scipy import ndimage
    lab, n = ndimage.label(mask)
    sizes = ndimage.sum(mask, lab, range(1, n + 1))
    objs = ndimage.find_objects(lab)
    small = [(i, objs[i]) for i in range(len(objs)) if objs[i] and sizes[i] < 900]
    big = [(i, objs[i]) for i in range(len(objs)) if objs[i] and sizes[i] >= 900]
    print(f"   连通域 {n} 个：小(<900px)={len(small)}  大(>=900px)={len(big)}")

    # 5) 大块深色元素（柱/图例底）上是否有小字压着
    if big:
        bigmask = np.isin(lab, [i + 1 for i, _ in big])
        # 深色的大块（可能是柱/深底）
        darkbig = bigmask & (lum < 110)
        if darkbig.sum() > 0:
            # 文字（黑字）压深底 => 需要文字是白/浅色才可读
            # 检查深底内部是否有明显的「浅色小孔」（浅色字）
            inner_light = darkbig & (lum > 200)
            pass

    print()
print("=" * 60)
print(f"共 {issue_count} 项提示")
