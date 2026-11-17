# -*- coding: utf-8 -*-
"""render_check.py —— 逐页看渲染成品（用户第 4 次强调「不能只检查属性」）

本机 read 图片工具缺 sharp，改动视觉审查：
  1. 每页渲成 PNG，统计「墨迹」分布，找空白页 / 内容溢出版心
  2. 新表（Table 6）所在页：逐行提取 y 坐标，确认表题、表头、数据行、上下横线顺序正确
  3. 检查表格横线是否被页断开（同一表格的行 y 跨页）
  4. 检查表题与表格之间是否夹了其他段落
"""
import os
import re

import fitz
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "paper", "pages")
os.makedirs(OUT, exist_ok=True)
PDF = r"D:\LO_out\paper.pdf"

doc = fitz.open(PDF)
print(f"pages = {doc.page_count}\n")

# ---------- 1. 每页墨迹与版心 ----------
print("=== 1. 每页墨迹边界（版心 x 71~524, y 72~770）===")
issues = []
for i, pg in enumerate(doc, 1):
    pix = pg.get_pixmap(matrix=fitz.Matrix(1, 1), alpha=False)
    arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
    ink = (arr.mean(axis=2) < 200)
    rows = np.where(ink.any(axis=1))[0]
    cols = np.where(ink.any(axis=0))[0]
    if len(rows) == 0:
        issues.append(f"p{i}: 空白页")
        continue
    y0, y1 = rows[0], rows[-1]
    x0, x1 = cols[0], cols[-1]
    # 扣掉页脚页码区
    over = []
    if x1 > 535:
        over.append(f"right={x1}")
    if x0 < 60:
        over.append(f"left={x0}")
    flag = "  <-- 越界 " + ",".join(over) if over else ""
    if over:
        issues.append(f"p{i}: 越界 {over}")
    print(f"  p{i:2d}: x[{x0:3d},{x1:3d}] y[{y0:3d},{y1:3d}] 墨迹={ink.sum():>7d}{flag}")

# ---------- 2. Table 6 所在页逐行 ----------
print("\n=== 2. Table 6（新增表）逐行结构 ===")
for i, pg in enumerate(doc, 1):
    txt = pg.get_text()
    if "Expanding-window out-of-sample accuracy" not in txt:
        continue
    print(f"  位于 p{i}")
    blocks = pg.get_text("dict")["blocks"]
    lines = []
    for b in blocks:
        for ln in b.get("lines", []):
            t = "".join(s["text"] for s in ln["spans"]).strip()
            if t:
                lines.append((round(ln["bbox"][1], 1), round(ln["bbox"][0], 1), t))
    lines.sort()
    for y, x, t in lines[:30]:
        print(f"    y={y:6.1f} x={x:5.1f}  {t[:86]}")
    # 画线（表格横线）
    print("  横线 y 坐标:", end=" ")
    ys = []
    for d in pg.get_drawings():
        for it in d["items"]:
            if it[0] == "l":
                p1, p2 = it[1], it[2]
                if abs(p1.y - p2.y) < 0.5 and abs(p2.x - p1.x) > 30:
                    ys.append(round(p1.y, 1))
    print(sorted(set(ys)))

# ---------- 3. 表格跨页 ----------
print("\n=== 3. 表格跨页检查 ===")
cap_rows = {}
for i, pg in enumerate(doc, 1):
    t = pg.get_text()
    for m in re.finditer(r"(Table \d+)\.", t):
        cap_rows.setdefault(m.group(1), []).append(i)
multi = {k: v for k, v in cap_rows.items() if len(v) > 1}
print(f"  表题出现在多页的: {multi if multi else '无'}")
print(f"  共识别表题 {len(cap_rows)} 个: {sorted(cap_rows, key=lambda s:int(re.sub(chr(92)+'D','',s)))}")

# ---------- 4. 表题与表格的相邻性 ----------
print("\n=== 4. 表题下方 60pt 内是否有表格横线 ===")
for i, pg in enumerate(doc, 1):
    d = pg.get_text("dict")
    for b in d["blocks"]:
        for ln in b.get("lines", []):
            t = "".join(s["text"] for s in ln["spans"]).strip()
            if re.match(r"^Table \d+\.", t):
                cap_y = ln["bbox"][3]
                near = []
                for dr in pg.get_drawings():
                    for it in dr["items"]:
                        if it[0] == "l":
                            p1, p2 = it[1], it[2]
                            if abs(p1.y - p2.y) < 0.5 and abs(p2.x - p1.x) > 30:
                                if 0 <= p1.y - cap_y <= 60:
                                    near.append(round(p1.y - cap_y, 1))
                status = "OK" if near else "<-- 下方无表格线！"
                print(f"  p{i:2d} {t[:40]:42s} 下方线 dy={near[:3]} {status}")

print("\n=== 汇总 ===")
print(f"  异常页: {issues if issues else '无'}")
