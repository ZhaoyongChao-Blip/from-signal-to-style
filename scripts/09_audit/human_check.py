# -*- coding: utf-8 -*-
"""human_check.py —— MEMORY.md 要求的「人眼清单」四项数值化检查。

① 占位符扫描  ② 章节顺序  ③ 每类内容出现次数  ④ 首页头部逐行 span
用法: python human_check.py [pdf路径]
"""
import sys, re
import fitz

PDF = sys.argv[1] if len(sys.argv) > 1 else r"paper/From_Signal_to_Style_2026-11-15.pdf"
d = fitz.open(PDF)
t = "".join(p.get_text() for p in d)
print(f"[pdf] {PDF}  pages={d.page_count}")

print("\n=== 1. 占位符扫描（应全 0，除数学区间 [ ]） ===")
for pat in ["TODO", "XXX", "to be inserted", "to be added", "TBD", "**", "##", "\\cite", "??"]:
    n = t.count(pat)
    flag = "" if n == 0 else "  <<< 需处理"
    print(f"  {pat:18s} {n}{flag}")

print("\n=== 2. 章节顺序 ===")
order = []
pat = r"^(Abstract|Keywords|References|Figures|Data and code availability|\d+\.\s+[A-Z][^\n]{0,60})$"
for m in re.finditer(pat, t, re.M):
    s = " ".join(m.group(1).split())
    if s not in order:
        order.append(s)
for s in order:
    print("  ", s)
# 顺序断言：Data availability 应在 References 之前，References 在 Figures 之前
def idx(k):
    for i, s in enumerate(order):
        if s.startswith(k):
            return i
    return -1
da, rf, fg = idx("Data and code"), idx("References"), idx("Figures")
ok = (da == -1 or rf == -1 or da < rf) and (rf == -1 or fg == -1 or rf < fg)
print(f"  顺序检查: data<{rf}, references<figures  ->  {'OK' if ok else 'WRONG'}")

print("\n=== 3. 每类内容出现次数（图注/表题各应为 1 遍） ===")
figs = re.findall(r"^Fig\. \d+", t, re.M)
tabs = re.findall(r"^Table \d+\.", t, re.M)
print(f"  Fig. 标题行: {len(figs)}  (期望 7)")
print(f"  Table 标题行: {len(tabs)}  (期望 12)")
print(f"  'Figure captions' 清单块: {t.count('Figure captions')}  (期望 0，图注只留图下)")

print("\n=== 4. 首页头部逐行 span ===")
for b in d[0].get_text("dict")["blocks"]:
    if b.get("type") != 0:
        continue
    for ln in b["lines"]:
        txt = "".join(sp["text"] for sp in ln["spans"]).strip()
        if not txt:
            continue
        sp0 = ln["spans"][0]
        y = ln["bbox"][1]
        if y > 330:
            break
        sizes = {round(sp["size"], 1) for sp in ln["spans"]}
        fonts = {sp["font"] for sp in ln["spans"]}
        print(f"  y={y:5.1f} sz={sorted(sizes)} font={sorted(fonts)}  | {txt[:78]}")

print("\n=== 5. 上标检查（作者单位标记应为上标，不应出现字面 ^） ===")
caret = [m.group(0) for m in re.finditer(r".{20}\^.{20}", t)]
print(f"  字面 ^ 出现次数: {len(caret)}", "" if not caret else "<<< 需检查")
for c in caret[:5]:
    print("   ...", c.replace("\n", " "))
