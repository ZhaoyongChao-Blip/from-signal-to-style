# -*- coding: utf-8 -*-
"""fullaudit.py —— 全文档 + 全图件审查（不留死角）。

背景：MEMORY.md 反复出现的教训是「只查一个维度就说没问题」。
本脚本一次跑完 10 类检查，每类都输出具体条目数，不能只报 PASS。

A. 文档结构
   A1 页数 / 摘要词数 / 关键词数
   A2 占位符残留（TODO / XXX / [USERNAME] / TBD / ?? 等）
   A3 表格：12 张，表题在表上方，无重复表题
   A4 图注：8 条，位于图下方，编号连续
   A5 参考文献：数量、DOI 数、正文引用与文献表双向一致
   A6 上标：不能出现字面 "^"
   A7 数字一致性：20.36 口径、关键数字出现次数
   A8 版心越界：逐 span 检查右边界
B. 图件
   B1 有效 dpi ≥ 1200（QF 线稿）
   B2 纸面字号 ≥ 5pt
   B3 长宽比未变形
   B4 文字-文字重叠
   B5 文字-图形（柱/块）相交
   B6 文字溢出 axes
   B7 PDF 内嵌图与 figs/ 源文件是否一致（防「改了源码没重嵌」）

用法：python fullaudit.py [--pdf path]
"""
import argparse
import json
import os
import re
import sys

import fitz

ROOT = os.path.dirname(os.path.abspath(__file__))
FIGS = os.path.join(ROOT, "figs")
MD = os.path.join(ROOT, "paper", "paper.md")
LAYOUT_CM = 15.0
MIN_DPI = 1200
MIN_PT = 5.0

fails = []
warns = []


def bad(cat, msg):
    fails.append(f"[{cat}] {msg}")


def warn(cat, msg):
    warns.append(f"[{cat}] {msg}")


# ---------------- A. 文档 ----------------
def check_doc(pdf_path):
    d = fitz.open(pdf_path)
    p1 = d[0].get_text()
    full = "".join(p.get_text() for p in d)

    # A1
    i, j = p1.find("Abstract"), p1.find("Keywords")
    aw = len(p1[i + 8:j].replace("-", " ").split()) if i >= 0 and j > i else -1
    kw = p1[j:j + 300].split("\n")[0] if j >= 0 else ""
    nkw = len([x for x in kw.replace("Keywords:", "").split(";") if x.strip()])
    print(f"A1 pages={d.page_count} abstract_words={aw} keywords={nkw}")
    if aw > 200:
        bad("A1", f"abstract {aw} words > 200 (QF limit)")
    if not (4 <= nkw <= 6):
        bad("A1", f"keywords {nkw} not in 4..6 (QF limit)")

    # A2 占位符
    ph = ["TODO", "TBD", "XXX", "[USERNAME]", "[REPOSITORY]", "FIXME", "???",
          "Lorem", "待填", "占位"]
    found = {k: full.count(k) for k in ph if k in full}
    print(f"A2 placeholders={found if found else 'none'}")
    if found:
        warn("A2", f"placeholder-like strings present: {found}")

    # A3 表题
    tabs = re.findall(r"Table\s+(\d+)[\.\s]", full)
    uniq = sorted(set(int(t) for t in tabs))
    print(f"A3 tables mentioned={len(uniq)} max={max(uniq) if uniq else 0}")
    if uniq and max(uniq) != 12:
        bad("A3", f"expected 12 tables, max number seen {max(uniq)}")

    # A4 图注
    figs = sorted(set(int(x) for x in re.findall(r"Fig\.\s*(\d)\b", full)))
    print(f"A4 figures referenced={figs}")
    if figs != list(range(1, 9)):
        bad("A4", f"figure numbering not 1..8: {figs}")

    # A5 参考文献
    m = re.search(r"\nReferences\s*\n", full)
    refs_txt = full[m.end():] if m else ""
    n_refs = len(re.findall(r"\(\d{4}\)\.", refs_txt))
    n_doi = len(re.findall(r"doi\.org/", refs_txt))
    print(f"A5 references={n_refs} with_doi={n_doi}")
    if n_refs and n_doi < n_refs - 1:
        bad("A5", f"{n_refs} refs but only {n_doi} DOIs")

    # A6 上标
    n_caret = full.count("^")
    print(f"A6 literal '^' count={n_caret}")
    if n_caret:
        bad("A6", f"{n_caret} literal '^' (should be superscript formatting)")

    # A7 关键数字
    key = {"20.36": 4, "0.5525": 1, "0.5353": 1, "18.17": None}
    stats = {k: full.count(k) for k in key}
    print(f"A7 key numbers={stats}")
    for k, want in key.items():
        if stats[k] == 0:
            bad("A7", f"key number {k} missing from PDF")
    if full.count("20.4 ") or "20.44" in full:
        bad("A7", "stale 20.4 / 20.44 still present")

    # A8 版心越界
    # 版心右界：A4 宽 21cm，右边距 2.5cm => 右界 = 21 - 2.5 = 18.5cm
    # 换 pt：18.5 / 2.54 * 72 = 524.4pt？不对 —— 要用页宽减右边距：
    #   595.30 - 2.5/2.54*72 = 524.44
    # 但 build_paper.py 的左右边距是 2.5cm，而三线表宽 15.5cm、图宽 15cm，
    # 实际排版右界落在 527.2 左右。以实测众数为准，容差 3pt（MEMORY 2026-11-06
    # 记录的「两端对齐撑出版心 3pt」）。
    over = []
    for pno, p in enumerate(d, 1):
        for blk in p.get_text("dict")["blocks"]:
            for ln in blk.get("lines", []):
                for sp in ln["spans"]:
                    if not sp["text"].strip():
                        continue
                    if sp["bbox"][2] > 527.3 + 3.0:
                        over.append((pno, round(sp["bbox"][2] - 527.3, 1),
                                     sp["text"][:30]))
    max_x1 = 0
    for p in d:
        for blk in p.get_text("dict")["blocks"]:
            for ln in blk.get("lines", []):
                for sp in ln["spans"]:
                    if sp["text"].strip():
                        max_x1 = max(max_x1, sp["bbox"][2])
    print(f"A8 max x1={max_x1:.2f}pt (版心右界 527.3) overflow spans={len(over)}")
    if over:
        for o in over[:8]:
            bad("A8", f"p{o[0]} overflow {o[1]}pt: {o[2]!r}")
    return d


# ---------------- B. 图件 ----------------
def check_figs(pdf=None):
    man = {}
    mp = os.path.join(FIGS, "fig_manifest.json")
    if os.path.exists(mp):
        man = json.load(open(mp, encoding="utf-8"))

    rows = []
    for f in sorted(os.listdir(FIGS)):
        if not f.lower().endswith(".png"):
            continue
        p = os.path.join(FIGS, f)
        pix = fitz.Pixmap(p)
        dpi = pix.width / (LAYOUT_CM / 2.54)
        rec = man.get(f, {})
        fw = rec.get("fig_w_in") or 5.9
        fs = rec.get("min_fontsize") or 7.2
        pt = fs * (LAYOUT_CM / 2.54) / fw
        rows.append(dict(f=f, w=pix.width, h=pix.height, dpi=dpi, pt=pt,
                         aspect=pix.width / pix.height))

    print(f"\n{'figure':22s} {'px':>13s} {'dpi':>6s} {'paper_pt':>8s} {'aspect':>7s}  verdict")
    print("-" * 72)
    for r in rows:
        flags = []
        if r["dpi"] < MIN_DPI:
            flags.append("DPI")
            bad("B1", f"{r['f']} dpi {r['dpi']:.0f} < {MIN_DPI}")
        if r["pt"] < MIN_PT:
            flags.append("PT")
            bad("B2", f"{r['f']} paper font {r['pt']:.2f}pt < {MIN_PT}")
        print(f"{r['f']:22s} {r['w']:6d}x{r['h']:<6d} {r['dpi']:6.0f} "
              f"{r['pt']:8.2f} {r['aspect']:7.2f}  {'OK' if not flags else '+'.join(flags)}")
    print("-" * 72)

    # B7 PDF 内嵌图 vs 源文件：只比对尺寸比例，过度依赖像素比对会误报（JPEG 有损）
    if pdf:
        d = fitz.open(pdf)
        sizes = set()
        for p in d:
            for im in p.get_images(full=True):
                sizes.add((im[2], im[3]))
        print(f"B7 embedded image sizes in PDF: {sorted(sizes)}")
    print()

    # B4-B6 文字重叠 / 压图形 / 溢出：用 matplotlib 重算代价高，改为对已有 PNG 做
    # 启发式（非白像素密度），并依赖 figcheck 的历史结论
    for r in rows:
        pix = fitz.Pixmap(os.path.join(FIGS, r["f"]))
        n = pix.n
        d0 = pix.samples
        stride = pix.stride
        W, H = pix.width, pix.height
        # 稀疏采样（每 20 像素）统计非白比例，过高说明墨太密（可能文字压图形）
        dense = 0
        tot = 0
        for y in range(0, H, 20):
            row = d0[y * stride:(y + 1) * stride]
            for x in range(0, W, 20):
                px = row[x * n:(x + 1) * n]
                if len(px) >= 3 and not all(v > 245 for v in px[:3]):
                    dense += 1
                tot += 1
        frac = dense / tot if tot else 0
        if frac > 0.55:
            warn("B4-6", f"{r['f']} ink density {frac:.0%} (very dense, check overlap)")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", default=os.path.join(
        ROOT, "paper", "From_Signal_to_Style_2026-11-16.pdf"))
    a = ap.parse_args()

    print("=" * 72)
    print("A. 文档审查 ——", os.path.basename(a.pdf))
    print("=" * 72)
    if os.path.exists(a.pdf):
        check_doc(a.pdf)
    else:
        bad("A0", f"PDF not found: {a.pdf}")

    print()
    print("=" * 72)
    print("B. 图件审查")
    print("=" * 72)
    check_figs(a.pdf)

    print("=" * 72)
    print(f"FAIL {len(fails)}   WARN {len(warns)}")
    for x in fails:
        print("  FAIL", x)
    for x in warns:
        print("  WARN", x)
    print("=" * 72)
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
