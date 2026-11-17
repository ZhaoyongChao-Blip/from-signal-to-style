# -*- coding: utf-8 -*-
"""split_anonymous.py —— 按 Elsevier double-anonymized 要求拆分稿件。

官方原文（PBFJ / JEF Guide for Authors，2026-11-16 抓取）：
    "double anonymized review process, meaning author identities are concealed from
     reviewers and vice versa. The title page (including author details) and anonymized
     manuscript (excluding author details) need to be submitted as separate files."

产出：
    submission/Title_Page.docx          含标题 + 作者 + 单位 + 通讯 + Funding + 声明
    submission/Manuscript_Anonymous.docx 正文，删首部作者块，数据可得性改匿名表述

做法：不重排版。直接对 paper.docx 做 XML 级操作——
  1) 删掉以 "Authors", "School of", "Corresponding author", "Funding" 开头的段落
     （它们在 docx 里是首部独立的几个段落）
  2) 在正文 "Data and code availability" 段落里把 "from the corresponding author upon
     reasonable request" 改成不含作者指向的表述
  3) Title_Page.docx 用同一模板新建，只放元数据

用法：python split_anonymous.py
"""
import copy
import os
import re
import sys

from docx import Document
from docx.shared import Pt
from docx.enum.text import WD_ALIGN_PARAGRAPH

ROOT = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(ROOT, "paper", "paper.docx")
OUTDIR = os.path.join(ROOT, "submission")
TITLE_OUT = os.path.join(OUTDIR, "Title_Page.docx")
MANU_OUT = os.path.join(OUTDIR, "Manuscript_Anonymous.docx")

# 首部要删除的段落前缀
DROP_PREFIX = ("**Authors**", "Authors:", "Yongchao Zhao", "School of Physics",
               "**Corresponding author**", "Corresponding author:", "**Funding**",
               "Funding:", "Nanning Normal University", "1792552210")

# 数据可得性的匿名化替换
# 注意：正文措辞改过一次，这里必须同时覆盖新旧两种写法，
# 否则新句会原样进匿名版（自检会报 "corresponding author" 残留）。
AVAIL_OLD = "are available from the corresponding author"
AVAIL_OLD_FLAT = ("are available from the corresponding author upon reasonable request, "
                  "and will be deposited in a public repository upon acceptance.")
# 当前正文实际措辞（2026-11-16 改）
AVAIL_OLD2 = "as a versioned code archive from the corresponding author on request"
AVAIL_OLD2_FLAT = ("as a versioned code archive from the corresponding author on request, "
                   "and will be deposited in a public repository upon acceptance.")
# 双盲版：不能出现具体仓库地址（会暴露身份），也不能指向作者
AVAIL_NEW = "will be made openly available in a public code repository upon acceptance."

TITLE = ("From Signal to Style: A Full-Chain Reassessment of Technical-Factor "
         "Stock Selection in the Chinese A-Share Market")


def strip_front(doc):
    """删除首部作者信息段落。返回被删的文本（用于填 Title Page）。"""
    removed = []
    for p in list(doc.paragraphs):
        t = p.text.strip()
        if not t:
            continue
        # 只处理正文开始前的段落（## Abstract 之前）
        if t.startswith("Abstract"):
            break
        if any(t.startswith(k) for k in DROP_PREFIX) or "Nanning Normal University" in t:
            removed.append(t)
            p._element.getparent().remove(p._element)
    return removed


def anonymize_availability(doc):
    """把“available from the corresponding author upon reasonable request”
    改成不含作者指向的表述。

    坑：原 markdown 的硬换行在 docx 里变成**两个独立段落**，句子被从中间切开
    （第 150 段以 "...corresponding author" 结尾，第 151 段以 "upon reasonable
    request..." 开头）。所以必须同时处理跨段情形。
    """
    paras = doc.paragraphs
    hits = 0
    for i, p in enumerate(paras):
        flat = re.sub(r"\s+", " ", p.text).strip()
        # 情形 A：整句在同一段
        if (AVAIL_OLD_FLAT in flat or AVAIL_OLD in flat
                or AVAIL_OLD2_FLAT in flat or AVAIL_OLD2 in flat):
            new = (flat.replace(AVAIL_OLD_FLAT, AVAIL_NEW)
                       .replace(AVAIL_OLD2_FLAT, AVAIL_NEW)
                       .replace(AVAIL_OLD, AVAIL_NEW)
                       .replace(AVAIL_OLD2, AVAIL_NEW))
            for r in p.runs:
                r.text = ""
            if p.runs:
                p.runs[0].text = new
            else:
                p.add_run(new)
            hits += 1
            continue
        # 情形 B：句子被硬换行切断
        if flat.endswith("available from the corresponding author"):
            new = flat.replace(
                "are available from the corresponding author",
                "will be deposited in a public repository and released with a "
                "permanent DOI")
            for r in p.runs:
                r.text = ""
            if p.runs:
                p.runs[0].text = new
            else:
                p.add_run(new)
            if i + 1 < len(paras):
                q = paras[i + 1]
                qf = re.sub(r"\s+", " ", q.text).strip()
                qf = qf.replace(
                    "upon reasonable request, and will be deposited in a public "
                    "repository upon acceptance.",
                    "concurrently with publication.")
                for r in q.runs:
                    r.text = ""
                if q.runs:
                    q.runs[0].text = qf
                else:
                    q.add_run(qf)
            hits += 1
    return hits


def build_title_page(removed):
    doc = Document()
    st = doc.styles["Normal"]
    st.font.name = "Times New Roman"
    st.font.size = Pt(12)

    h = doc.add_paragraph()
    h.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = h.add_run(TITLE)
    r.bold = True
    r.font.size = Pt(16)

    doc.add_paragraph()
    doc.add_paragraph("Yongchao Zhao").bold = True
    doc.add_paragraph("School of Physics and Electronic Engineering, "
                      "Nanning Normal University, Nanning, Guangxi 530000, China")
    doc.add_paragraph()
    doc.add_paragraph("Corresponding author: Yongchao Zhao (1792552210@qq.com)")

    for head, body in [
        ("Funding",
         "This research received no specific grant from any funding agency in the public, "
         "commercial, or not-for-profit sectors."),
        ("Declaration of competing interests",
         "The author declares no competing financial interests or personal relationships "
         "that could have appeared to influence the work reported in this paper."),
        ("Acknowledgments", "None."),
        ("CRediT author statement",
         "Yongchao Zhao: Conceptualization, Data curation, Formal analysis, "
         "Methodology, Software, Validation, Visualization, Writing – original draft, "
         "Writing – review & editing."),
        ("Declaration of generative AI use",
         "The author used DeepSeek, a large language model, in the preparation of this "
         "manuscript, for language editing, for assistance in writing and debugging "
         "Python code, and for locating and verifying bibliographic metadata such as "
         "DOIs. The tool was not used to generate research ideas, design the "
         "methodology, perform the statistical analysis, or interpret results; all such "
         "decisions and all empirical results are the author's own. Every output was "
         "independently verified against the underlying data or source. The author "
         "takes full responsibility for the content of the publication."),
    ]:
        doc.add_paragraph()
        doc.add_paragraph(head).runs[0].bold = True
        doc.add_paragraph(body)

    doc.save(TITLE_OUT)
    return len(removed)


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    doc = Document(SRC)

    removed = strip_front(doc)
    hits = anonymize_availability(doc)
    doc.save(MANU_OUT)

    build_title_page(removed)

    print(f"从正文删除首部段落 {len(removed)} 个：")
    for t in removed:
        print("   -", t[:80])
    print(f"数据可得性匿名化替换：{hits} 处")
    print(f"[written] {TITLE_OUT}")
    print(f"[written] {MANU_OUT}")

    # 自检：匿名版里不该再出现可识别信息
    chk = Document(MANU_OUT)
    full = "\n".join(p.text for p in chk.paragraphs)
    bad = [k for k in ("Yongchao", "Nanning", "1792552210", "School of Physics",
                       "corresponding author") if k in full]
    print("匿名自检：", "PASS" if not bad else f"FAIL 残留 {bad}")
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main())
