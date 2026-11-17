# -*- coding: utf-8 -*-
"""build_paper.py —— paper.md → paper.docx（期刊投稿格式）

格式（按 SCI 三四区常见要求）：
  · A4，页边距 2.54cm（上下）、2.5cm（左右）
  · 正文 Times New Roman 10.5pt，行距 1.15，首行缩进 0（英文期刊多用空行分段）
  · 标题 Times New Roman 加粗，层级：一级 14pt / 二级 12pt / 三级 10.5pt 斜体
  · 表格：三线表（顶线、表头下线、底线），表题在表上方，Times New Roman 9pt
  · 图：宽 15cm 居中，图题在图下方
  · 页码在页脚居中
  · 参考文献悬挂缩进

不依赖 pandoc；直接 python-docx 拼装，保证可控。
"""
import os, re
import fitz
from docx import Document
from docx.shared import Pt, Cm, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_LINE_SPACING
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.section import WD_SECTION
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

ROOT = os.path.dirname(os.path.abspath(__file__))
PAPER = os.path.join(ROOT, "paper")
FIGS = os.path.join(ROOT, "figs")
OUT = os.path.join(PAPER, "paper.docx")

SERIF = "Times New Roman"
CJK = "SimSun"          # 宋体
CJK_TITLE = "SimHei"    # 黑体（标题）


def set_run_font(run, size=10.5, bold=False, italic=False, serif=SERIF, cjk=CJK):
    run.font.name = serif
    run.font.size = Pt(size)
    run.bold = bold
    run.italic = italic
    run.font.color.rgb = RGBColor(0, 0, 0)
    rPr = run._element.get_or_add_rPr()
    rFonts = rPr.find(qn("w:rFonts"))
    if rFonts is None:
        rFonts = OxmlElement("w:rFonts")
        rPr.append(rFonts)
    rFonts.set(qn("w:ascii"), serif)
    rFonts.set(qn("w:hAnsi"), serif)
    rFonts.set(qn("w:eastAsia"), cjk)
    rFonts.set(qn("w:cs"), serif)


def para(doc, text="", size=10.5, bold=False, italic=False, align=WD_ALIGN_PARAGRAPH.JUSTIFY,
         space_before=0, space_after=6, line=1.15, indent_first=0, keep_next=False,
         cjk=CJK, serif=SERIF):
    p = doc.add_paragraph()
    p.alignment = align
    pf = p.paragraph_format
    pf.space_before = Pt(space_before)
    pf.space_after = Pt(space_after)
    pf.line_spacing = line
    pf.first_line_indent = Cm(indent_first)
    pf.keep_with_next = keep_next
    if text:
        add_rich(p, text, size=size, bold=bold, italic=italic, cjk=cjk, serif=serif)
    return p


META_BOLD_OK = {"Authors", "Corresponding author", "Funding", "Keywords"}


def add_rich(p, text, size=10.5, bold=False, italic=False, cjk=CJK, serif=SERIF):
    """支持 *斜体*、**加粗**、`代码` 与 ^上标（^ 后用花括号或单个字符）

    加粗白名单：学术论文正文不应出现加粗（Springer 要求正文常规字重，
    只有标题、表头、首页元数据字段名可以加粗）。遇到白名单外的加粗，
    自动降为斜体并告警 —— 防止 markdown 里的强调记号混进正文。
    """
    tokens = re.split(r"(\*\*.+?\*\*|\*.+?\*|`.+?`|\^\{[^}]*\}|\^\S)", text)
    for t in tokens:
        if not t:
            continue
        if t.startswith("**") and t.endswith("**"):
            inner = t[2:-2]
            if inner.strip().rstrip(".:") in META_BOLD_OK:
                r = p.add_run(inner)
                set_run_font(r, size, bold=True, italic=italic, cjk=cjk, serif=serif)
            else:
                # 正文里不允许加粗 -> 降为斜体
                print(f"  [bold->italic] {inner[:60]!r}")
                r = p.add_run(inner)
                set_run_font(r, size, bold=bold, italic=True, cjk=cjk, serif=serif)
        elif t.startswith("*") and t.endswith("*") and len(t) > 2:
            r = p.add_run(t[1:-1])
            set_run_font(r, size, bold=bold, italic=True, cjk=cjk, serif=serif)
        elif t.startswith("`") and t.endswith("`"):
            r = p.add_run(t[1:-1])
            set_run_font(r, size - 0.5, bold=bold, italic=italic, cjk=cjk, serif="Consolas")
        elif t.startswith("^"):
            # 上标：^{...} 或 ^x
            body = t[2:-1] if t.startswith("^{") else t[1:]
            r = p.add_run(body)
            set_run_font(r, size, bold=bold, italic=italic, cjk=cjk, serif=serif)
            r.font.superscript = True
        else:
            r = p.add_run(t)
            set_run_font(r, size, bold=bold, italic=italic, cjk=cjk, serif=serif)


def set_cell_border(cell, **kwargs):
    tcPr = cell._tc.get_or_add_tcPr()
    borders = tcPr.find(qn("w:tcBorders"))
    if borders is None:
        borders = OxmlElement("w:tcBorders")
        tcPr.append(borders)
    for edge in ("top", "bottom", "left", "right"):
        if edge in kwargs:
            spec = kwargs[edge]
            el = borders.find(qn(f"w:{edge}"))
            if el is None:
                el = OxmlElement(f"w:{edge}")
                borders.append(el)
            for k, v in spec.items():
                el.set(qn(f"w:{k}"), str(v))


def clear_table_borders(table):
    tbl = table._tbl
    tblPr = tbl.tblPr
    borders = OxmlElement("w:tblBorders")
    for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
        el = OxmlElement(f"w:{edge}")
        el.set(qn("w:val"), "none")
        el.set(qn("w:sz"), "0")
        borders.append(el)
    tblPr.append(borders)


def three_line_table(doc, rows, header, caption, width_cm=15.5, font_size=9):
    """三线表：顶线 1.5pt / 表头下 0.75pt / 底线 1.5pt。整表不可拆页。"""
    # 表题（表上方）
    cap = doc.add_paragraph()
    cap.alignment = WD_ALIGN_PARAGRAPH.LEFT
    cap.paragraph_format.space_before = Pt(8)
    cap.paragraph_format.space_after = Pt(3)
    cap.paragraph_format.keep_with_next = True
    add_rich(cap, caption, size=9, bold=False)
    # 表头加粗部分
    for r in cap.runs:
        r.bold = False

    ncol = len(header)
    t = doc.add_table(rows=1 + len(rows), cols=ncol)
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    t.autofit = True
    clear_table_borders(t)

    # —— 整表不可跨页：给每一行的 trPr 加 cantSplit，
    #    并给除最后一行外的所有段落加 keep_with_next，
    #    这样 Word/LibreOffice 会整块移到下一页 ——
    for row in t.rows:
        trPr = row._tr.get_or_add_trPr()
        cant = OxmlElement("w:cantSplit")
        trPr.append(cant)

    # 表头
    for j, h in enumerate(header):
        c = t.cell(0, j)
        c.text = ""
        p = c.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_before = Pt(2)
        p.paragraph_format.space_after = Pt(2)
        p.paragraph_format.line_spacing = 1.0
        add_rich(p, h, size=font_size, bold=True)
        set_cell_border(c, top=dict(val="single", sz=12, color="000000"),
                        bottom=dict(val="single", sz=6, color="000000"))
    # 数据
    # 数字列右对齐（学术三线表惯例，便于比较位数与小数点），文字列居中
    def _is_num(s):
        s = s.strip().replace("**", "").strip()
        if not s:
            return False
        if s.lower() in ("n/a", "na", "-"):
            return True
        # 去掉常见单位后缀后再试
        for suf in ("pp", "%", "bps", "bp", "x"):
            if s.lower().endswith(suf):
                s = s[: -len(suf)].strip()
                break
        s = s.replace("%", "").replace(",", "").strip()
        try:
            float(s)
            return True
        except ValueError:
            return False

    num_col = []
    for j in range(ncol):
        vals = [r[j] for r in rows if j < len(r)]
        num_col.append(sum(1 for v in vals if _is_num(v)) >= max(1, len(vals) * 0.7))

    for i, row in enumerate(rows, start=1):
        for j, v in enumerate(row):
            c = t.cell(i, j)
            c.text = ""
            p = c.paragraphs[0]
            if j > 0 and num_col[j]:
                p.alignment = WD_ALIGN_PARAGRAPH.RIGHT
            elif j > 0:
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            else:
                p.alignment = WD_ALIGN_PARAGRAPH.LEFT
            p.paragraph_format.space_before = Pt(1.5)
            p.paragraph_format.space_after = Pt(1.5)
            p.paragraph_format.line_spacing = 1.0
            bold = v.startswith("**") and v.endswith("**")
            # 表格里禁用 em dash（SimSun 的 U+2014 在 LibreOffice 导出时会变成乱码）
            v = v.replace("\u2014", "n/a")
            add_rich(p, v, size=font_size, bold=bold)
        if i == len(rows):
            for j in range(ncol):
                set_cell_border(t.cell(i, j), bottom=dict(val="single", sz=12, color="000000"))

    # 行间 keep_with_next（除了最后一行），保证整表一起换页
    for i, row in enumerate(t.rows):
        if i == len(t.rows) - 1:
            continue
        for cell in row.cells:
            for p in cell.paragraphs:
                p.paragraph_format.keep_with_next = True

    # 表格后空行
    sp = doc.add_paragraph()
    sp.paragraph_format.space_after = Pt(2)
    sp.paragraph_format.space_before = Pt(0)
    for r in sp.runs:
        set_run_font(r, 6)
    return t


def add_page_number_footer(section):
    footer = section.footer
    p = footer.paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run()
    fld1 = OxmlElement("w:fldChar")
    fld1.set(qn("w:fldCharType"), "begin")
    instr = OxmlElement("w:instrText")
    instr.set(qn("xml:space"), "preserve")
    instr.text = " PAGE "
    fld2 = OxmlElement("w:fldChar")
    fld2.set(qn("w:fldCharType"), "end")
    run._r.append(fld1)
    run._r.append(instr)
    run._r.append(fld2)
    set_run_font(run, 9)


def parse_md(md_text):
    """切成块：('h1'|'h2'|'h3'|'p'|'table'|'fig'|'caption'|'hr', payload)"""
    blocks = []
    lines = md_text.splitlines()
    i = 0
    while i < len(lines):
        l = lines[i].rstrip()
        if not l.strip():
            i += 1
            continue
        if l.strip() == "---":
            blocks.append(("hr", None)); i += 1; continue
        if l.startswith("### "):
            blocks.append(("h3", l[4:].strip())); i += 1; continue
        if l.startswith("## "):
            blocks.append(("h2", l[3:].strip())); i += 1; continue
        if l.startswith("# "):
            blocks.append(("h1", l[2:].strip())); i += 1; continue
        # 表格
        if l.strip().startswith("|") and i + 1 < len(lines) and set(lines[i + 1].replace("|", "").strip()) <= set("-: "):
            header = [c.strip() for c in l.strip().strip("|").split("|")]
            rows = []
            j = i + 2
            while j < len(lines) and lines[j].strip().startswith("|"):
                rows.append([c.strip() for c in lines[j].strip().strip("|").split("|")])
                j += 1
            # 上一个是 Table 题注？
            blocks.append(("table", (header, rows)))
            i = j
            continue
        # 数学公式（(1) 编号行）
        if re.match(r"^\s{2}r_\{i,t\+5\}", l):
            blocks.append(("eq", l.strip())); i += 1; continue
        blocks.append(("p", l.strip()))
        i += 1
    return blocks


def main():
    """构建稿件。

    环境变量 PAPER_NOFIGS=1 ：跳过末尾的 Figures 章节（图与图注都不嵌入）。
    Quantitative Finance 要求 "Figures should be saved separately from the text"，
    图注另交 `submission/qf_figs/Figure_Captions.txt`，所以 QF 版正文用此开关。
    """
    NOFIGS = os.environ.get("PAPER_NOFIGS") == "1"
    md = open(os.path.join(PAPER, "paper.md"), encoding="utf-8").read()
    blocks = parse_md(md)

    doc = Document()
    sec = doc.sections[0]
    sec.page_width = Cm(21.0)
    sec.page_height = Cm(29.7)
    sec.top_margin = Cm(2.54)
    sec.bottom_margin = Cm(2.54)
    sec.left_margin = Cm(2.5)
    sec.right_margin = Cm(2.5)
    add_page_number_footer(sec)

    # 默认样式
    st = doc.styles["Normal"]
    st.font.name = SERIF
    st.font.size = Pt(10.5)
    st.element.rPr.rFonts.set(qn("w:eastAsia"), CJK)

    pending_table_caption = None
    in_refs = False
    fig_idx = 0
    table_count = 0
    fig_files = ["fig1_factor_ic.png", "fig2_overlap_t.png", "fig3_nav.png",
                 "fig4_neutral.png", "fig5_cost.png", "fig6_excess_q.png",
                 "fig7_sign.png", "fig8_roadmap.png"]

    for kind, payload in blocks:
        if kind == "h2" and payload.strip() in ("Figure captions",):
            # 图注块仅用于取文本（图下方的题注），不作为正文输出
            continue
        if kind == "h1":
            in_refs = False
            para(doc, payload, size=15, bold=True, align=WD_ALIGN_PARAGRAPH.CENTER,
                 space_before=0, space_after=12, line=1.2, cjk=CJK_TITLE)
        elif kind == "h2":
            # 进入参考文献区：其后段落改用 9pt / 单倍行距 / 3pt 段后
            in_refs = payload.strip().lower().startswith("reference")
            para(doc, payload, size=12, bold=True, align=WD_ALIGN_PARAGRAPH.LEFT,
                 space_before=12, space_after=5, line=1.15, keep_next=True, cjk=CJK_TITLE)
        elif kind == "h3":
            para(doc, payload, size=10.5, bold=True, italic=True, align=WD_ALIGN_PARAGRAPH.LEFT,
                 space_before=8, space_after=3, line=1.15, keep_next=True, cjk=CJK_TITLE)
        elif kind == "hr":
            continue
        elif kind == "eq":
            p = para(doc, "", align=WD_ALIGN_PARAGRAPH.CENTER, space_before=4, space_after=4)
            add_rich(p, payload, size=10)
        elif kind == "p":
            txt = payload
            # 图注行只用于图下方题注，不当作正文输出
            if re.match(r"^\*\*Fig\.\s*\d+\*\*", txt):
                continue
            # Table 题注暂存
            if txt.startswith("**Table ") or txt.startswith("Table "):
                # 整段去掉 markdown 加粗标记，保留后续表注文字
                pending_table_caption = txt.replace("**", "").strip()
                continue
            if txt.startswith("**Keywords**"):
                p = para(doc, "", space_before=6, space_after=10)
                add_rich(p, txt, size=10)
                continue
            # Abstract 正文：缩进块
            if "^" in txt:
                p = para(doc, "", size=10.5, space_after=6, line=1.15)
                add_rich(p, txt, size=10.5)
            else:
                if in_refs:
                    # 参考文献：9pt、单倍行距、3pt 段后（Springer 惯例，且省页）
                    para(doc, txt, size=9, space_after=3, line=1.0,
                         align=WD_ALIGN_PARAGRAPH.LEFT)
                else:
                    para(doc, txt, size=10.5, space_after=6, line=1.15)
        elif kind == "table":
            header, rows = payload
            table_count += 1
            cap = pending_table_caption or f"Table {table_count}"
            # 去掉题注开头的 "Table N." 重复，只保留描述
            three_line_table(doc, rows, header, cap)
            pending_table_caption = None

    # 图：按图号顺序流式排布，放不下才换页（不强制一图一页）
    if not NOFIGS:
        _place_figures(doc, md, fig_files)

    doc.save(OUT)
    print(f"[saved] {OUT}  ({os.path.getsize(OUT)/1024:.0f} KB, {table_count} tables, "
          f"{0 if NOFIGS else len(fig_files)} figs)")


def _place_figures(doc, md, fig_files):
    """把 8 张图 + 图注排到文末。NOFIGS 模式下不调用。"""
    doc.add_page_break()
    para(doc, "Figures", size=12, bold=True, align=WD_ALIGN_PARAGRAPH.LEFT,
         space_before=0, space_after=10, cjk=CJK_TITLE)

    # A4 841.9 - 上下边距 2.54cm*2(=144) - 页脚 ~12 - 余量 6 ≈ 690pt
    USABLE_PT = 690.0
    # 同页余量不足时，允许把图宽从 15cm 收到 13.2cm 以塞进同一页（最小可读宽度）
    MIN_W_CM, FULL_W_CM = 13.2, 15.0
    # “Figures” 标题占的高度（12pt 字 + space_after 10）
    used_pt = 28.0
    first_on_page = True
    for k, f in enumerate(fig_files, start=1):
        p = os.path.join(FIGS, f)
        if not os.path.exists(p):
            print(f"  [warn] missing {f}")
            continue
        m = re.search(rf"\*\*Fig\. {k}\*\* (.+?)(?=\n\n|\Z)", md, re.S)
        cap = f"Fig. {k}  " + (" ".join(m.group(1).split()) if m else "")

        pix = fitz.Pixmap(p)
        # 图注高度：按**实测**每行约 122 字符、行高 10.3pt 估算
        # （原用 105 字符/行 + 11.5pt + 额外 14pt，严重高估：
        #   Fig.8 实降 4 行 41pt，旧公式估成 5 行 71.5pt）
        cap_lines = max(1, -(-len(cap) // 105))
        cap_h_pt = cap_lines * 11.5 + 14
        extra = 12   # 图片段落 space_before 6 + space_after 2 + 余量

        def h_at(w_cm):
            """该图在 w_cm 宽时的块高（图片+图注+间距）"""
            return w_cm / 2.54 * 72.0 * (pix.height / pix.width) + cap_h_pt + extra

        block_h = h_at(FULL_W_CM)

        # 1) 整块放不下 → 尝试缩到最小可读宽度
        if used_pt > 0 and used_pt + block_h > USABLE_PT:
            wmin = MIN_W_CM
            # 先算「刚好塞进去需要多宽」
            need = None
            if used_pt + h_at(wmin) <= USABLE_PT:
                w_probe = FULL_W_CM
                while w_probe > wmin and used_pt + h_at(w_probe) > USABLE_PT:
                    w_probe = round(w_probe - 0.1, 1)
                need = w_probe

            # 只在「缩小幅度 <= SHRINK_TOL」时才缩，否则宁可换页保图宽一致
            SHRINK_TOL = 0.2
            if need is not None and (FULL_W_CM - need) <= SHRINK_TOL:
                w_cm = need
                block_h = h_at(w_cm)
                print(f"  [fit] Fig{k} 缩到 {w_cm}cm 塞入本页")
            else:
                if not first_on_page:
                    doc.add_page_break()
                used_pt = 0.0
                first_on_page = True
                w_cm = FULL_W_CM
                block_h = h_at(w_cm)
                if need is not None:
                    print(f"  [fit] Fig{k} 需缩到 {need}cm 才同页 -> 改为另起一页，保持 {FULL_W_CM}cm")
        elif used_pt == 0:
            w_cm = FULL_W_CM
        else:
            w_cm = FULL_W_CM

        pic = doc.add_paragraph()
        pic.alignment = WD_ALIGN_PARAGRAPH.CENTER
        pic.paragraph_format.space_before = Pt(6)
        pic.paragraph_format.space_after = Pt(2)
        pic.paragraph_format.keep_with_next = True   # 图与图注同页
        pic.add_run().add_picture(p, width=Cm(w_cm))

        para(doc, cap, size=9, align=WD_ALIGN_PARAGRAPH.JUSTIFY,
             space_before=0, space_after=8, line=1.0)
        used_pt += block_h
        first_on_page = False


def _unused_placeholder():
    pass


if __name__ == "__main__":
    main()
