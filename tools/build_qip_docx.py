from __future__ import annotations

import re
import unicodedata
from pathlib import Path

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK, WD_LINE_SPACING
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Inches, Pt, RGBColor


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "submission_qip" / "QIP_LaTeX_Submission"
TEX = PACKAGE / "manuscript.tex"
BIB = PACKAGE / "references.bib"
OUTPUT = ROOT / "submission_qip" / "Hardware_Aware_Quantum_Transfer_Learning_QIP.docx"


def set_font(run, name: str = "Times New Roman", size: float = 10, bold=None, italic=None):
    run.font.name = name
    run._element.get_or_add_rPr().rFonts.set(qn("w:ascii"), name)
    run._element.get_or_add_rPr().rFonts.set(qn("w:hAnsi"), name)
    run.font.size = Pt(size)
    if bold is not None:
        run.bold = bold
    if italic is not None:
        run.italic = italic


def set_cell_shading(cell, fill: str):
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = tc_pr.find(qn("w:shd"))
    if shd is None:
        shd = OxmlElement("w:shd")
        tc_pr.append(shd)
    shd.set(qn("w:fill"), fill)


def set_cell_margins(cell, top=80, start=90, bottom=80, end=90):
    tc = cell._tc
    tc_pr = tc.get_or_add_tcPr()
    tc_mar = tc_pr.first_child_found_in("w:tcMar")
    if tc_mar is None:
        tc_mar = OxmlElement("w:tcMar")
        tc_pr.append(tc_mar)
    for margin, value in (("top", top), ("start", start), ("bottom", bottom), ("end", end)):
        node = tc_mar.find(qn(f"w:{margin}"))
        if node is None:
            node = OxmlElement(f"w:{margin}")
            tc_mar.append(node)
        node.set(qn("w:w"), str(value))
        node.set(qn("w:type"), "dxa")


def set_table_borders(table, color="B7B7B7", size="4"):
    tbl_pr = table._tbl.tblPr
    borders = tbl_pr.find(qn("w:tblBorders"))
    if borders is None:
        borders = OxmlElement("w:tblBorders")
        tbl_pr.append(borders)
    for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
        element = borders.find(qn(f"w:{edge}"))
        if element is None:
            element = OxmlElement(f"w:{edge}")
            borders.append(element)
        element.set(qn("w:val"), "single")
        element.set(qn("w:sz"), size)
        element.set(qn("w:color"), color)


def add_page_number(paragraph):
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = paragraph.add_run()
    fld_char1 = OxmlElement("w:fldChar")
    fld_char1.set(qn("w:fldCharType"), "begin")
    instr = OxmlElement("w:instrText")
    instr.set(qn("xml:space"), "preserve")
    instr.text = " PAGE "
    fld_char2 = OxmlElement("w:fldChar")
    fld_char2.set(qn("w:fldCharType"), "end")
    run._r.extend([fld_char1, instr, fld_char2])
    set_font(run, size=9)


def bib_entries(text: str):
    entries = {}
    for match in re.finditer(r"@(\w+)\{([^,]+),\s*(.*?)(?=\n\}\s*(?:\n|$))", text, re.S):
        kind, key, body = match.groups()
        fields = {}
        for field_match in re.finditer(r"^\s*(\w+)\s*=\s*\{(.*?)\}\s*,?\s*$", body, re.M):
            fields[field_match.group(1).lower()] = field_match.group(2)
        fields["kind"] = kind.lower()
        entries[key] = fields
    return entries


def plain_latex(text: str) -> str:
    # Resolve the compact accent forms commonly used in BibTeX before braces
    # and generic LaTeX commands are removed.
    accent_marks = {"'": "\u0301", '"': "\u0308", "`": "\u0300", "~": "\u0303", "^": "\u0302"}

    def accent_replace(match):
        mark, letter = match.groups()
        return unicodedata.normalize("NFC", letter + accent_marks[mark])

    text = re.sub(r"\{\\(['\"`~^])([A-Za-z])\}", accent_replace, text)
    text = re.sub(r"\\(['\"`~^])\{?([A-Za-z])\}?", accent_replace, text)
    replacements = {
        r"\%": "%",
        r"\pm": "±",
        r"\times": "×",
        r"\ldots": "…",
        r"\lVert": "||",
        r"\rVert": "||",
        r"\langle": "<",
        r"\rangle": ">",
        r"\lvert": "|",
        r"\rvert": "|",
        r"\boldsymbol": "",
        r"\operatorname": "",
        r"\text": "",
        r"\qquad": "   ",
        r"\quad": "  ",
        r"\dagger": "†",
        r"\otimes": "⊗",
        r"\prod": "∏",
        r"\sum": "Σ",
        r"\frac": "",
        r"\left": "",
        r"\right": "",
        r"\bar": "",
        r"\widehat": "",
        r"\sim": "~",
        r"\in": "∈",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    text = re.sub(r"\\(?:textbf|textit|mathrm|mathbf|mathbb|mathcal)\{([^{}]*)\}", r"\1", text)
    text = re.sub(r"\\url\{([^{}]*)\}", r"\1", text)
    text = re.sub(r"\\[a-zA-Z]+", "", text)
    text = text.replace("{", "").replace("}", "").replace("$", "")
    text = text.replace("--", "–").replace("~", " ")
    return re.sub(r"\s+", " ", text).strip()


def collect_order_and_labels(tex: str):
    cite_order = []
    for match in re.finditer(r"\\cite\{([^}]+)\}", tex):
        for key in match.group(1).split(","):
            if key not in cite_order:
                cite_order.append(key)
    labels = {}
    counters = {"fig": 0, "tab": 0, "eq": 0}
    for label in re.findall(r"\\label\{([^}]+)\}", tex):
        prefix = label.split(":", 1)[0]
        if prefix in counters:
            counters[prefix] += 1
            labels[label] = counters[prefix]
    return cite_order, labels


def inline_text(text: str, cite_numbers: dict[str, int], labels: dict[str, int]) -> str:
    def cite_replace(match):
        nums = [cite_numbers[key] for key in match.group(1).split(",")]
        return "[" + ", ".join(str(number) for number in nums) + "]"

    text = re.sub(r"\\cite\{([^}]+)\}", cite_replace, text)
    text = re.sub(r"\\ref\{([^}]+)\}", lambda m: str(labels.get(m.group(1), "?")), text)
    return plain_latex(text)


def add_body_paragraph(doc, text, cite_numbers, labels):
    clean = inline_text(text, cite_numbers, labels)
    if not clean:
        return
    p = doc.add_paragraph(style="Body Text")
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    p.paragraph_format.first_line_indent = Cm(0.6)
    p.paragraph_format.space_after = Pt(5)
    p.paragraph_format.line_spacing = 1.05
    run = p.add_run(clean)
    set_font(run, size=10)


def parse_table_block(block: str, cite_numbers, labels):
    caption_match = re.search(r"\\caption\{(.*?)\}\\label", block, re.S)
    caption = inline_text(caption_match.group(1), cite_numbers, labels) if caption_match else ""
    tabular_match = re.search(r"\\begin\{tabularx?\}.*?\n(.*?)\\end\{tabularx?\}", block, re.S)
    if not tabular_match:
        return caption, []
    body = tabular_match.group(1)
    body = re.sub(r"\\(?:toprule|midrule|bottomrule)", "\n", body)
    rows = []
    for raw in re.split(r"\\\\", body):
        raw = raw.strip()
        if not raw or "&" not in raw:
            continue
        cells = []
        for cell in raw.split("&"):
            cell = re.sub(r"\\multirow\{[^}]+\}\{[^}]+\}\{([^}]+)\}", r"\1", cell)
            cell = re.sub(r"\\textbf\{([^}]+)\}", r"\1", cell)
            cells.append(inline_text(cell, cite_numbers, labels))
        rows.append(cells)
    width = max((len(row) for row in rows), default=0)
    rows = [row + [""] * (width - len(row)) for row in rows]
    return caption, rows


def add_table(doc, caption: str, rows, number: int):
    if not rows:
        return
    p = doc.add_paragraph(style="Caption")
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.keep_with_next = True
    r = p.add_run(f"Table {number} {caption}")
    set_font(r, size=9, bold=True)
    table = doc.add_table(rows=len(rows), cols=len(rows[0]))
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = True
    set_table_borders(table)
    header_pr = table.rows[0]._tr.get_or_add_trPr()
    repeat = OxmlElement("w:tblHeader")
    repeat.set(qn("w:val"), "true")
    header_pr.append(repeat)
    for row_index, row in enumerate(rows):
        for column_index, value in enumerate(row):
            cell = table.cell(row_index, column_index)
            cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
            set_cell_margins(cell)
            if row_index == 0:
                set_cell_shading(cell, "D9EAF7")
            elif row_index % 2 == 0:
                set_cell_shading(cell, "F4F7FA")
            paragraph = cell.paragraphs[0]
            paragraph.alignment = WD_ALIGN_PARAGRAPH.LEFT if column_index == 0 else WD_ALIGN_PARAGRAPH.CENTER
            paragraph.paragraph_format.space_after = Pt(0)
            run = paragraph.add_run(value)
            set_font(run, size=8, bold=(row_index == 0))
    doc.add_paragraph().paragraph_format.space_after = Pt(1)


def format_reference(number: int, fields: dict) -> str:
    authors = plain_latex(fields.get("author", "")).replace(" and ", ", ")
    authors = authors.replace(", others", " et al.")
    title = plain_latex(fields.get("title", ""))
    year = fields.get("year", "")
    if fields.get("kind") == "inproceedings":
        venue = plain_latex(fields.get("booktitle", ""))
    elif fields.get("kind") == "techreport":
        venue = plain_latex(fields.get("institution", ""))
    else:
        venue = plain_latex(fields.get("journal", ""))
    details = []
    if fields.get("volume"):
        details.append(fields["volume"])
    if fields.get("number"):
        details.append(f"({fields['number']})")
    if fields.get("pages"):
        details.append(fields["pages"].replace("--", "–"))
    core = f"[{number}] {authors}. {title}. {venue}"
    if details:
        core += " " + ", ".join(details)
    if year:
        core += f" ({year})."
    doi = fields.get("doi")
    url = fields.get("url")
    if doi:
        core += f" https://doi.org/{doi}"
    elif url:
        core += f" {url}"
    return core


def build():
    tex = TEX.read_text(encoding="utf-8")
    bib = bib_entries(BIB.read_text(encoding="utf-8"))
    cite_order, labels = collect_order_and_labels(tex)
    cite_numbers = {key: index + 1 for index, key in enumerate(cite_order)}

    doc = Document()
    section = doc.sections[0]
    section.page_width = Cm(21.0)
    section.page_height = Cm(29.7)
    section.top_margin = Cm(2.2)
    section.bottom_margin = Cm(2.0)
    section.left_margin = Cm(2.2)
    section.right_margin = Cm(2.2)
    add_page_number(section.footer.paragraphs[0])

    styles = doc.styles
    normal = styles["Normal"]
    normal.font.name = "Times New Roman"
    normal._element.rPr.rFonts.set(qn("w:ascii"), "Times New Roman")
    normal._element.rPr.rFonts.set(qn("w:hAnsi"), "Times New Roman")
    normal.font.size = Pt(10)
    for name, size, before, after in (("Title", 16, 0, 10), ("Heading 1", 12, 12, 5), ("Heading 2", 10.5, 8, 4)):
        style = styles[name]
        style.font.name = "Times New Roman"
        style._element.rPr.rFonts.set(qn("w:ascii"), "Times New Roman")
        style._element.rPr.rFonts.set(qn("w:hAnsi"), "Times New Roman")
        style.font.size = Pt(size)
        style.font.bold = True
        style.font.color.rgb = RGBColor(0, 0, 0)
        style.paragraph_format.space_before = Pt(before)
        style.paragraph_format.space_after = Pt(after)
        style.paragraph_format.keep_with_next = True
    body_style = styles["Body Text"]
    body_style.font.name = "Times New Roman"
    body_style._element.rPr.rFonts.set(qn("w:ascii"), "Times New Roman")
    body_style._element.rPr.rFonts.set(qn("w:hAnsi"), "Times New Roman")
    body_style.font.size = Pt(10)

    title = doc.add_paragraph(style="Title")
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    title.add_run("Hardware-aware simulation of quantum transfer learning across three visual domains")

    authors = doc.add_paragraph()
    authors.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = authors.add_run("Fajri Yanuar Shiddiq Juanda¹*  ·  Kiki Ahmad Baihaqi¹")
    set_font(r, size=10, bold=True)
    aff = doc.add_paragraph()
    aff.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = aff.add_run("¹ Department of Informatics Engineering, Faculty of Computer Science, Universitas Buana Perjuangan Karawang, Karawang, Indonesia")
    set_font(r, size=9)
    corr = doc.add_paragraph()
    corr.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = corr.add_run("* Corresponding author: fajriyanuar1@gmail.com")
    set_font(r, size=9, italic=True)
    orcid = doc.add_paragraph()
    orcid.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = orcid.add_run("ORCID: F.Y.S.J. 0009-0007-8894-0265; K.A.B. 0000-0002-2430-9235")
    set_font(r, size=8.5)

    abstract_match = re.search(r"\\abstract\{(.*?)\}\s*\n", tex, re.S)
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    r = p.add_run("Abstract  ")
    set_font(r, size=9.5, bold=True)
    r = p.add_run(plain_latex(abstract_match.group(1)))
    set_font(r, size=9.5)
    keywords_match = re.search(r"\\keywords\{(.*?)\}", tex, re.S)
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    r = p.add_run("Keywords  ")
    set_font(r, size=9, bold=True)
    r = p.add_run(plain_latex(keywords_match.group(1)))
    set_font(r, size=9, italic=True)

    content = tex[tex.index("\\section{Introduction}"):tex.index("\\backmatter")]
    position = 0
    buffer = []
    section_counter = 0
    subsection_counter = 0
    figure_counter = 0
    table_counter = 0
    equation_counter = 0

    def flush_buffer():
        nonlocal buffer
        if buffer:
            add_body_paragraph(doc, " ".join(buffer), cite_numbers, labels)
            buffer = []

    token_re = re.compile(r"\\section\{.*?\}|\\subsection\{.*?\}|\\begin\{figure\*?\}.*?\\end\{figure\*?\}|\\begin\{table\*?\}.*?\\end\{table\*?\}|\\begin\{equation\}.*?\\end\{equation\}", re.S)
    for match in token_re.finditer(content):
        before = content[position:match.start()]
        for paragraph in re.split(r"\n\s*\n", before):
            if paragraph.strip():
                buffer.append(paragraph.strip())
                flush_buffer()
        flush_buffer()
        block = match.group(0)
        if block.startswith("\\section"):
            section_counter += 1
            subsection_counter = 0
            heading = re.search(r"\{(.*?)\}", block).group(1)
            doc.add_heading(f"{section_counter} {plain_latex(heading)}", level=1)
        elif block.startswith("\\subsection"):
            subsection_counter += 1
            heading = re.search(r"\{(.*?)\}", block).group(1)
            doc.add_heading(f"{section_counter}.{subsection_counter} {plain_latex(heading)}", level=2)
        elif block.startswith("\\begin{figure"):
            figure_counter += 1
            image_match = re.search(r"\\includegraphics\[[^]]*\]\{([^}]+)\}", block)
            caption_match = re.search(r"\\caption\{(.*?)\}\\label", block, re.S)
            if image_match:
                p = doc.add_paragraph()
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                p.paragraph_format.keep_with_next = True
                run = p.add_run()
                image_path = PACKAGE / image_match.group(1)
                run.add_picture(str(image_path), width=Inches(6.35))
            if caption_match:
                p = doc.add_paragraph(style="Caption")
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                r = p.add_run(f"Fig. {figure_counter} {inline_text(caption_match.group(1), cite_numbers, labels)}")
                set_font(r, size=9)
        elif block.startswith("\\begin{table"):
            table_counter += 1
            caption, rows = parse_table_block(block, cite_numbers, labels)
            add_table(doc, caption, rows, table_counter)
        else:
            equation_counter += 1
            label_match = re.search(r"\\label\{([^}]+)\}", block)
            equation_text = {
                "eq:scale": "xⱼ = π tanh(zⱼ),   j = 1, …, 6.",
                "eq:circuit": "U(x, θ) = ∏ₗ₌₁ᴸ E_G [⊗ᵢ₌₁ᑫ R_Z(θₗᵢ₃) R_Y(θₗᵢ₂) R_Z(θₗᵢ₁)] [∏ⱼ₌₁⁶ R_Y⁽ʲ mod q⁾(xⱼ)],",
                "eq:expectation": "hᵢ(x) = ⟨0| U†(x, θ) Zᵢ U(x, θ) |0⟩.",
                "eq:gradient": "G_b = ‖g_b‖₂,   V_b = (1/P) Σₖ₌₁ᴾ (g_bk − ḡ_b)²,",
                "eq:shots": "Kᵢ ~ Binomial(S, (1 + hᵢ)/2),   ĥᵢ = 2Kᵢ/S − 1.",
                "eq:drift": "D_p = (1/NC) Σₙ₌₁ᴺ Σ꜀₌₁ᶜ |p̂_nc − p_nc|,",
            }.get(label_match.group(1) if label_match else "")
            if equation_text is None:
                equation = re.sub(r"\\begin\{equation\}|\\end\{equation\}|\\label\{[^}]+\}", "", block, flags=re.S)
                equation_text = plain_latex(equation)
            p = doc.add_paragraph()
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p.paragraph_format.space_before = Pt(4)
            p.paragraph_format.space_after = Pt(4)
            r = p.add_run(f"{equation_text}    ({equation_counter})")
            set_font(r, name="Cambria Math", size=10, italic=True)
        position = match.end()
    tail = content[position:]
    for paragraph in re.split(r"\n\s*\n", tail):
        if paragraph.strip():
            add_body_paragraph(doc, paragraph.strip(), cite_numbers, labels)

    backmatter = tex[tex.index("\\bmhead{Acknowledgments}"):tex.index("\\bibliography")]
    for match in re.finditer(r"\\bmhead\{([^}]+)\}\s*(.*?)(?=\\bmhead\{|\Z)", backmatter, re.S):
        doc.add_heading(plain_latex(match.group(1)), level=1)
        add_body_paragraph(doc, match.group(2).strip(), cite_numbers, labels)

    doc.add_heading("References", level=1)
    for number, key in enumerate(cite_order, start=1):
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        p.paragraph_format.left_indent = Cm(0.6)
        p.paragraph_format.first_line_indent = Cm(-0.6)
        p.paragraph_format.space_after = Pt(3)
        r = p.add_run(format_reference(number, bib[key]))
        set_font(r, size=8.5)

    props = doc.core_properties
    props.title = "Hardware-aware simulation of quantum transfer learning across three visual domains"
    props.subject = "Quantum Information Processing regular article"
    props.author = "Fajri Yanuar Shiddiq Juanda; Kiki Ahmad Baihaqi"
    props.keywords = "quantum machine learning; finite shots; quantum noise; transpilation; transfer learning"
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    doc.save(OUTPUT)
    print(OUTPUT)


if __name__ == "__main__":
    build()
