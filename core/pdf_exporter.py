"""Exportación a PDF para Mi Fábrica de Cuentos."""

from __future__ import annotations

from typing import List
from fpdf import FPDF


def clean_pdf_text(text: str) -> str:
    """Evita caracteres que las fuentes estándar del PDF no soportan bien."""
    replacements = {
        "“": '"',
        "”": '"',
        "‘": "'",
        "’": "'",
        "—": "-",
        "–": "-",
        "…": "...",
        "·": "-",
        "🏭": "",
        "✨": "",
        "📚": "",
        "🪄": "",
        "🎨": "",
        "✅": "",
        "💡": "",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


class StoryPDF(FPDF):
    def header(self):
        self.set_font("Helvetica", "", 9)
        self.set_text_color(120, 120, 120)
        self.cell(0, 8, "Mi Fábrica de Cuentos", align="R", new_x="LMARGIN", new_y="NEXT")
        self.ln(2)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "", 8)
        self.set_text_color(140, 140, 140)
        self.cell(0, 10, f"Página {self.page_no()}", align="C")


def story_to_pdf_bytes(
    title: str,
    author: str,
    level_label: str,
    story_text: str,
    selection_lines: List[str],
) -> bytes:
    """Genera un PDF en memoria con el cuento."""
    pdf = StoryPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    title = clean_pdf_text(title.strip() or "Mi cuento")
    author = clean_pdf_text(author.strip() or "Autor/a sin indicar")
    level_label = clean_pdf_text(level_label)
    story_text = clean_pdf_text(story_text)
    selection_lines = [clean_pdf_text(line) for line in selection_lines]

    pdf.set_text_color(40, 40, 40)
    pdf.set_font("Helvetica", "B", 20)
    pdf.multi_cell(0, 10, title, align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(3)

    pdf.set_font("Helvetica", "", 12)
    pdf.multi_cell(0, 7, f"Autor/a: {author}", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.multi_cell(0, 7, f"Nivel: {level_label}", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(6)

    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 8, "Elementos del cuento", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    for line in selection_lines:
        pdf.multi_cell(0, 6, f"- {line}", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(5)

    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 8, "Cuento", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 12)
    for paragraph in story_text.split("\n\n"):
        if paragraph.strip():
            pdf.multi_cell(0, 7, paragraph.strip(), new_x="LMARGIN", new_y="NEXT")
            pdf.ln(2)

    raw = pdf.output(dest="S")
    if isinstance(raw, str):
        return raw.encode("latin-1", errors="replace")
    return bytes(raw)
