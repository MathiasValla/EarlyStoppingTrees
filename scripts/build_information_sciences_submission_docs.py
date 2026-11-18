"""Build the editable Information Sciences submission documents."""

from __future__ import annotations

from datetime import date
from pathlib import Path

from docx import Document
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor


ROOT = Path(__file__).resolve().parents[1]
OUTDIR = ROOT / "RESEARCH_ARTICLE" / "INFORMATION_SCIENCES_SUBMISSION"
TITLE = "Secretary-style split search for decision trees"


def _base_document() -> Document:
    document = Document()
    section = document.sections[0]
    section.top_margin = Inches(0.8)
    section.bottom_margin = Inches(0.8)
    section.left_margin = Inches(0.9)
    section.right_margin = Inches(0.9)

    styles = document.styles
    normal = styles["Normal"]
    normal.font.name = "Arial"
    normal.font.size = Pt(11)
    normal.paragraph_format.space_after = Pt(7)
    normal.paragraph_format.line_spacing = 1.08

    title = styles["Title"]
    title.font.name = "Arial"
    title.font.size = Pt(20)
    title.font.bold = True
    title.font.color.rgb = RGBColor(0, 0, 0)
    title.paragraph_format.space_after = Pt(14)
    title_properties = title.element.get_or_add_pPr()
    title_border = title_properties.find(qn("w:pBdr"))
    if title_border is not None:
        title_properties.remove(title_border)

    for name, size in (("Heading 1", 14), ("Heading 2", 12)):
        style = styles[name]
        style.font.name = "Arial"
        style.font.size = Pt(size)
        style.font.bold = True
        style.font.color.rgb = RGBColor(0, 0, 0)
        style.paragraph_format.space_before = Pt(10)
        style.paragraph_format.space_after = Pt(4)
    return document


def _set_core_properties(document: Document, subject: str) -> None:
    document.core_properties.title = subject
    document.core_properties.author = "Mathias Valla"
    document.core_properties.subject = f"Submission document for {TITLE}"


def _add_manuscript_identity(document: Document) -> None:
    paragraph = document.add_paragraph()
    paragraph.add_run("Manuscript title: ").bold = True
    paragraph.add_run(TITLE)
    paragraph = document.add_paragraph()
    paragraph.add_run("Author: ").bold = True
    paragraph.add_run("Mathias Valla")


def build_declarations() -> Path:
    document = _base_document()
    _set_core_properties(document, "Declarations")
    document.add_paragraph("Declarations", style="Title")
    _add_manuscript_identity(document)

    sections = [
        (
            "Funding",
            "This work was supported through the Research Chair DIALog under the aegis "
            "of the Risk Foundation, a joint initiative by Universite Paris-Dauphine "
            "PSL and CNP Assurances. The author was solely responsible for the study "
            "design, analysis, interpretation, manuscript preparation, and decision to "
            "submit.",
        ),
        (
            "Declaration of competing interest",
            "The author declares no known competing financial interests or personal "
            "relationships that could have appeared to influence the work reported in "
            "this paper.",
        ),
        (
            "Data and code availability",
            "The source datasets are publicly available through PMLB. The retained-entry "
            "inventory, complete per-seed benchmark outputs, inferential summaries, "
            "implemented methods, and scripts used to run the benchmark and generate "
            "every reported figure and table are available as checksummed archives at "
            "https://github.com/MathiasValla/EarlyStoppingTrees, together with "
            "clean-clone reproduction instructions.",
        ),
        (
            "CRediT authorship contribution statement",
            "Mathias Valla: Conceptualization, Methodology, Software, Validation, Formal "
            "analysis, Investigation, Data curation, Writing - original draft, Writing - "
            "review and editing, Visualization, and Project administration.",
        ),
        (
            "Declaration of generative AI and AI-assisted technologies",
            "During preparation of this revision, the author used OpenAI Codex to assist "
            "with language editing, source-code review, and preparation of analysis "
            "scripts. The author reviewed and edited the resulting material as needed "
            "and takes full responsibility for the content of the article.",
        ),
    ]
    for heading, text in sections:
        document.add_heading(heading, level=1)
        document.add_paragraph(text)

    path = OUTDIR / "declarations.docx"
    document.save(path)
    return path


def build_cover_letter() -> Path:
    document = _base_document()
    _set_core_properties(document, "Cover letter")
    document.add_paragraph("Cover letter", style="Title")
    document.add_paragraph(date.today().strftime("%B %d, %Y"))
    document.add_paragraph("Editors\nInformation Sciences")
    document.add_paragraph("Dear Editors,")
    document.add_paragraph(
        f'I submit the manuscript "{TITLE}" for consideration as a research article in '
        "Information Sciences. The paper studies whether adaptive stopping rules can "
        "reduce the repeated cost of exact split search in decision trees while "
        "retaining most of the predictive performance of exhaustive CART."
    )
    document.add_paragraph(
        "The main contribution is a controlled empirical characterization of this "
        "time-performance trade-off. The study covers 113 regression and 122 "
        "classification datasets, paired multi-dataset inference, exact random-feature "
        "controls, a focused bagging validation, operation counters, and complete "
        "per-seed reproducibility records. The results identify S_all(1/e) as a "
        "conservative observed Pareto option rather than claiming universal superiority."
    )
    document.add_paragraph(
        "This manuscript is original, is not under consideration elsewhere, and has "
        "been approved by its sole author. The competing-interest, funding, data and "
        "code availability, CRediT, and assisted-technology declarations are supplied "
        "with the submission."
    )
    document.add_paragraph("Thank you for considering this work.")
    document.add_paragraph(
        "Sincerely,\n"
        "Mathias Valla\n"
        "Research Chair DIALog, Institut Louis Bachelier\n"
        "mathias.valla@gmail.com"
    )
    path = OUTDIR / "cover_letter.docx"
    document.save(path)
    return path


def build_biography() -> Path:
    document = _base_document()
    _set_core_properties(document, "Author biography")
    document.add_paragraph("Author biography", style="Title")
    document.add_heading("Mathias Valla", level=1)
    document.add_paragraph(
        "Mathias Valla is a postdoctoral researcher with the DIALog Chair at Institut "
        "Louis Bachelier and a fully qualified actuary. He holds a PhD and studies "
        "statistical learning methods for insurance, with an emphasis on decision "
        "trees, longitudinal data, climate risk, and computationally efficient "
        "predictive modeling."
    )
    path = OUTDIR / "author_biography.docx"
    document.save(path)
    return path


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    for path in (build_declarations(), build_cover_letter(), build_biography()):
        print(path)


if __name__ == "__main__":
    main()
