# /// script
# requires-python = ">=3.12"
# dependencies = ["reportlab==4.4.10"]
# ///
"""Build the math-eval software documentation report."""

import os
from pathlib import Path

# ReportLab reads this during import; keep PDF metadata reproducible and dated.
os.environ["SOURCE_DATE_EPOCH"] = "1788134400"

from reportlab import rl_config
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import KeepTogether, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "math-eval-v0.1.0.pdf"
TITLE = "math-eval: Reproducible Mathematical Reasoning Generation and Evaluation"
REPORT_DATE = "31 August 2026"

rl_config.invariant = True


def page_number(canvas, document):
    canvas.saveState()
    canvas.setStrokeColor(colors.HexColor("#d9e0e7"))
    canvas.line(document.leftMargin, 1.5 * cm, A4[0] - document.rightMargin, 1.5 * cm)
    canvas.setFillColor(colors.HexColor("#5d6875"))
    canvas.setFont("Helvetica", 8.5)
    canvas.drawString(document.leftMargin, 1.05 * cm, "Software documentation report")
    canvas.drawRightString(A4[0] - document.rightMargin, 1.05 * cm, str(document.page))
    canvas.restoreState()


def build():
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="TitleReport", parent=styles["Title"], fontName="Helvetica-Bold", fontSize=24, leading=28, alignment=TA_CENTER, textColor=colors.HexColor("#17212b"), spaceAfter=12))
    styles.add(ParagraphStyle(name="Author", parent=styles["Normal"], fontName="Helvetica", fontSize=16, leading=20, alignment=TA_CENTER, textColor=colors.HexColor("#283746"), spaceAfter=3))
    styles.add(ParagraphStyle(name="Meta", parent=styles["Normal"], fontName="Helvetica", fontSize=9.5, leading=14, alignment=TA_CENTER, textColor=colors.HexColor("#52606d"), spaceAfter=18))
    styles.add(ParagraphStyle(name="Abstract", parent=styles["Normal"], fontName="Times-Roman", fontSize=9.8, leading=14.2, spaceAfter=11))
    styles.add(ParagraphStyle(name="BodyReport", parent=styles["Normal"], fontName="Times-Roman", fontSize=9.8, leading=14.2, spaceAfter=8))
    styles.add(ParagraphStyle(name="HeadingReport", parent=styles["Heading2"], fontName="Helvetica-Bold", fontSize=13, leading=17, textColor=colors.HexColor("#0b4f8a"), spaceBefore=16, spaceAfter=7))
    styles.add(ParagraphStyle(name="Ref", parent=styles["Normal"], fontName="Times-Roman", fontSize=8.5, leading=11.2, leftIndent=14, firstLineIndent=-14, spaceAfter=3))
    doc = SimpleDocTemplate(str(OUTPUT), pagesize=A4, leftMargin=2.15 * cm, rightMargin=2.15 * cm, topMargin=2.15 * cm, bottomMargin=2.25 * cm, title=TITLE, author="Xinmu Ge", subject="Software documentation report", invariant=1)
    p, h, a, ref = (styles["BodyReport"], styles["HeadingReport"], styles["Abstract"], styles["Ref"])
    story = [
        Paragraph(TITLE, styles["TitleReport"]),
        Paragraph("Xinmu Ge", styles["Author"]),
        Paragraph(f"Shanghai Innovation Institute · Shanghai Jiao Tong University<br/>g3ra1d@sjtu.edu.cn<br/>{REPORT_DATE} · Software documentation report", styles["Meta"]),
        Paragraph("<b>Abstract.</b> math-eval is a software workflow for reproducible evaluation of mathematical reasoning systems. It accepts canonical JSONL records, generates model outputs through vLLM or an OpenAI-compatible API, preserves raw sample artifacts, and replays parsing and scoring independently. This separation lets an evaluation change its parser or metrics without repeating model inference. The implementation records configuration, prompt, environment, and artifact hashes; supports resumable runs and deterministic data partitions; and validates shard merges before replay. Its default math-v5.2-dual contract reports boxed-only strict results as formal scores and retains whole-response soft results for diagnosis when no complete box is present. Mathematical equivalence is evaluated with Math-Verify 0.9.0. Canonical datasets are maintained separately by math-vault, preserving their provenance and licensing records.", a),
        Paragraph("1. Scope and workflow", h),
        Paragraph("math-eval evaluates generated mathematical reasoning without coupling an evaluation result to a single model call. A canonical JSONL input contains at least an identifier, a problem, and a string answer. The generation stage uses either vLLM or an OpenAI-compatible API and writes sample-level raw JSONL artifacts. The replay stage then reads frozen artifacts, applies a parser and mathematical verifier, and writes parsed verdicts and metrics.", p),
        Paragraph("<b>canonical JSONL → generation → raw artifacts → replay parser/scoring → parsed results and metrics</b>", p),
        Paragraph("This arrangement is deliberate: modifying parser behavior or recomputing metrics does not re-call a model. Raw and parsed artifacts remain separate, so the source of a reported result can be replayed from the saved generation output.", p),
        Paragraph("2. Reproducible execution", h),
        Paragraph("A run records configuration and prompt snapshots, hashes, and environment state in its manifest. The storage layer writes raw data in shards and supports resuming an interrupted run with the same configuration and run identifier. When a process is interrupted or fails, the resume path can repair a damaged final in-progress JSONL line and regenerate only samples not yet committed to storage.", p),
        Paragraph("Independent workers may run deterministic partitions of one dataset. The partition count identifies workers, while raw shard size controls only output-file boundaries. A later merge creates one ordinary run directory. Before accepting a merge, the tool checks for missing, duplicate, incomplete, or hash-inconsistent partitions, and it rejects parts whose code revision or Python/core-package versions do not match. The manifest records checkpoint paths but does not fingerprint model-weight contents; cross-machine users must therefore use the same clean revision and ensure equivalent weights at the recorded path.", p),
        KeepTogether([
            Paragraph("3. Parsing and scoring contract", h),
            Paragraph("The default parser identifier is math-v5.2-dual. Its strict result is the formal score: it selects the final complete \\boxed{} answer. If a response has no complete box, strict evaluation records no candidate. The soft result may instead submit a nonempty full response to the verifier, but only as a diagnostic signal. A response truncated after a complete box still uses that completed box; an incomplete trailing box does not replace an earlier complete one.", p),
        ]),
        Paragraph("math-eval records correct, incorrect, no-candidate, parse-error, and verification-error states separately. Parse and verification failures count as failures but are not conflated with mathematical disagreement. Prediction normalization has a five-second hard timeout; a timeout is a parse error, so a pathological symbolic expression cannot stall an entire replay.", p),
        Paragraph("For mathematical equivalence, the current pipeline uses Math-Verify 0.9.0 with shared extraction configuration for gold answers and predictions. Metrics are recomputed from parsed verdicts, including accuracy, pass@k, and failure rates. The implementation also retains earlier parser identifiers when an older evaluation contract must be reproduced.", p),
        KeepTogether([
            Paragraph("4. Data boundary and documented use", h),
            Paragraph("math-eval does not define a new source dataset. math-vault maintains curated, traceable snapshots of public mathematical reasoning datasets and records their provenance and licensing. Its canonical JSONL files can be used directly as math-eval inputs. This boundary keeps data conversion and source records separate from generation and scoring software.", p),
        ]),
        Paragraph("The paper <i>Towards Understanding On-Policy Distillation through the Lens of Test-Time Scaling</i> documents a use of math-eval and math-vault in an evaluation setting. That reference is a documented use case rather than an endorsement or a claim about the general applicability of the artifacts.", p),
        Paragraph("5. Availability and citation", h),
        Paragraph(f"Software documentation report citation: Ge, X. (2026). <i>{TITLE}</i>. Software documentation report, published {REPORT_DATE}.<br/><br/>Related software artifact citation: Ge, X. (2026). <i>{TITLE}</i> (Version v0.1.0). Zenodo. https://doi.org/10.5281/zenodo.21411208.<br/><br/>Source code: https://github.com/Geraldxm/math-eval<br/>Related software artifact DOI: https://doi.org/10.5281/zenodo.21411208<br/>License: Apache-2.0<br/>Build this report: uv run --script docs/build_report.py", p),
        Paragraph("References", h),
        Paragraph("[1] X. Ge. math-vault: Curated, Traceable Snapshots of Public Mathematical Reasoning Datasets, Version v0.1.0, 2026. Zenodo. https://doi.org/10.5281/zenodo.21411214.", ref),
        Paragraph("[2] X. Ge et al. Towards Understanding On-Policy Distillation through the Lens of Test-Time Scaling, 2026. arXiv:2608.11829. https://arxiv.org/abs/2608.11829.", ref),
        Paragraph("[3] Hynek Kydlicek. Math-Verify, Version 0.9.0, 2026. Python package. https://pypi.org/project/math-verify/0.9.0/.", ref),
    ]
    doc.build(story, onFirstPage=page_number, onLaterPages=page_number)


if __name__ == "__main__":
    build()
