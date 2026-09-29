#!/usr/bin/env python3
"""Build the math-eval technical report from its LaTeX source."""

import os
from pathlib import Path
from shutil import which
from subprocess import run


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "math-eval-v0.1.0.tex"


def build() -> None:
    os.environ.setdefault("SOURCE_DATE_EPOCH", "1788134400")
    if which("latexmk") is None:
        raise RuntimeError("latexmk is required; install a TeX distribution first.")
    run(
        ["latexmk", "-pdf", "-interaction=nonstopmode", "-halt-on-error", SOURCE.name],
        cwd=ROOT,
        check=True,
    )
    run(["latexmk", "-c", SOURCE.name], cwd=ROOT, check=True)


if __name__ == "__main__":
    build()
