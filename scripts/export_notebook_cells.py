#!/usr/bin/env python3
"""
Split colab/predictive_maintenance_ai4i2020.ipynb into one file per cell.

Usage (from repo root):
    python scripts/export_notebook_cells.py

Output: colab/notebook_export/cell_NNN_kind_slug.ext
Remove the colab/notebook_export/ line from .gitignore if you want these in Git.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NB = ROOT / "colab" / "predictive_maintenance_ai4i2020.ipynb"
OUT = ROOT / "colab" / "notebook_export"


def slugify(text: str, max_len: int = 48) -> str:
    s = text.strip().split("\n", 1)[0]
    s = re.sub(r"[^a-zA-Z0-9]+", "_", s)
    s = s.strip("_").lower() or "cell"
    return s[:max_len]


def main() -> None:
    if not NB.exists():
        raise SystemExit(f"Notebook not found: {NB}")

    OUT.mkdir(parents=True, exist_ok=True)
    nb = json.loads(NB.read_text(encoding="utf-8"))
    cells = nb.get("cells", [])

    for i, cell in enumerate(cells):
        src = cell.get("source", [])
        if isinstance(src, list):
            text = "".join(src)
        else:
            text = str(src)
        kind = cell.get("cell_type", "unknown")
        head = text.lstrip().split("\n", 1)[0] if text.strip() else kind
        slug = slugify(head)
        ext = ".md" if kind == "markdown" else ".py"
        path = OUT / f"cell_{i:03d}_{kind}_{slug}{ext}"
        banner = (
            f"# Exported from notebook cell {i} ({kind})\n"
            f"# Source: {NB.name}\n\n"
            if kind == "markdown"
            else f'"""Exported from notebook cell {i} ({kind}). Source: {NB.name}."""\n\n'
        )
        path.write_text(banner + text.rstrip() + "\n", encoding="utf-8")
        print("Wrote", path.relative_to(ROOT))

    print(f"\nExported {len(cells)} cells to {OUT.relative_to(ROOT)}/")


if __name__ == "__main__":
    main()
