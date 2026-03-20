"""High-level legal parser that builds Document -> Article -> Clause -> Point tree."""

from __future__ import annotations

from app.pipeline.structure_detector import DocumentStructure, detect_structure


def build_article_tree(text: str) -> DocumentStructure:
    """Parse cleaned legal text into a hierarchical document tree."""
    return detect_structure(text)

