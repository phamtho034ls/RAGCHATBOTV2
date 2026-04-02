"""Article grouper – groups retrieved chunks by article for coherent context.

Prevents fragmented answers by:
1. Grouping chunks from the same article together
2. Deduplicating overlapping chunks
3. Building structured context with article headers
"""

from __future__ import annotations

import re
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple


def extract_article_number(text: str) -> Optional[str]:
    """Extract article number from text (e.g. 'Điều 7' → '7')."""
    m = re.search(r"[Đđ]iều\s+(\d+[a-zA-Z]?)", text or "")
    return m.group(1) if m else None


def _chunk_dedup_key(chunk: Dict) -> str:
    """Stable key: ưu tiên id vector/chunk — tránh gộp nhầm nhiều đoạn khác nhau
    chỉ vì 120 ký tự đầu trùng (header Điều, v.v.)."""
    cid = chunk.get("id")
    if cid is not None and str(cid).strip():
        return f"id:{cid}"
    aid = chunk.get("article_id")
    cl = chunk.get("clause_id")
    if aid is not None:
        if cl is not None:
            return f"a:{aid}:c:{cl}"
        vtx = chunk.get("vector_id")
        if vtx is not None:
            return f"a:{aid}:v:{vtx}"
    txt = (chunk.get("text_chunk") or "").strip()
    if len(txt) >= 24:
        return f"txt:{txt[:500]}"
    return ""


def dedup_chunks(chunks: List[Dict]) -> List[Dict]:
    """Remove duplicate chunks (same id / cùng điều-khoản / trùng đoạn dài)."""
    seen: set[str] = set()
    out: List[Dict] = []
    for i, c in enumerate(chunks):
        key = _chunk_dedup_key(c)
        if not key:
            key = f"row:{i}:{hash((c.get('article_id'), c.get('document_id'), (c.get('text_chunk') or '')[:800]))}"
        if key not in seen:
            seen.add(key)
            out.append(c)
    return out


def _group_key(chunk: Dict) -> Tuple[Optional[int], Optional[str]]:
    """(document_id, article_number) grouping key."""
    art = chunk.get("article_number")
    if art:
        m = re.search(r"(\d+[a-zA-Z]?)", str(art))
        art = m.group(1) if m else str(art).strip()
    return (chunk.get("document_id"), art)


def group_chunks_by_article(chunks: List[Dict]) -> Dict[Tuple, List[Dict]]:
    """Group chunks by (document_id, article_number).

    Preserves insertion order per group. Ungroupable chunks
    (no article_number) go under key (doc_id, None).
    """
    groups: Dict[Tuple, List[Dict]] = OrderedDict()
    for chunk in chunks:
        key = _group_key(chunk)
        groups.setdefault(key, []).append(chunk)
    return groups


def format_grouped_context(groups: Dict[Tuple, List[Dict]]) -> str:
    """Build a structured context string from article groups.

    Each group gets a clear header:
        [Nguồn: <doc_title> | Số hiệu | Điều <N>. <title> | Hiệu lực thi hành]
        <chunk texts joined>
    """
    parts: List[str] = []
    idx = 0
    for (doc_id, art_num), chunks in groups.items():
        if not chunks:
            continue
        idx += 1
        first = chunks[0]
        doc_title = first.get("document_title") or first.get("doc_number", "Văn bản")
        doc_number = first.get("doc_number", "")
        art_title = first.get("article_title", "")

        header_parts = [f"Nguồn {idx}: {doc_title}"]
        if doc_number:
            header_parts.append(f"Số hiệu: {doc_number}")
        if art_num:
            label = f"Điều {art_num}"
            if art_title:
                label += f". {art_title}"
            header_parts.append(label)

        clause = first.get("clause_number")
        if clause:
            header_parts.append(f"Khoản {clause}")

        eff = first.get("effective_date")
        if eff:
            header_parts.append(f"Hiệu lực thi hành: từ ngày {eff}")
        issued = first.get("issued_date")
        if issued and not eff:
            header_parts.append(f"Ngày ban hành: {issued}")

        header = "[" + " | ".join(header_parts) + "]"

        texts = []
        seen: set[str] = set()
        for c in chunks:
            txt = (c.get("text_chunk") or "").strip()
            key = txt[:120]
            if key and key not in seen:
                seen.add(key)
                texts.append(txt)

        parts.append(f"{header}\n" + "\n".join(texts))

    return "\n\n---\n\n".join(parts)
