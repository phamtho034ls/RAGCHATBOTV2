#!/usr/bin/env python3
"""EDA intent routing on random questions from data.json.

Runs the lightweight intent pipeline (no OpenAI calls):
- compute_intent_bundle(query): detector_intent + routing_intent + rag_flags
- query_understanding.extract_metadata_filters(query) (optional)

Outputs:
- JSON: aggregated counts + sampled examples
- Markdown: human-readable summary
"""

from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple


ROOT = Path(__file__).resolve().parents[2]  # .../rag_chatbot
DATA_JSON = ROOT / "data.json"
RESULTS_DIR = ROOT / "backend" / "tests" / "evaluation" / "results"


def _load_jsonl_rows(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for ln in path.read_text(encoding="utf-8", errors="replace").splitlines():
        ln = ln.strip()
        if not ln:
            continue
        try:
            rows.append(json.loads(ln))
        except json.JSONDecodeError:
            continue
    return rows


def _is_explicit_doc_ref(q: str) -> bool:
    import re

    if not q:
        return False
    return bool(re.search(r"\b\d+/\d{4}/[A-ZĐa-zđ0-9\-]+\b", q))


def _mentions_article(q: str) -> bool:
    import re

    return bool(re.search(r"\bđiều\s+\d+[a-zA-Z]?\b", q or "", re.IGNORECASE))


def _question_len_bucket(n: int) -> str:
    # buckets chosen for Vietnamese question lengths
    if n < 40:
        return "<40"
    if n < 80:
        return "40-79"
    if n < 120:
        return "80-119"
    if n < 180:
        return "120-179"
    if n < 260:
        return "180-259"
    return ">=260"


def _confidence_bucket(x: float) -> str:
    if x < 0.35:
        return "<0.35"
    if x < 0.55:
        return "0.35-0.54"
    if x < 0.75:
        return "0.55-0.74"
    if x < 0.90:
        return "0.75-0.89"
    return ">=0.90"


def _tokenize_for_eda(q: str) -> List[str]:
    import re

    tokens = re.findall(r"[0-9a-zà-ỹđ]+", (q or "").lower(), re.IGNORECASE)
    stop = {
        "là",
        "về",
        "theo",
        "của",
        "cho",
        "và",
        "các",
        "những",
        "như",
        "thế",
        "nào",
        "quy",
        "định",
        "điều",
        "khoản",
        "điểm",
        "mục",
        "chương",
        "phần",
        "trong",
        "khi",
        "được",
        "gì",
        "bao",
        "nhiêu",
        "không",
        "có",
        "văn",
        "bản",
        "pháp",
        "luật",
        "nghị",
        "định",
        "thông",
        "tư",
        "quyết",
        "luật",
        "ubnd",
        "cấp",
        "xã",
        "huyện",
        "tỉnh",
    }
    out = [t for t in tokens if len(t) >= 4 and t not in stop and not t.isdigit()]
    return out[:60]

@dataclass
class Example:
    question: str
    detector_intent: str
    routing_intent: str
    detector_confidence: float
    rag_flags: Dict[str, Any]
    filters: Dict[str, Any]


def _summarize_flags(bundles: List[Dict[str, Any]]) -> Dict[str, Any]:
    key_counts: Dict[str, Counter] = {}
    for k in ("is_legal_lookup", "use_multi_article", "needs_expansion"):
        c = Counter()
        for b in bundles:
            rf = b.get("rag_flags") or {}
            c[str(bool(rf.get(k, False)))] += 1
        key_counts[k] = c
    return {k: dict(v) for k, v in key_counts.items()}


def _md_table_counter(title: str, counter: Counter, *, top: int = 30) -> str:
    lines = [f"### {title}", "", "| value | count | pct |", "|---|---:|---:|"]
    total = sum(counter.values()) or 1
    for v, n in counter.most_common(top):
        pct = n / total * 100
        lines.append(f"| `{v}` | {n} | {pct:.2f}% |")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=500, help="Sample size from data.json")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-prefix", type=str, default="intent_eda_datajson_500", help="Output filename prefix")
    args = p.parse_args()

    if not DATA_JSON.is_file():
        print(f"Missing {DATA_JSON}", file=sys.stderr)
        return 2

    rows = _load_jsonl_rows(DATA_JSON)
    if not rows:
        print("data.json empty/unreadable", file=sys.stderr)
        return 2

    random.seed(args.seed)
    sample = random.sample(rows, min(args.n, len(rows)))

    # Import here so script can run from anywhere once backend path is on sys.path
    # IMPORTANT: For EDA speed and determinism, disable local intent model warmup.
    # Some environments load PhoBERT weights when INTENT_MODEL_ENABLED=true; force it off via env
    # BEFORE importing any app modules.
    os.environ.setdefault("INTENT_MODEL_ENABLED", "false")

    from app.services.query_intent import compute_intent_bundle
    from app.services.query_understanding import extract_metadata_filters
    from app.services.legal_scope import query_has_strong_legal_scope_signals

    bundles: List[Dict[str, Any]] = []
    examples_oos: List[Example] = []
    examples_nan: List[Example] = []
    examples_explicit_doc: List[Example] = []
    examples_article: List[Example] = []
    examples_weak_scope: List[Tuple[str, str]] = []  # (routing_intent, question)
    examples_low_conf: List[Example] = []

    detector_conf: List[float] = []
    cnt_detector = Counter()
    cnt_routing = Counter()
    cnt_pair = Counter()
    cnt_len_bucket = Counter()
    cnt_conf_bucket = Counter()
    cnt_keywords = Counter()

    # cross tabs
    xtab_route_by_docref = Counter()
    xtab_route_by_article = Counter()
    xtab_route_by_len = Counter()

    explicit_doc_n = 0
    article_n = 0

    for r in sample:
        q = (r.get("question") or "").strip()
        if not q:
            continue
        b = compute_intent_bundle(q)
        bundles.append(b)

        det = str(b.get("detector_intent") or "")
        rout = str(b.get("routing_intent") or "")
        conf = float(b.get("detector_confidence") or 0.0)
        rf = dict(b.get("rag_flags") or {})
        flt = dict(extract_metadata_filters(q) or {})

        cnt_detector[det] += 1
        cnt_routing[rout] += 1
        cnt_pair[f"{det} → {rout}"] += 1
        detector_conf.append(conf)
        cnt_len_bucket[_question_len_bucket(len(q))] += 1
        cnt_conf_bucket[_confidence_bucket(conf)] += 1
        for t in _tokenize_for_eda(q):
            cnt_keywords[t] += 1

        ex = Example(
            question=q,
            detector_intent=det,
            routing_intent=rout,
            detector_confidence=conf,
            rag_flags=rf,
            filters=flt,
        )
        if rout == "out_of_scope" and len(examples_oos) < 25:
            examples_oos.append(ex)
        if det == "nan" and len(examples_nan) < 25:
            examples_nan.append(ex)
        if _is_explicit_doc_ref(q):
            explicit_doc_n += 1
            if len(examples_explicit_doc) < 20:
                examples_explicit_doc.append(ex)
            xtab_route_by_docref[f"{rout} | doc_ref=true"] += 1
        else:
            xtab_route_by_docref[f"{rout} | doc_ref=false"] += 1
        if _mentions_article(q):
            article_n += 1
            if len(examples_article) < 20:
                examples_article.append(ex)
            xtab_route_by_article[f"{rout} | mentions_dieu=true"] += 1
        else:
            xtab_route_by_article[f"{rout} | mentions_dieu=false"] += 1
        if not query_has_strong_legal_scope_signals(q) and len(examples_weak_scope) < 30:
            examples_weak_scope.append((rout, q))
        if conf <= 0.35 and len(examples_low_conf) < 30:
            examples_low_conf.append(ex)
        xtab_route_by_len[f"{rout} | len={_question_len_bucket(len(q))}"] += 1

    total = len(detector_conf) or 1
    conf_stats = {
        "n": total,
        "min": min(detector_conf) if detector_conf else 0.0,
        "p25": statistics.quantiles(detector_conf, n=4)[0] if len(detector_conf) >= 4 else 0.0,
        "median": statistics.median(detector_conf) if detector_conf else 0.0,
        "p75": statistics.quantiles(detector_conf, n=4)[2] if len(detector_conf) >= 4 else 0.0,
        "max": max(detector_conf) if detector_conf else 0.0,
        "mean": sum(detector_conf) / total,
    }

    out = {
        "meta": {
            "data_path": str(DATA_JSON),
            "seed": args.seed,
            "requested_n": args.n,
            "actual_n": len(sample),
        },
        "counts": {
            "detector_intent": dict(cnt_detector),
            "routing_intent": dict(cnt_routing),
            "detector_to_routing": dict(cnt_pair),
        },
        "eda": {
            "question_length_buckets": dict(cnt_len_bucket),
            "detector_confidence_buckets": dict(cnt_conf_bucket),
            "top_keywords": dict(cnt_keywords.most_common(80)),
            "cross_tabs": {
                "routing_by_doc_ref": dict(xtab_route_by_docref),
                "routing_by_mentions_article": dict(xtab_route_by_article),
                "routing_by_len_bucket": dict(xtab_route_by_len),
            },
        },
        "rag_flags": _summarize_flags(bundles),
        "signals": {
            "explicit_doc_ref_count": explicit_doc_n,
            "mentions_article_count": article_n,
            "weak_legal_scope_signal_count": len(examples_weak_scope),
        },
        "detector_confidence": conf_stats,
        "examples": {
            "out_of_scope": [asdict(x) for x in examples_oos],
            "detector_nan": [asdict(x) for x in examples_nan],
            "explicit_doc_ref": [asdict(x) for x in examples_explicit_doc],
            "mentions_article": [asdict(x) for x in examples_article],
            "weak_scope_signal": [{"routing_intent": ri, "question": q} for ri, q in examples_weak_scope],
            "low_confidence": [asdict(x) for x in examples_low_conf],
        },
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    json_path = RESULTS_DIR / f"{args.out_prefix}.json"
    md_path = RESULTS_DIR / f"{args.out_prefix}.md"

    json_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    md: List[str] = []
    md.append(f"## Intent EDA (data.json sample)\n")
    md.append(f"- **sample**: {len(sample)} (requested {args.n}), seed={args.seed}")
    md.append(f"- **explicit doc ref**: {explicit_doc_n} ({explicit_doc_n/ (len(sample) or 1) * 100:.2f}%)")
    md.append(f"- **mentions Điều**: {article_n} ({article_n/ (len(sample) or 1) * 100:.2f}%)")
    md.append("")
    md.append(_md_table_counter("Routing intent distribution", cnt_routing, top=40))
    md.append(_md_table_counter("Detector intent distribution", cnt_detector, top=60))
    md.append(_md_table_counter("Detector → routing (top pairs)", cnt_pair, top=35))
    md.append(_md_table_counter("Question length buckets (chars)", cnt_len_bucket, top=20))
    md.append(_md_table_counter("Detector confidence buckets", cnt_conf_bucket, top=20))
    md.append(_md_table_counter("Top keywords (EDA tokenizer)", cnt_keywords, top=50))
    md.append("### RAG flags\n")
    for k, c in out["rag_flags"].items():
        md.append(f"- **{k}**: {c}")
    md.append("")
    md.append("### Detector confidence\n")
    md.append("```json")
    md.append(json.dumps(conf_stats, ensure_ascii=False, indent=2))
    md.append("```")
    md.append("")
    if examples_oos:
        md.append("### Examples: routed to out_of_scope\n")
        for x in examples_oos[:10]:
            md.append(f"- `{x.routing_intent}` / `{x.detector_intent}` conf={x.detector_confidence:.2f}: {x.question}")
        md.append("")
    if examples_nan:
        md.append("### Examples: detector_intent = nan\n")
        for x in examples_nan[:10]:
            md.append(f"- `{x.routing_intent}` / `{x.detector_intent}` conf={x.detector_confidence:.2f}: {x.question}")
        md.append("")
    if examples_weak_scope:
        md.append("### Examples: weak legal-scope heuristic (not necessarily wrong)\n")
        for ri, q in examples_weak_scope[:12]:
            md.append(f"- `{ri}`: {q}")
        md.append("")
    if examples_low_conf:
        md.append("### Examples: low detector confidence (<= 0.35)\n")
        for x in examples_low_conf[:12]:
            md.append(f"- `{x.routing_intent}` / `{x.detector_intent}` conf={x.detector_confidence:.2f}: {x.question}")
        md.append("")

    md_path.write_text("\n".join(md).strip() + "\n", encoding="utf-8")

    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")
    return 0


if __name__ == "__main__":
    # Ensure backend module path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    raise SystemExit(main())

