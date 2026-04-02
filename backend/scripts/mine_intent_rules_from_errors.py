from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


ROOT = Path(__file__).resolve().parents[2]
BACKEND = Path(__file__).resolve().parents[1]
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

DATA_JSON = ROOT / "data.json"
OUT_DIR = BACKEND / "tests" / "evaluation" / "results"
OUT_JSON = OUT_DIR / "intent_rule_mining_candidates.json"
OUT_MD = OUT_DIR / "intent_rule_mining_candidates.md"

TOKEN_RE = re.compile(r"[a-zA-ZÀ-ỹ0-9_]+", re.UNICODE)
STOPWORDS = {
    "va",
    "và",
    "la",
    "là",
    "the",
    "thế",
    "nao",
    "nào",
    "co",
    "có",
    "duoc",
    "được",
    "cho",
    "toi",
    "tôi",
    "ban",
    "bạn",
    "giup",
    "giúp",
    "nhu",
    "như",
    "nao",
    "gi",
    "gì",
    "ve",
    "về",
    "quy",
    "định",
    "quy định",
}


def _normalize_text(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip().lower())


def _tokens(s: str) -> List[str]:
    toks = [t.lower() for t in TOKEN_RE.findall(_normalize_text(s))]
    return [t for t in toks if len(t) >= 3 and t not in STOPWORDS]


def _load_cases(limit: int | None = None) -> List[Dict[str, Any]]:
    with open(DATA_JSON, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        return []
    out: List[Dict[str, Any]] = []
    for row in data:
        if not isinstance(row, dict):
            continue
        inp = str(row.get("input") or "").strip()
        o = row.get("output") or {}
        intent = str((o if isinstance(o, dict) else {}).get("intent") or "").strip()
        if not inp or not intent:
            continue
        out.append({"input": inp, "intent": intent})
        if limit and len(out) >= limit:
            break
    return out


def _eval_predictions(rows: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    from app.services.intent_detector import normalize_legacy_intent
    from app.services.query_intent import compute_intent_bundle

    out: List[Dict[str, str]] = []
    for r in rows:
        q = r["input"]
        gold = normalize_legacy_intent(r["intent"])
        pred = normalize_legacy_intent(str(compute_intent_bundle(q).get("detector_intent") or "nan"))
        out.append({"input": q, "gold": gold, "pred": pred})
    return out


def _candidate_phrases(texts: Iterable[str], ngram_max: int = 3) -> Counter:
    c: Counter = Counter()
    for t in texts:
        toks = _tokens(t)
        for n in range(1, ngram_max + 1):
            for i in range(0, max(0, len(toks) - n + 1)):
                ng = " ".join(toks[i : i + n])
                c[ng] += 1
    return c


def _phrase_to_regex(phrase: str) -> str:
    parts = [re.escape(x) for x in phrase.split()]
    return r"(?i)\b" + r"\s+".join(parts) + r"\b"


def _mine_for_pair(
    samples: List[Dict[str, str]],
    expected: str,
    predicted: str,
    min_support: int = 5,
    min_precision: float = 0.70,
    top_k: int = 12,
) -> List[Dict[str, Any]]:
    pair_rows = [x for x in samples if x["gold"] == expected and x["pred"] == predicted]
    if not pair_rows:
        return []
    all_texts = [x["input"] for x in samples]
    pair_texts = [x["input"] for x in pair_rows]

    freq_pair = _candidate_phrases(pair_texts)
    cands: List[Dict[str, Any]] = []

    for phrase, pair_count in freq_pair.most_common(400):
        if pair_count < min_support:
            continue
        if len(phrase) < 4:
            continue
        rgx = re.compile(_phrase_to_regex(phrase))
        matched = [x for x in samples if rgx.search(x["input"])]
        m = len(matched)
        if m < min_support:
            continue
        m_expected = sum(1 for x in matched if x["gold"] == expected)
        precision = m_expected / m
        if precision < min_precision:
            continue
        base = sum(1 for x in samples if x["gold"] == expected) / max(1, len(samples))
        lift = (precision / base) if base > 0 else 0.0
        cands.append(
            {
                "intent_id": expected,
                "from_confusion": f"{expected}->{predicted}",
                "phrase": phrase,
                "pattern": _phrase_to_regex(phrase),
                "support_in_confusion": pair_count,
                "matched_total": m,
                "matched_expected": m_expected,
                "precision_expected": round(precision, 4),
                "lift_expected": round(lift, 3),
            }
        )

    cands.sort(
        key=lambda x: (
            -x["lift_expected"],
            -x["precision_expected"],
            -x["support_in_confusion"],
            x["phrase"],
        )
    )
    return cands[:top_k]


def _top_confusions(samples: List[Dict[str, str]], limit: int = 8) -> List[Tuple[str, str, int]]:
    c = Counter((x["gold"], x["pred"]) for x in samples if x["gold"] != x["pred"])
    return [(a, b, n) for (a, b), n in c.most_common(limit)]


def _to_markdown(report: Dict[str, Any]) -> str:
    lines: List[str] = []
    lines.append("# Intent Rule Mining Candidates")
    lines.append("")
    lines.append(f"- Samples: **{report['samples']}**")
    lines.append(f"- Focus predicted class: **{report['focus_predicted_class']}**")
    lines.append("")
    lines.append("## Top Confusions")
    lines.append("")
    lines.append("| expected | predicted | count |")
    lines.append("|---|---|---:|")
    for x in report["top_confusions"]:
        lines.append(f"| `{x['expected']}` | `{x['predicted']}` | {x['count']} |")
    lines.append("")
    lines.append("## Suggested Rule Candidates")
    lines.append("")
    for g in report["candidate_groups"]:
        lines.append(f"### `{g['for_expected_intent']}` from `{g['from_confusion']}`")
        lines.append("")
        lines.append("| phrase | precision | lift | matched |")
        lines.append("|---|---:|---:|---:|")
        for c in g["candidates"]:
            lines.append(
                f"| `{c['phrase']}` | {c['precision_expected']:.2%} | {c['lift_expected']:.2f} | {c['matched_total']} |"
            )
        lines.append("")
    lines.append("## YAML Snippet (manual review required)")
    lines.append("")
    lines.append("```yaml")
    lines.append("structural_rules:")
    for g in report["candidate_groups"]:
        if not g["candidates"]:
            continue
        top = g["candidates"][:3]
        lines.append(f"  - intent_id: {g['for_expected_intent']}")
        lines.append("    priority: 92")
        lines.append("    confidence: 0.93")
        lines.append("    patterns:")
        for c in top:
            lines.append(f"      - '{c['pattern']}'")
    lines.append("```")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--limit", type=int, default=0, help="0 means full dataset")
    p.add_argument("--focus-pred", type=str, default="legal_explanation")
    args = p.parse_args()

    rows = _load_cases(limit=args.limit or None)
    if not rows:
        print("No valid rows loaded from data.json", file=sys.stderr)
        return 2
    samples = _eval_predictions(rows)

    confs = _top_confusions(samples, limit=20)
    focus = [x for x in confs if x[1] == args.focus_pred][:6]

    groups: List[Dict[str, Any]] = []
    for exp, pred, cnt in focus:
        cands = _mine_for_pair(samples, expected=exp, predicted=pred)
        groups.append(
            {
                "for_expected_intent": exp,
                "from_confusion": f"{exp}->{pred}",
                "confusion_count": cnt,
                "candidates": cands,
            }
        )

    report = {
        "samples": len(samples),
        "focus_predicted_class": args.focus_pred,
        "top_confusions": [{"expected": a, "predicted": b, "count": n} for a, b, n in confs],
        "candidate_groups": groups,
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    OUT_MD.write_text(_to_markdown(report), encoding="utf-8")

    print(json.dumps({"samples": report["samples"], "groups": len(groups)}, ensure_ascii=False))
    print("Wrote:")
    print(f"  {OUT_JSON}")
    print(f"  {OUT_MD}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

