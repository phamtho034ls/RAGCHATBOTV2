#!/usr/bin/env python3
"""EDA pipeline v2 trên mẫu ngẫu nhiên từ data_clean.json (JSONL).

Không gọi OpenAI / không truy vấn DB — chỉ:
- compute_intent_bundle (rule-based, INTENT_MODEL_ENABLED=false)
- extract_query_features, compute_strategy_scores, select_strategies
- So khớp nhãn gold `intent` trong file vs `detector_intent`
- Tính precision / recall / F1 / support per-class (strict match)

Output: JSON + Markdown trong backend/tests/evaluation/results/
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple


ROOT = Path(__file__).resolve().parents[2]
DATA_CLEAN = ROOT / "data_clean.json"
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


def _question_len_bucket(n: int) -> str:
    if n < 40:
        return "<40"
    if n < 80:
        return "40-79"
    if n < 120:
        return "80-119"
    if n < 180:
        return "120-179"
    return ">=180"


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


def _gold_vs_detector_match(gold: str, det: str, aliases: dict | None = None) -> bool:
    """Nhãn file data_clean chủ yếu tra_cuu_van_ban | article_query.

    Normalize both gold and det through LEGACY_INTENT_ALIASES so that e.g.
    gold='thu_tuc_hanh_chinh' / det='huong_dan_thu_tuc' counts as a match.
    """
    _aliases = aliases or {}
    g = _aliases.get((gold or "").strip(), (gold or "").strip())
    d = _aliases.get((det or "").strip(), (det or "").strip())
    if not g:
        return False
    if g == d:
        return True
    # Tra cứu tổng quát: detector có thể tra_cuu_van_ban hoặc article_query tùy câu
    if g == "tra_cuu_van_ban" and d in (
        "tra_cuu_van_ban",
        "trich_xuat_van_ban",
        "can_cu_phap_ly",
        "document_meta_relation",
    ):
        return True
    if g == "article_query" and d in ("article_query", "tra_cuu_van_ban"):
        return True
    return False


def _md_table_counter(title: str, counter: Counter, *, top: int = 30) -> str:
    lines = [f"### {title}", "", "| value | count | pct |", "|---|---:|---:|"]
    total = sum(counter.values()) or 1
    for v, n in counter.most_common(top):
        pct = n / total * 100
        lines.append(f"| `{v}` | {n} | {pct:.2f}% |")
    lines.append("")
    return "\n".join(lines)


def _compute_per_class_metrics(
    pairs: List[Tuple[str, str]],
) -> Dict[str, Any]:
    """Compute per-class precision / recall / F1 from (gold, predicted) pairs.

    Uses strict equality (gold == predicted).  Rows where gold is blank or
    ``__missing__`` are excluded from per-class stats but counted separately.
    Returns a dict suitable for JSON + Markdown output.
    """
    gold_cnt: Counter = Counter()
    pred_cnt: Counter = Counter()
    tp_cnt: Counter = Counter()
    missing_gold = 0

    for gold, pred in pairs:
        if not gold or gold == "__missing__":
            missing_gold += 1
            continue
        gold_cnt[gold] += 1
        pred_cnt[pred] += 1
        if gold == pred:
            tp_cnt[gold] += 1

    classes = sorted(set(gold_cnt.keys()) | set(pred_cnt.keys()))
    rows: List[Dict[str, Any]] = []
    for cls in classes:
        sup = gold_cnt[cls]
        predicted = pred_cnt[cls]
        tp = tp_cnt[cls]
        prec = tp / predicted if predicted else 0.0
        rec = tp / sup if sup else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0
        rows.append({
            "intent": cls,
            "support": sup,
            "predicted": predicted,
            "TP": tp,
            "precision": round(prec, 3),
            "recall": round(rec, 3),
            "f1": round(f1, 3),
        })

    rows.sort(key=lambda r: -r["support"])

    total_gold = sum(gold_cnt.values()) or 1
    total_pred = sum(pred_cnt.values()) or 1
    total_tp = sum(tp_cnt.values())
    micro_prec = total_tp / total_pred if total_pred else 0.0
    micro_rec = total_tp / total_gold
    micro_f1 = (2 * micro_prec * micro_rec / (micro_prec + micro_rec)) if (micro_prec + micro_rec) else 0.0

    n_cls = len(rows) or 1
    macro_prec = sum(r["precision"] for r in rows) / n_cls
    macro_rec = sum(r["recall"] for r in rows) / n_cls
    macro_f1 = sum(r["f1"] for r in rows) / n_cls

    sup_vals = [r["support"] for r in rows if r["support"] > 0]
    total_sup = sum(sup_vals) or 1
    weighted_f1 = sum(r["f1"] * r["support"] for r in rows) / total_sup

    return {
        "per_class": rows,
        "micro": {
            "precision": round(micro_prec, 3),
            "recall": round(micro_rec, 3),
            "f1": round(micro_f1, 3),
            "accuracy": round(micro_rec, 3),
        },
        "macro": {
            "precision": round(macro_prec, 3),
            "recall": round(macro_rec, 3),
            "f1": round(macro_f1, 3),
        },
        "weighted_f1": round(weighted_f1, 3),
        "missing_gold_rows": missing_gold,
        "total_evaluated": len(pairs) - missing_gold,
    }


def _md_intent_metrics_table(metrics: Dict[str, Any]) -> str:
    rows = metrics["per_class"]
    lines = [
        "### Per-class Intent Metrics (strict match: gold == detector)\n",
        "| intent | support | predicted | TP | precision | recall | F1 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| `{r['intent']}` | {r['support']} | {r['predicted']} "
            f"| {r['TP']} | {r['precision']:.3f} | {r['recall']:.3f} | {r['f1']:.3f} |"
        )
    lines.append("")
    m = metrics["micro"]
    mac = metrics["macro"]
    lines.append(
        f"**Micro** — precision: `{m['precision']:.3f}` | recall: `{m['recall']:.3f}` "
        f"| accuracy: `{m['accuracy']:.3f}` | F1: `{m['f1']:.3f}`  "
    )
    lines.append(
        f"**Macro** — precision: `{mac['precision']:.3f}` | recall: `{mac['recall']:.3f}` "
        f"| F1: `{mac['f1']:.3f}`  "
    )
    lines.append(f"**Weighted F1**: `{metrics['weighted_f1']:.3f}`  ")
    lines.append(f"*(rows excluded — missing gold label: {metrics['missing_gold_rows']})*")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=100, help="Sample size")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--out-prefix",
        type=str,
        default="pipeline_eda_v2_100",
        help="Output filename prefix",
    )
    args = p.parse_args()

    if not DATA_CLEAN.is_file():
        print(f"Missing {DATA_CLEAN}", file=sys.stderr)
        return 2

    rows = _load_jsonl_rows(DATA_CLEAN)
    if not rows:
        print("data_clean.json empty/unreadable", file=sys.stderr)
        return 2

    random.seed(args.seed)
    n = min(args.n, len(rows))
    sample = random.sample(rows, n)

    os.environ.setdefault("INTENT_MODEL_ENABLED", "false")

    from app.services.query_intent import compute_intent_bundle
    from app.services.query_features import extract_query_features
    from app.services.strategy_router import (
        STRATEGY_LOOKUP,
        STRATEGY_MULTI_QUERY,
        STRATEGY_SEMANTIC,
        compute_strategy_scores,
        select_strategies,
    )
    from app.services.intent_detector import LEGACY_INTENT_ALIASES

    for _log_name in ("app.services.strategy_router", "app.services.query_intent"):
        logging.getLogger(_log_name).setLevel(logging.WARNING)

    cnt_gold = Counter()
    cnt_detector = Counter()
    cnt_routing = Counter()
    cnt_strategies_tuple = Counter()
    cnt_strategy_member = Counter()
    cnt_len = Counter()
    cnt_conf = Counter()
    mismatch_examples: List[Dict[str, Any]] = []

    feature_true: Dict[str, int] = defaultdict(int)
    score_sums: Dict[str, float] = {
        STRATEGY_LOOKUP: 0.0,
        STRATEGY_SEMANTIC: 0.0,
        STRATEGY_MULTI_QUERY: 0.0,
    }
    confidences: List[float] = []

    gold_match = 0
    parallel_eligible = 0
    multi_strategy_selected = 0

    # For per-class precision/recall/F1
    gold_pred_pairs: List[Tuple[str, str]] = []

    rows_out: List[Dict[str, Any]] = []

    for r in sample:
        q = (r.get("question") or "").strip()
        gold_raw = str(r.get("intent") or "").strip()
        # Normalize legacy gold labels through LEGACY_INTENT_ALIASES so that
        # e.g. gold='thu_tuc_hanh_chinh' is treated as 'huong_dan_thu_tuc'.
        gold = LEGACY_INTENT_ALIASES.get(gold_raw, gold_raw)
        if not q:
            continue

        bundle = compute_intent_bundle(q)
        det = str(bundle.get("detector_intent") or "")
        rout = str(bundle.get("routing_intent") or "")
        conf = float(bundle.get("detector_confidence") or 0.0)
        rf = dict(bundle.get("rag_flags") or {})

        feats = extract_query_features(q)
        scores = compute_strategy_scores(feats)
        selected = select_strategies(scores, top_k=2)

        cnt_gold[gold or "__missing__"] += 1
        cnt_detector[det] += 1
        cnt_routing[rout] += 1
        key_sel = ",".join(selected)
        cnt_strategies_tuple[key_sel] += 1
        for s in selected:
            cnt_strategy_member[s] += 1
        cnt_len[_question_len_bucket(len(q))] += 1
        cnt_conf[_confidence_bucket(conf)] += 1
        confidences.append(conf)

        for fk, fv in feats.items():
            if fv is True:
                feature_true[fk] += 1

        for sk, sv in scores.items():
            score_sums[sk] += float(sv)

        gold_pred_pairs.append((gold, det))

        if _gold_vs_detector_match(gold_raw, det, LEGACY_INTENT_ALIASES):
            gold_match += 1
        elif len(mismatch_examples) < 40:
            mismatch_examples.append(
                {
                    "gold": gold_raw,
                    "gold_normalized": gold,
                    "detector": det,
                    "routing": rout,
                    "question": q[:220],
                }
            )

        if len(selected) >= 2:
            multi_strategy_selected += 1
        if len(selected) >= 2 and STRATEGY_LOOKUP not in selected:
            parallel_eligible += 1

        rows_out.append(
            {
                "question": q[:300],
                "gold_intent": gold,
                "detector_intent": det,
                "routing_intent": rout,
                "detector_confidence": conf,
                "selected_strategies": selected,
                "scores": {k: round(v, 3) for k, v in scores.items()},
                "rag_flags": rf,
                "features": {k: v for k, v in feats.items() if v},
            }
        )

    total = len(confidences) or 1
    intent_metrics = _compute_per_class_metrics(gold_pred_pairs)

    conf_stats = {
        "n": total,
        "min": min(confidences) if confidences else 0.0,
        "median": statistics.median(confidences) if confidences else 0.0,
        "mean": sum(confidences) / total,
        "max": max(confidences) if confidences else 0.0,
    }
    mean_scores = {k: round(v / total, 4) for k, v in score_sums.items()}

    out: Dict[str, Any] = {
        "meta": {
            "data_path": str(DATA_CLEAN),
            "seed": args.seed,
            "requested_n": args.n,
            "actual_n": total,
            "intent_model_enabled": os.environ.get("INTENT_MODEL_ENABLED"),
        },
        "label_alignment": {
            "gold_vs_detector_relaxed_match": gold_match,
            "gold_vs_detector_relaxed_match_pct": round(100.0 * gold_match / total, 2),
            "note": "So khớp mềm có alias: gold normalize qua LEGACY_INTENT_ALIASES trước; tra_cuu↔tra_cuu family; article_query↔article_query|tra_cuu_van_ban",
        },
        "counts": {
            "gold_intent": dict(cnt_gold),
            "detector_intent": dict(cnt_detector),
            "routing_intent": dict(cnt_routing),
            "selected_strategies_joined": dict(cnt_strategies_tuple),
            "strategy_in_top2": dict(cnt_strategy_member),
        },
        "features_true_counts": dict(sorted(feature_true.items(), key=lambda x: -x[1])),
        "strategy_score_means_across_sample": mean_scores,
        "retrieval_path_heuristics": {
            "rows_with_2plus_strategies": multi_strategy_selected,
            "rows_parallel_eligible_no_lookup_in_top2": parallel_eligible,
            "note": "Song song thực tế trong rag_chain_v2 còn phụ thuộc use_multi, multi_article_conditions",
        },
        "eda": {
            "question_length_buckets": dict(cnt_len),
            "detector_confidence_buckets": dict(cnt_conf),
        },
        "detector_confidence_stats": conf_stats,
        "intent_metrics": intent_metrics,
        "mismatch_examples": mismatch_examples,
        "sample_rows": rows_out[:15],
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    json_path = RESULTS_DIR / f"{args.out_prefix}.json"
    md_path = RESULTS_DIR / f"{args.out_prefix}.md"

    json_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    md: List[str] = []
    md.append("## Pipeline v2 EDA (`data_clean.json`)\n")
    md.append(f"- **Mẫu**: {total} câu (yêu cầu {args.n}), seed={args.seed}")
    md.append(f"- **INTENT_MODEL_ENABLED**: `{out['meta']['intent_model_enabled']}`")
    md.append(
        f"- **Gold vs detector (khớp mềm)**: {gold_match}/{total} "
        f"({out['label_alignment']['gold_vs_detector_relaxed_match_pct']}%)"
    )
    md.append(f"- **Chiến lược top-2 (trung bình điểm)**: `{mean_scores}`")
    md.append(
        f"- **Heuristic song song**: ≥2 strategy chọn được: {multi_strategy_selected}; "
        f"eligible (không có lookup trong top-2): {parallel_eligible}"
    )
    md.append("")
    md.append(_md_table_counter("Gold intent (nhãn file)", cnt_gold, top=20))
    md.append(_md_table_counter("Detector intent", cnt_detector, top=30))
    md.append(_md_intent_metrics_table(intent_metrics))
    md.append(_md_table_counter("Routing intent", cnt_routing, top=25))
    md.append(_md_table_counter("Chuỗi strategy đã chọn (top-2)", cnt_strategies_tuple, top=20))
    md.append(_md_table_counter("Strategy xuất hiện trong top-2", cnt_strategy_member, top=10))
    md.append(_md_table_counter("Độ dài câu (bucket ký tự)", cnt_len, top=15))
    md.append(_md_table_counter("Detector confidence buckets", cnt_conf, top=15))
    md.append("### Feature = true (số lần)\n")
    for k, v in sorted(feature_true.items(), key=lambda x: -x[1])[:25]:
        md.append(f"- `{k}`: {v}")
    md.append("")
    md.append("### Detector confidence (thống kê)\n")
    md.append("```json")
    md.append(json.dumps(conf_stats, ensure_ascii=False, indent=2))
    md.append("```")
    md.append("")
    if mismatch_examples:
        md.append("### Ví dụ không khớp gold vs detector (tối đa 12)\n")
        for ex in mismatch_examples[:12]:
            md.append(
                f"- gold=`{ex['gold']}` det=`{ex['detector']}` route=`{ex['routing']}`: {ex['question']}"
            )
        md.append("")

    md_path.write_text("\n".join(md).strip() + "\n", encoding="utf-8")

    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    raise SystemExit(main())
