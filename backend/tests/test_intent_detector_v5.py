"""Validation tests for Intent Detector v5 — PhoBERT multitask (8 grouped labels)."""
import sys
import os
import asyncio

os.environ["PYTHONIOENCODING"] = "utf-8"
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Tắt model khi chạy test semantic/structural để kết quả ổn định
os.environ["INTENT_MODEL_ENABLED"] = "false"

from app.services.intent_detector import (
    detect_intent,
    detect_intent_rule_based,
    _detect_structural,
    _detect_semantic,
    warmup_intent_index,
    get_index_stats,
    VALID_INTENTS,
    COMMUNE_LEVEL_INTENTS,
    INTENT_PROTOTYPES,
    INTENT_SEMANTIC_DESCRIPTIONS,
    INTENT_MAPPING,
)


def _sep(title: str):
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


def test_exports():
    _sep("EXPORT CHECKS — 8 grouped labels")
    checks = [
        ("VALID_INTENTS is list", isinstance(VALID_INTENTS, list)),
        ("8 grouped intents", len(VALID_INTENTS) == 8),
        ("COMMUNE_LEVEL_INTENTS is frozenset", isinstance(COMMUNE_LEVEL_INTENTS, frozenset)),
        ("4 commune intents", len(COMMUNE_LEVEL_INTENTS) == 4),
        ("legal_lookup in VALID_INTENTS", "legal_lookup" in VALID_INTENTS),
        ("legal_explanation in VALID_INTENTS", "legal_explanation" in VALID_INTENTS),
        ("violation in VALID_INTENTS", "violation" in VALID_INTENTS),
        ("admin_scenario in VALID_INTENTS", "admin_scenario" in VALID_INTENTS),
        ("INTENT_PROTOTYPES has 8 groups", len(INTENT_PROTOTYPES) == 8),
        ("INTENT_SEMANTIC_DESCRIPTIONS has 8 groups", len(INTENT_SEMANTIC_DESCRIPTIONS) == 8),
        ("Each group has prototypes >=3", all(len(p) >= 3 for p in INTENT_PROTOTYPES.values())),
        ("INTENT_MAPPING has 18 entries", len(INTENT_MAPPING) == 18),
        ("detect_intent_rule_based callable", callable(detect_intent_rule_based)),
        ("warmup_intent_index callable", callable(warmup_intent_index)),
        ("get_index_stats callable", callable(get_index_stats)),
    ]
    passed = 0
    for label, ok in checks:
        if ok:
            passed += 1
        print(f"  [{'PASS' if ok else 'FAIL'}] {label}")
    print(f"\n  Result: {passed}/{len(checks)} passed")
    return passed == len(checks)


def test_legacy_alias_mapping():
    _sep("LEGACY INTENT MAPPING (old 18 → new 8 groups)")
    from app.services.intent_detector import normalize_legacy_intent

    expected = {
        "article_query": "legal_lookup",
        "tra_cuu_van_ban": "legal_lookup",
        "trich_xuat_van_ban": "legal_lookup",
        "can_cu_phap_ly": "legal_lookup",
        "giai_thich_quy_dinh": "legal_explanation",
        "hoi_dap_chung": "legal_explanation",
        "huong_dan_thu_tuc": "procedure",
        "kiem_tra_ho_so": "procedure",
        "xu_ly_vi_pham_hanh_chinh": "violation",
        "kiem_tra_thanh_tra": "violation",
        "so_sanh_van_ban": "comparison",
        "tom_tat_van_ban": "summarization",
        "soan_thao_van_ban": "document_generation",
        "tao_bao_cao": "document_generation",
        "admin_planning": "admin_scenario",
        "to_chuc_su_kien_cong": "admin_scenario",
        "hoa_giai_van_dong": "admin_scenario",
        "document_meta_relation": "admin_scenario",
        # very old aliases
        "bao_ve_xa_hoi": "legal_explanation",
        "program_goal": "legal_explanation",
        "thu_tuc_hanh_chinh": "procedure",
    }
    passed = 0
    for old_intent, new_group in expected.items():
        got = normalize_legacy_intent(old_intent)
        ok = got == new_group
        if ok:
            passed += 1
        print(f"  [{'PASS' if ok else 'FAIL'}] {old_intent} → {got} (expected: {new_group})")
    print(f"\n  Result: {passed}/{len(expected)} passed")
    return passed == len(expected)


def test_structural():
    _sep("STRUCTURAL LAYER (sync, no LLM) — kết quả qua normalize_legacy")
    # Structural YAML vẫn dùng tên cũ → normalize_legacy map về 8 nhóm
    cases = [
        # (query, expected_grouped_intent, min_conf)
        ("Điều 6 Luật Đầu tư 2025 quy định gì?", "legal_lookup", 0.90),
        ("Tóm tắt Luật Thể dục thể thao 2006", "summarization", 0.90),
        ("Soạn công văn xin gia hạn giấy phép", "document_generation", 0.90),
        ("Lập báo cáo tổng kết năm 2025", "document_generation", 0.90),
        ("", None, None),
        ("Karaoke gây ồn", None, None),
        ("Chính sách bảo trợ xã hội", None, None),
    ]
    passed = 0
    for query, expected_intent, min_conf in cases:
        result = _detect_structural(query)
        if expected_intent is None:
            ok = result is None
            label = "None" if ok else f"{result[0]} ({result[1]})"
        else:
            ok = result is not None and result[0] == expected_intent and result[1] >= min_conf
            label = f"{result[0]} ({result[1]})" if result else "None"
        if ok:
            passed += 1
        print(f"  [{'PASS' if ok else 'FAIL'}] \"{query[:55]}\" => {label}")
    print(f"\n  Result: {passed}/{len(cases)} passed")
    return passed == len(cases)


def test_semantic_layer():
    _sep("SEMANTIC LAYER (embedding cosine, requires warmup) — 8 groups")

    print("  Building prototype index...")
    warmup_intent_index()
    stats = get_index_stats()
    print(f"  Index stats: {stats['total_prototypes']} prototypes, {stats['intents_covered']} intents")
    if not stats["index_loaded"]:
        print("  [SKIP] Index not loaded — embedding model unavailable")
        return True

    cases = [
        ("Chính sách bảo trợ xã hội đối với người cao tuổi là gì?", "legal_explanation"),
        ("Karaoke gây ồn ào quá giờ, xử phạt thế nào?", "violation"),
        ("Các ngành nghề cấm đầu tư kinh doanh", "legal_lookup"),
        ("Tổ chức Đại hội Thể dục thể thao cấp xã", "admin_scenario"),
        ("Hàng xóm tranh chấp ranh giới đất", "admin_scenario"),
        ("Lập kế hoạch thanh tra cơ sở internet", "violation"),
        ("Xin chào bạn", "legal_explanation"),
        ("Điều 47 Luật Di sản văn hóa quy định gì?", "legal_lookup"),
        ("Soạn công văn xin gia hạn giấy phép", "document_generation"),
        ("Luật Đầu tư 2020 và 2025 khác gì nhau?", "comparison"),
    ]

    passed = 0
    for query, expected_intent in cases:
        result = _detect_semantic(query)
        if result is None:
            ok = False
            label = "None (deferred to LLM)"
        else:
            intent, conf = result
            ok = intent == expected_intent
            label = f"{intent} (conf={conf:.2f})"
        if ok:
            passed += 1
        print(f"  [{'PASS' if ok else 'FAIL'}] \"{query[:55]}\" => {label}")

    print(f"\n  Result: {passed}/{len(cases)} passed")
    return passed >= len(cases) * 0.7  # 70% threshold


def test_full_pipeline():
    _sep("FULL PIPELINE (async, Guard + Classifier + Semantic + Structural + LLM)")
    try:
        import openai  # noqa: F401
    except Exception:
        print("  [SKIP] openai package not installed for LLM layer")
        return True
    cases = [
        ("", "legal_explanation", "guard"),
        ("123", "legal_explanation", "guard"),
        ("Điều 47 Luật Di sản văn hóa", "legal_lookup", None),
        ("Tóm tắt Luật Đầu tư 2025", "summarization", None),
        ("Chính sách bảo trợ xã hội đối với người cao tuổi là gì?", "legal_explanation", None),
        ("Các ngành, nghề cấm đầu tư kinh doanh theo Luật Đầu tư 2025", "legal_lookup", None),
        ("Karaoke gây ồn ào đến 2 giờ sáng, lập biên bản xử phạt thế nào?", "violation", None),
        ("Xin chào", "legal_explanation", None),
    ]

    async def _run():
        passed = 0
        for query, expected_intent, expected_method in cases:
            result = await detect_intent(query)
            intent = result["intent"]
            conf = result["confidence"]
            method = result.get("method", "?")
            ok_intent = intent == expected_intent
            ok_method = expected_method is None or method == expected_method
            ok = ok_intent and ok_method
            if ok:
                passed += 1
            status = "PASS" if ok else "FAIL"
            extra = ""
            if not ok_intent:
                extra += f" [expected intent={expected_intent}]"
            if not ok_method:
                extra += f" [expected method={expected_method}]"
            print(f"  [{status}] \"{query[:55]}\" => {intent} ({conf:.2f}) via {method}{extra}")
        print(f"\n  Result: {passed}/{len(cases)} passed")
        return passed >= len(cases) * 0.7

    return asyncio.run(_run())


def test_rag_flags():
    _sep("RAG FLAGS — map_intent_to_rag_flags cho 8 nhóm")
    from app.services.intent_detector import map_intent_to_rag_flags

    cases = [
        ("legal_lookup",      {"is_legal_lookup": True,  "use_multi_article": False}),
        ("legal_explanation", {"is_legal_lookup": False, "use_multi_article": True}),
        ("procedure",         {"is_legal_lookup": False, "use_multi_article": True}),
        ("violation",         {"is_legal_lookup": False, "use_multi_article": True}),
        ("comparison",        {"is_legal_lookup": False, "use_multi_article": True}),
        ("summarization",     {"is_legal_lookup": False, "use_multi_article": True}),
        ("document_generation", {"is_legal_lookup": False, "use_multi_article": True}),
        ("admin_scenario",    {"is_legal_lookup": False, "use_multi_article": True}),
    ]
    passed = 0
    for intent, expected_subset in cases:
        flags = map_intent_to_rag_flags(intent)
        ok = all(flags.get(k) == v for k, v in expected_subset.items())
        if ok:
            passed += 1
        print(f"  [{'PASS' if ok else 'FAIL'}] {intent}: {flags}")
    print(f"\n  Result: {passed}/{len(cases)} passed")
    return passed == len(cases)


def test_backward_compat():
    _sep("BACKWARD COMPAT — detect_intent_rule_based returns new grouped labels")
    cases = [
        ("Điều 6 Luật Đầu tư", "legal_lookup"),
        ("Soạn công văn xin gia hạn giấy phép", "document_generation"),
        ("", "legal_explanation"),
        ("Karaoke gây ồn, xử phạt?", None),  # violation or legal_lookup
    ]

    passed = 0
    for query, expected_intent in cases:
        intent, conf = detect_intent_rule_based(query)
        if expected_intent is None:
            ok = intent in VALID_INTENTS
        else:
            ok = intent == expected_intent
        if ok:
            passed += 1
        print(f"  [{'PASS' if ok else 'FAIL'}] \"{query[:55]}\" => {intent} ({conf:.2f})")

    print(f"\n  Result: {passed}/{len(cases)} passed")
    return passed == len(cases)


if __name__ == "__main__":
    all_ok = True
    all_ok &= test_exports()
    all_ok &= test_legacy_alias_mapping()
    all_ok &= test_rag_flags()
    all_ok &= test_structural()
    all_ok &= test_semantic_layer()
    all_ok &= test_backward_compat()
    all_ok &= test_full_pipeline()

    _sep("SUMMARY")
    if all_ok:
        print("  ALL TESTS PASSED")
    else:
        print("  SOME TESTS FAILED — check details above")
    sys.exit(0 if all_ok else 1)
