"""Validation tests for Intent Detector v5 — 3-Tier Semantic + LLM + Auto-index."""
import sys
import os
import asyncio

os.environ["PYTHONIOENCODING"] = "utf-8"
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# PhoBERT classifier thay đổi kết quả rule_based; test semantic/structural giữ ổn định.
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
)


def _sep(title: str):
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


def test_exports():
    _sep("EXPORT CHECKS")
    checks = [
        ("VALID_INTENTS is list", isinstance(VALID_INTENTS, list)),
        ("23 intents", len(VALID_INTENTS) == 23),
        ("COMMUNE_LEVEL_INTENTS is set", isinstance(COMMUNE_LEVEL_INTENTS, set)),
        ("7 commune intents", len(COMMUNE_LEVEL_INTENTS) == 7),
        ("hoi_dap_chung in VALID_INTENTS", "hoi_dap_chung" in VALID_INTENTS),
        ("INTENT_PROTOTYPES has 23 intents", len(INTENT_PROTOTYPES) == 23),
        ("INTENT_SEMANTIC_DESCRIPTIONS has 23 intents", len(INTENT_SEMANTIC_DESCRIPTIONS) == 23),
        ("Each intent has prototypes", all(len(p) >= 3 for p in INTENT_PROTOTYPES.values())),
        ("detect_intent_rule_based exists", callable(detect_intent_rule_based)),
        ("warmup_intent_index exists", callable(warmup_intent_index)),
        ("get_index_stats exists", callable(get_index_stats)),
    ]
    passed = 0
    for label, ok in checks:
        if ok:
            passed += 1
        print(f"  [{'PASS' if ok else 'FAIL'}] {label}")
    print(f"\n  Result: {passed}/{len(checks)} passed")
    return passed == len(checks)


def test_structural():
    _sep("STRUCTURAL LAYER (sync, no LLM)")
    cases = [
        ("Điều 6 Luật Đầu tư 2025 quy định gì?", "article_query", 0.95),
        ("Tóm tắt nội dung 36/2019/NĐ-CP", "article_query", 0.95),
        ("Tóm tắt Luật Thể dục thể thao 2006", "tom_tat_van_ban", 0.95),
        ("Soạn công văn xin gia hạn giấy phép", "soan_thao_van_ban", 0.95),
        ("Khoản 2 Điều 6 Luật Đầu tư", "article_query", 0.97),
        ("Lập báo cáo tổng kết năm 2025", "tao_bao_cao", 0.95),
        ("Ai ban hành Nghị định 144?", "document_metadata", 0.93),
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
    _sep("SEMANTIC LAYER (embedding cosine, requires warmup)")

    print("  Building prototype index...")
    warmup_intent_index()
    stats = get_index_stats()
    print(f"  Index stats: {stats['total_prototypes']} prototypes, {stats['intents_covered']} intents")
    if not stats["index_loaded"]:
        print("  [SKIP] Index not loaded — embedding model unavailable")
        return True

    cases = [
        ("Chính sách bảo trợ xã hội đối với người cao tuổi là gì?", "giai_thich_quy_dinh"),
        ("Trẻ em bị bố đánh đập hàng ngày, cần can thiệp ngay", "bao_ve_xa_hoi"),
        ("Karaoke gây ồn ào quá giờ, xử phạt thế nào?", "xu_ly_vi_pham_hanh_chinh"),
        ("Các ngành nghề cấm đầu tư kinh doanh", "trich_xuat_van_ban"),
        ("Tổ chức Đại hội Thể dục thể thao cấp xã", "to_chuc_su_kien_cong"),
        ("Bảo tồn làng nghề truyền thống đang mai một", "bao_ton_phat_trien"),
        ("Hàng xóm tranh chấp ranh giới đất", "hoa_giai_van_dong"),
        ("Lập kế hoạch thanh tra cơ sở internet", "kiem_tra_thanh_tra"),
        ("Xin chào bạn", "hoi_dap_chung"),
        ("Mục tiêu chương trình nông thôn mới là gì?", "program_goal"),
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
    return passed >= len(cases) * 0.7  # 70% threshold (some may defer to LLM)


def test_full_pipeline():
    _sep("FULL PIPELINE (async, Guard + Classifier + Semantic + Structural + LLM)")
    cases = [
        ("", "hoi_dap_chung", "guard"),
        ("123", "hoi_dap_chung", "guard"),
        # Có thể semantic (prototype) hoặc structural (YAML) — không khóa method
        ("Điều 47 Luật Di sản văn hóa", "article_query", None),
        ("Tóm tắt Luật Đầu tư 2025", "tom_tat_van_ban", None),
        ("Chính sách bảo trợ xã hội đối với người cao tuổi là gì?", "giai_thich_quy_dinh", None),
        ("Trẻ em bị bố đánh đập hàng ngày, cần can thiệp ngay", "bao_ve_xa_hoi", None),
        ("Các ngành, nghề cấm đầu tư kinh doanh theo Luật Đầu tư 2025", "trich_xuat_van_ban", None),
        ("Karaoke gây ồn ào đến 2 giờ sáng, lập biên bản xử phạt thế nào?", "xu_ly_vi_pham_hanh_chinh", None),
        ("Xin chào", "hoi_dap_chung", None),
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
            print(f"  [{status}] \"{query[:55]}\" => {intent} ({conf}) via {method}{extra}")
        print(f"\n  Result: {passed}/{len(cases)} passed")
        return passed == len(cases)

    return asyncio.run(_run())


def test_backward_compat():
    _sep("BACKWARD COMPATIBILITY (detect_intent_rule_based)")
    cases = [
        ("Điều 6 Luật Đầu tư", "article_query"),
        ("Soạn công văn xin gia hạn giấy phép", "soan_thao_van_ban"),
        ("", "hoi_dap_chung"),
        ("Chính sách bảo trợ xã hội là gì?", None),  # may be semantic or fallback
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
        print(f"  [{'PASS' if ok else 'FAIL'}] \"{query[:55]}\" => {intent} ({conf})")

    print(f"\n  Result: {passed}/{len(cases)} passed")
    return passed == len(cases)


if __name__ == "__main__":
    all_ok = True
    all_ok &= test_exports()
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
