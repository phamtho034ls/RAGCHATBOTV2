"""Sanity checks for Intent Detector v4 — Zero-shot LLM Classification."""
import sys
import os
import asyncio

os.environ["PYTHONIOENCODING"] = "utf-8"
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.services.intent_detector import detect_intent, _detect_structural, VALID_INTENTS, COMMUNE_LEVEL_INTENTS


def _sep(title: str):
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


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
        ("Karaoke gây ồn", None, None),  # should NOT match structural
        ("Chính sách bảo trợ xã hội", None, None),  # should NOT match
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
        status = "PASS" if ok else "FAIL"
        if ok:
            passed += 1
        print(f"  [{status}] \"{query[:55]}\" => {label}")
    print(f"\n  Result: {passed}/{len(cases)} passed")
    return passed == len(cases)


def test_full_pipeline():
    _sep("FULL PIPELINE (async, with LLM)")
    cases = [
        ("", "hoi_dap_chung", "guard"),
        ("123", "hoi_dap_chung", "guard"),
        ("Điều 47 Luật Di sản văn hóa", "article_query", "structural"),
        ("Tóm tắt Luật Đầu tư 2025", "tom_tat_van_ban", "structural"),
        ("Chính sách bảo trợ xã hội đối với người cao tuổi là gì?", "giai_thich_quy_dinh", "llm"),
        ("Trẻ em bị bố đánh đập hàng ngày, cần can thiệp ngay", "bao_ve_xa_hoi", "llm"),
        ("Các ngành, nghề cấm đầu tư kinh doanh theo Luật Đầu tư 2025", "trich_xuat_van_ban", "llm"),
        ("Karaoke gây ồn ào đến 2 giờ sáng, lập biên bản xử phạt thế nào?", "xu_ly_vi_pham_hanh_chinh", "llm"),
        ("Xin chào", "hoi_dap_chung", "llm"),
    ]

    async def _run():
        passed = 0
        for query, expected_intent, expected_method in cases:
            result = await detect_intent(query)
            intent = result["intent"]
            conf = result["confidence"]
            method = result.get("method", "?")
            ok_intent = intent == expected_intent
            ok_method = method == expected_method
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


def test_exports():
    _sep("EXPORT CHECKS")
    checks = [
        ("VALID_INTENTS is list", isinstance(VALID_INTENTS, list)),
        ("23 intents", len(VALID_INTENTS) == 23),
        ("COMMUNE_LEVEL_INTENTS is set", isinstance(COMMUNE_LEVEL_INTENTS, set)),
        ("7 commune intents", len(COMMUNE_LEVEL_INTENTS) == 7),
        ("hoi_dap_chung in VALID_INTENTS", "hoi_dap_chung" in VALID_INTENTS),
        ("detect_intent_rule_based exists", callable(
            __import__("app.services.intent_detector", fromlist=["detect_intent_rule_based"]).detect_intent_rule_based
        )),
    ]
    passed = 0
    for label, ok in checks:
        if ok:
            passed += 1
        print(f"  [{'PASS' if ok else 'FAIL'}] {label}")
    print(f"\n  Result: {passed}/{len(checks)} passed")
    return passed == len(checks)


if __name__ == "__main__":
    all_ok = True
    all_ok &= test_exports()
    all_ok &= test_structural()
    all_ok &= test_full_pipeline()

    _sep("SUMMARY")
    if all_ok:
        print("  ALL TESTS PASSED")
    else:
        print("  SOME TESTS FAILED")
    sys.exit(0 if all_ok else 1)
