"""Tests for Legal Domain Classifier."""
import sys
import os

os.environ["PYTHONIOENCODING"] = "utf-8"
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.services.domain_classifier import (
    classify_query_domain,
    classify_document_domain,
    get_domain_filter_values,
    warmup_domain_index,
    LEGAL_DOMAINS,
    DOMAIN_PROTOTYPES,
)


def _sep(title: str):
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


def test_exports():
    _sep("EXPORT CHECKS")
    checks = [
        ("LEGAL_DOMAINS has 16 entries", len(LEGAL_DOMAINS) == 16),
        ("DOMAIN_PROTOTYPES has 16 entries", len(DOMAIN_PROTOTYPES) == 16),
        ("classify_query_domain callable", callable(classify_query_domain)),
        ("classify_document_domain callable", callable(classify_document_domain)),
        ("get_domain_filter_values callable", callable(get_domain_filter_values)),
    ]
    passed = 0
    for label, ok in checks:
        if ok:
            passed += 1
        print(f"  [{'PASS' if ok else 'FAIL'}] {label}")
    print(f"\n  Result: {passed}/{len(checks)} passed")
    return passed == len(checks)


def test_semantic_classification():
    _sep("SEMANTIC CLASSIFICATION (requires warmup)")

    print("  Building domain index...")
    warmup_domain_index()

    cases = [
        ("Bảo tồn di sản văn hóa phi vật thể", "van_hoa"),
        ("Xử phạt karaoke gây ồn", "xu_phat"),
        ("Chính sách bảo trợ xã hội cho người cao tuổi", "an_sinh"),
        ("Luật Đầu tư quy định ngành nghề cấm kinh doanh", "dau_tu"),
        ("Tổ chức Đại hội thể dục thể thao cấp xã", "the_thao"),
        ("Biển quảng cáo vi phạm kích thước trên đường", "quang_cao"),
        ("Đăng ký sinh hoạt tôn giáo tập trung", "ton_giao"),
        ("Tranh chấp ranh giới đất giữa hai hộ", "dat_dai"),
        ("Trẻ em bị bạo lực gia đình cần can thiệp", "an_sinh"),
        ("Xin chào", "chung"),
    ]

    passed = 0
    for query, expected_domain in cases:
        result = classify_query_domain(query, top_n=2)
        top_domain = result[0]["domain"]
        conf = result[0]["confidence"]
        ok = top_domain == expected_domain
        if ok:
            passed += 1
        print(f"  [{'PASS' if ok else 'FAIL'}] \"{query[:50]}\" => {top_domain} ({conf:.3f}) [expected: {expected_domain}]")

    print(f"\n  Result: {passed}/{len(cases)} passed")
    return passed >= len(cases) * 0.7


def test_document_classification():
    _sep("DOCUMENT CLASSIFICATION")

    cases = [
        ("Luật Di sản văn hóa 2001", "di sản văn hóa phi vật thể", "van_hoa"),
        ("Nghị định xử phạt vi phạm hành chính trong lĩnh vực văn hóa", "mức phạt tiền", "xu_phat"),
        ("Luật Đầu tư 2025", "ngành nghề cấm đầu tư kinh doanh", "dau_tu"),
        ("Luật Thể dục thể thao", "hoạt động thể thao quần chúng", "the_thao"),
        ("Luật Tín ngưỡng tôn giáo", "đăng ký sinh hoạt tôn giáo", "ton_giao"),
    ]

    passed = 0
    for title, snippet, expected in cases:
        result = classify_document_domain(title, snippet)
        ok = result == expected
        if ok:
            passed += 1
        print(f"  [{'PASS' if ok else 'FAIL'}] \"{title[:40]}\" => {result} [expected: {expected}]")

    print(f"\n  Result: {passed}/{len(cases)} passed")
    return passed >= len(cases) * 0.7


def test_filter_values():
    _sep("FILTER VALUES (get_domain_filter_values)")

    cases = [
        ("Bảo tồn di tích lịch sử", True),
        ("Mức phạt karaoke gây ồn", True),
        ("Xin chào bạn", False),
        ("", False),
    ]

    passed = 0
    for query, expect_filter in cases:
        result = get_domain_filter_values(query)
        has_filter = result is not None
        ok = has_filter == expect_filter
        if ok:
            passed += 1
        print(f"  [{'PASS' if ok else 'FAIL'}] \"{query[:40]}\" => {result} (filter={'yes' if has_filter else 'no'})")

    print(f"\n  Result: {passed}/{len(cases)} passed")
    return passed >= len(cases) * 0.7


if __name__ == "__main__":
    all_ok = True
    all_ok &= test_exports()
    all_ok &= test_semantic_classification()
    all_ok &= test_document_classification()
    all_ok &= test_filter_values()

    _sep("SUMMARY")
    if all_ok:
        print("  ALL TESTS PASSED")
    else:
        print("  SOME TESTS FAILED — check details above")
    sys.exit(0 if all_ok else 1)
