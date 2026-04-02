# -*- coding: utf-8 -*-
"""Kiểm tra nhanh multitask model inference (CLS pooling, corrected label order)."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.environ["INTENT_MODEL_ENABLED"] = "true"

import torch
from app.services.intent_model_classifier import (
    _ensure_loaded, classify_multitask_sync, MULTITASK_INTENT_LABELS, FLAGS_ORDER
)

_ensure_loaded()

from app.services import intent_model_classifier as imc

print(f"Label order: {MULTITASK_INTENT_LABELS}")
print()

test_cases = [
    ("Điều 6 Luật Đầu tư 2025 quy định gì?", "legal_lookup"),
    ("Karaoke gây ồn quá giờ, xử phạt thế nào?", "violation"),
    ("Soạn công văn xin gia hạn giấy phép", "document_generation"),
    ("Tóm tắt Luật Đầu tư 2025", "summarization"),
    ("Thủ tục xin cấp phép lễ hội cấp xã", "procedure"),
    ("So sánh Luật Đầu tư 2020 và 2025", "comparison"),
    ("Xây dựng kế hoạch quản lý di tích năm 2025", "admin_scenario"),
    ("Xin chào", "legal_explanation"),
    ("Danh sách ngành nghề cấm đầu tư theo Luật Đầu tư", "legal_lookup"),
    ("Lập kế hoạch kiểm tra đột xuất cơ sở karaoke", "violation"),
    ("Ai ký ban hành Nghị định 36/2019?", "admin_scenario"),
    ("Hòa giải tranh chấp đất giữa hai hàng xóm", "admin_scenario"),
    ("Hồ sơ của tôi đã đủ chưa?", "procedure"),
    ("Lập báo cáo tổng kết hoạt động văn hóa", "document_generation"),
]

passed = 0
total = 0
for q, expected in test_cases:
    enc = imc._tokenizer(q, return_tensors="pt", truncation=True, max_length=256, padding=False)
    enc = {k: v.to(imc._device) for k, v in enc.items()}
    with torch.no_grad():
        il, fl = imc._model(**enc)
        probs = torch.softmax(il, dim=-1).squeeze(0)
        flag_probs = torch.sigmoid(fl).squeeze(0)

    conf = float(probs.max())
    top_idx = int(probs.argmax())
    top_intent = MULTITASK_INTENT_LABELS[top_idx]
    flags = {FLAGS_ORDER[i]: bool(flag_probs[i].item() >= 0.5) for i in range(len(FLAGS_ORDER))}

    ok = top_intent == expected
    total += 1
    if ok:
        passed += 1
    status = "PASS" if ok else "FAIL"
    print(f"[{status}] {top_intent}({conf:.3f}) expected={expected}")
    print(f"  {q}")
    if not ok:
        top3 = sorted(zip(MULTITASK_INTENT_LABELS, probs.tolist()), key=lambda x: -x[1])[:3]
        print(f"  top3: {[(l, f'{p:.3f}') for l, p in top3]}")
    print(f"  flags: {flags}")
    print()

print(f"SCORE: {passed}/{total} ({100*passed/total:.0f}%)")
