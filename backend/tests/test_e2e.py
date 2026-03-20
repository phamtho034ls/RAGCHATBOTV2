# -*- coding: utf-8 -*-
"""End-to-end test of the RAG pipeline with legal queries."""
import requests
import json

BASE_URL = "http://localhost:8000"

test_queries = [
    "Theo Luật Di sản văn hóa 2024, Điều 47 quy định gì?",
    "Theo Nghị định 38/2021/NĐ-CP, hành vi quảng cáo sai sự thật bị xử phạt thế nào?",
    "Theo Thông tư 13/2025/TT-BVHTTDL, báo cáo ngành văn hóa gồm những nội dung gì?",
]

for i, q in enumerate(test_queries, 1):
    print(f"\n{'='*80}")
    print(f"TEST {i}: {q}")
    print(f"{'='*80}")
    
    try:
        r = requests.post(f"{BASE_URL}/api/chat", json={
            "question": q,
            "temperature": 0.3,
        }, timeout=120)
        
        if r.status_code != 200:
            print(f"ERROR: status={r.status_code}, body={r.text[:500]}")
            continue
        
        data = r.json()
        answer = data.get("answer", "")
        sources = data.get("sources", [])
        intent = data.get("intent", "")
        confidence = data.get("confidence", 0)
        
        print(f"Intent: {intent} (confidence={confidence})")
        print(f"\nAnswer (first 800 chars):\n{answer[:800]}")
        print(f"\nSources ({len(sources)}):")
        for s in sources[:3]:
            meta = s.get("metadata", {})
            print(f"  - score={s.get('score', 0):.4f}, law={meta.get('law_name', 'N/A')}, article={meta.get('article_number', 'N/A')}")
    except Exception as e:
        print(f"ERROR: {e}")
