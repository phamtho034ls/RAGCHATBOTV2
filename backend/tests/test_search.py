# -*- coding: utf-8 -*-
"""Debug search for Test 1 query."""
import requests
import json

BASE_URL = "http://localhost:8000"

# Test search directly
q = "Theo Luật Di sản văn hóa 2024, Điều 47 quy định gì?"
r = requests.post(f"{BASE_URL}/api/search", json={"query": q, "top_k": 10}, timeout=60)
print(f"Search status: {r.status_code}")
data = r.json()
results = data.get("results", [])
print(f"Results: {len(results)}")
for s in results[:5]:
    meta = s.get("metadata", {})
    print(f"  score={s.get('score', 0):.4f}, law={meta.get('law_name', 'N/A')}, article={meta.get('article_number', 'N/A')}")
    print(f"  text[:150]: {s.get('content', '')[:150]}")
    print()

# Test query analysis
r2 = requests.post(f"{BASE_URL}/api/query/analyze", json={"question": q}, timeout=30)
print(f"\nQuery analysis: {json.dumps(r2.json(), ensure_ascii=False, indent=2)}")
