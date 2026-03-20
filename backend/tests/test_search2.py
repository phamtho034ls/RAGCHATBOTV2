# -*- coding: utf-8 -*-
"""Debug search via API with different filter combinations."""
import requests
import json

BASE_URL = "http://localhost:8000"

q = "Theo Luật Di sản văn hóa 2024, Điều 47 quy định gì?"

# Test 1: Search with article_number filter
print("=== Search with article_number=47 filter ===")
r = requests.post(f"{BASE_URL}/api/search", json={
    "query": q,
    "top_k": 10,
    "filters": {"article_number": "47"}
}, timeout=60)
data = r.json()
results = data.get("results", [])
print(f"Results: {len(results)}")
for s in results[:5]:
    meta = s.get("metadata", {})
    print(f"  score={s.get('score', 0):.4f}, article={meta.get('article_number')}, law={meta.get('law_name', 'N/A')}")
    print(f"  text[:200]: {s.get('content', '')[:200]}")
    print()

# Test 2: Search without filters
print("\n=== Search without filters ===")
r = requests.post(f"{BASE_URL}/api/search", json={"query": q, "top_k": 10}, timeout=60)
data = r.json()
results = data.get("results", [])
print(f"Results: {len(results)}")
for s in results[:5]:
    meta = s.get("metadata", {})
    print(f"  score={s.get('score', 0):.4f}, article={meta.get('article_number')}, law={meta.get('law_name', 'N/A')}")
    
# Test 3: Search with document_type+year filter only
print("\n=== Search with document_type=luat, year=2024 ===")
r = requests.post(f"{BASE_URL}/api/search", json={
    "query": q,
    "top_k": 10,
    "filters": {"document_type": "luat", "year": 2024}
}, timeout=60)
data = r.json()
results = data.get("results", [])
print(f"Results: {len(results)}")
for s in results[:5]:
    meta = s.get("metadata", {})
    print(f"  score={s.get('score', 0):.4f}, article={meta.get('article_number')}, law={meta.get('law_name', 'N/A')}")
