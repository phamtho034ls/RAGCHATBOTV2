# -*- coding: utf-8 -*-
"""Debug: check if BM25 finds Dieu 47."""
import requests
import json

BASE_URL = "http://localhost:8000"

# Search with a direct article reference
q = "Điều 47 bảo quản phục chế di vật cổ vật bảo vật quốc gia Luật Di sản văn hóa"
print(f"=== Direct query: {q} ===")
r = requests.post(f"{BASE_URL}/api/search", json={"query": q, "top_k": 10}, timeout=60)
data = r.json()
results = data.get("results", [])
print(f"Results: {len(results)}")
for s in results[:10]:
    meta = s.get("metadata", {})
    print(f"  score={s.get('score', 0):.4f}, article={meta.get('article_number')}, law={meta.get('law_name', 'N/A')}")
