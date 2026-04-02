# -*- coding: utf-8 -*-
"""Diagnose pooling method for phobert_multitask_a100.pt."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.environ["INTENT_MODEL_ENABLED"] = "true"

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoTokenizer, RobertaModel

PT_PATH = "app/intent_model/phobert_multitask_a100.pt"
HF_NAME = "vinai/phobert-base"
LABELS = ["admin_scenario","comparison","document_generation","legal_explanation",
          "legal_lookup","procedure","summarization","violation"]

print("Loading tokenizer + config...")
tokenizer = AutoTokenizer.from_pretrained(HF_NAME, use_fast=False)
cfg = AutoConfig.from_pretrained(HF_NAME)

print("Loading checkpoint...")
ckpt = torch.load(PT_PATH, map_location="cpu", weights_only=False)

q = "Điều 6 Luật Đầu tư 2025 quy định gì?"
enc = tokenizer(q, return_tensors="pt", truncation=True, max_length=256, padding=False)

intent_w = ckpt["intent_head.weight"]  # [8, 768]
intent_b = ckpt["intent_head.bias"]

# Rebuild backbone keys: strip "phobert." prefix → "roberta." or check what HF uses
# PhoBERT uses "roberta" module name internally in HF
backbone_ckpt = {k.replace("phobert.", ""): v for k, v in ckpt.items()
                 if k.startswith("phobert.")}

# Test approach 1: Use pooler_output
print("\n--- Approach 1: pooler_output ---")
model1 = RobertaModel(cfg, add_pooling_layer=True)
missing, unexpected = model1.load_state_dict(backbone_ckpt, strict=False)
print(f"  missing={len(missing)} unexpected={len(unexpected)}")
if missing: print("  missing keys:", missing[:5])
model1.eval()
with torch.no_grad():
    out = model1(**enc)
    pooled = out.pooler_output
    logits = pooled @ intent_w.T + intent_b
    probs = torch.softmax(logits, dim=-1).squeeze(0)
    top = int(probs.argmax())
    print(f"  top intent: {LABELS[top]} conf={float(probs[top]):.3f}")
    sorted_p = sorted(zip(LABELS, probs.tolist()), key=lambda x: -x[1])
    for label, p in sorted_p[:4]:
        print(f"    {label}: {p:.3f}")

# Test approach 2: CLS token (last_hidden_state[:, 0, :])
print("\n--- Approach 2: CLS from last_hidden_state ---")
model2 = RobertaModel(cfg, add_pooling_layer=False)
missing2, unexpected2 = model2.load_state_dict(backbone_ckpt, strict=False)
print(f"  missing={len(missing2)} unexpected={len(unexpected2)}")
model2.eval()
with torch.no_grad():
    out2 = model2(**enc)
    cls_emb = out2.last_hidden_state[:, 0, :]
    logits2 = cls_emb @ intent_w.T + intent_b
    probs2 = torch.softmax(logits2, dim=-1).squeeze(0)
    top2 = int(probs2.argmax())
    print(f"  top intent: {LABELS[top2]} conf={float(probs2[top2]):.3f}")
    sorted_p2 = sorted(zip(LABELS, probs2.tolist()), key=lambda x: -x[1])
    for label, p in sorted_p2[:4]:
        print(f"    {label}: {p:.3f}")

# Test approach 3: mean pooling
print("\n--- Approach 3: mean pooling ---")
with torch.no_grad():
    out3 = model2(**enc, output_hidden_states=False)
    mask = enc["attention_mask"].unsqueeze(-1).float()
    mean_emb = (out3.last_hidden_state * mask).sum(1) / mask.sum(1)
    logits3 = mean_emb @ intent_w.T + intent_b
    probs3 = torch.softmax(logits3, dim=-1).squeeze(0)
    top3 = int(probs3.argmax())
    print(f"  top intent: {LABELS[top3]} conf={float(probs3[top3]):.3f}")
    sorted_p3 = sorted(zip(LABELS, probs3.tolist()), key=lambda x: -x[1])
    for label, p in sorted_p3[:4]:
        print(f"    {label}: {p:.3f}")

# Now test all approaches with multiple queries
print("\n=== MULTI QUERY TEST ===")
test_queries = [
    ("Điều 6 Luật Đầu tư 2025 quy định gì?", "legal_lookup"),
    ("Karaoke gây ồn quá giờ, xử phạt thế nào?", "violation"),
    ("Soạn công văn xin gia hạn giấy phép", "document_generation"),
    ("Tóm tắt Luật Đầu tư 2025", "summarization"),
    ("So sánh Luật Đầu tư 2020 và 2025", "comparison"),
    ("Xin chào", "legal_explanation"),
]
for q, expected in test_queries:
    enc_q = tokenizer(q, return_tensors="pt", truncation=True, max_length=256, padding=False)
    with torch.no_grad():
        o = model1(**enc_q)
        p1 = torch.softmax(o.pooler_output @ intent_w.T + intent_b, dim=-1).squeeze(0)
        o2 = model2(**enc_q)
        p2 = torch.softmax(o2.last_hidden_state[:, 0, :] @ intent_w.T + intent_b, dim=-1).squeeze(0)
    top1 = LABELS[int(p1.argmax())]
    top2 = LABELS[int(p2.argmax())]
    ok1 = "OK" if top1 == expected else "!!"
    ok2 = "OK" if top2 == expected else "!!"
    print(f"  [{ok1}] pooler={top1}({float(p1.max()):.3f})  [{ok2}] cls={top2}({float(p2.max()):.3f})  expected={expected}")
    print(f"    {q[:60]}")
