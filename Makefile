# RAG Chatbot — chạy từ thư mục gốc repo (có Makefile này)
.PHONY: test eval eval-comprehensive build-gold-comprehensive all

BACKEND = backend

test:
	cd $(BACKEND) && python -m pytest tests/ -q --tb=short 2>nul || python -m pytest tests/ -q --tb=short

eval:
	cd $(BACKEND) && python -m pytest tests/evaluation/ -q --tb=short 2>nul || python -m pytest tests/evaluation/ -q --tb=short

# Đánh giá retrieval + E2E + A/B (Postgres, Qdrant, OpenAI). Ghi tests/evaluation/results/comprehensive_eval.*
eval-comprehensive:
	cd $(BACKEND) && python tests/evaluation/eval_comprehensive.py

# 500 câu gold từ data_clean.json (thư mục gốc repo)
build-gold-comprehensive:
	cd $(BACKEND) && python tests/evaluation/build_gold_comprehensive.py --count 500

all: test eval
