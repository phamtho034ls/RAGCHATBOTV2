# RAG Chatbot — chạy từ thư mục gốc repo (có Makefile này)
.PHONY: test eval all

BACKEND = backend

test:
	cd $(BACKEND) && python -m pytest tests/ -q --tb=short 2>nul || python -m pytest tests/ -q --tb=short

eval:
	cd $(BACKEND) && python -m pytest tests/evaluation/ -q --tb=short 2>nul || python -m pytest tests/evaluation/ -q --tb=short

all: test eval
