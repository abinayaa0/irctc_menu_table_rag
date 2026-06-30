# IRCTC Menu Table RAG — Terminal Pipeline

## Setup
```bash
pip install -r requirements.txt
cp .env.example .env
# Edit .env — optionally set OLLAMA_MODEL (default: gemma2:2b)
# Place South-Central.pdf in data/
```

## Usage

### Full pipeline (parse → build → query)
```bash
python main.py
```

### One-shot query
```bash
python main.py --query "I have Rs.60 what can I get?"
```

### Individual steps
```bash
python parse_pdf.py       # PDF → data/chunks.json (32 chunks)
python build_index.py     # chunks → Qdrant index
python query.py           # Start interactive terminal
python query.py --query "What's for breakfast on Set 3?"
```

## Features
- 32 chunks: 7 sets × 4 meal types + 4 overview chunks
- Hybrid retrieval: BGE-M3 dense + sparse → RRF fusion
- Cross-encoder reranking: bge-reranker-base
- Gemma 3 4B local generation (via transformers)
- Gradio web UI (app.py) or terminal (query.py)
- Qdrant local on-disk storage (no Docker)

## Menu Prices
| Meal            | Price  |
|-----------------|--------|
| Morning Tea     | Rs.15  |
| Breakfast       | Rs.65  |
| Evening Snacks  | Rs.50  |
| Lunch & Dinner  | Rs.120 |
