# IRCTC Menu Table RAG

A Retrieval-Augmented Generation (RAG) system for querying the IRCTC South Central Zone food menu served on Rajdhani and Duronto trains in Sleeper Class (SL). The system parses a structured menu PDF, indexes it into a local vector store, and answers natural language queries using hybrid retrieval and a locally-hosted language model.

---

## Overview

Indian Railways (IRCTC) serves a 7-day rotating meal menu on select express trains. This project builds an end-to-end RAG pipeline over that menu, enabling passengers to query meal availability, prices, and dietary options without reading through the full document.

**Supported queries include:**
- What is served for breakfast on Set 3?
- Which sets include paneer in lunch?
- I have Rs.60 — what can I get?
- Is there a Jain meal option available?
- What is common across all evening snack sets?

---

## Architecture

```
South-Central.pdf
      |
      v
 parse_pdf_llm.py        -- PDF extraction and LLM-assisted structured chunking
      |
      v
 data/chunks_llm.json    -- 26 structured chunks (7 sets x meal types + overviews)
      |
      v
 build_index_llm.py      -- BGE-M3 dense + sparse embedding, Qdrant indexing
      |
      v
 qdrant_local/           -- On-disk Qdrant vector store (no Docker required)
      |
      v
 query_llm.py            -- Hybrid search (RRF), cross-encoder reranking, LLM generation
      |
      v
 app_llm.py              -- Gradio web UI (optional)
```

**Retrieval pipeline:**
1. Query is encoded with BGE-M3 (dense + sparse vectors)
2. Hybrid search using Reciprocal Rank Fusion (RRF) over Qdrant
3. Candidates are reranked using `bge-reranker-base`
4. Top chunks are passed to a locally-hosted LLM via Ollama (OpenAI-compatible API)
5. The LLM generates a grounded answer constrained to the retrieved context

---

## Menu Structure

| Meal Type       | Price (incl. tax) | Sets     |
|-----------------|-------------------|----------|
| Morning Tea     | Rs. 15            | 1 to 7   |
| Breakfast       | Rs. 65            | 1 to 7   |
| Evening Snacks  | Rs. 50            | 1 to 7   |
| Lunch & Dinner  | Rs. 120           | 1 to 7   |

- 7 rotating sets served cyclically across journeys
- Jain and diabetic meals available on request from train staff
- Ready-made Masala Tea available on demand

---

## Requirements

- Python 3.10 or later
- [Ollama](https://ollama.com) running locally with a compatible model (default: `gemma2:2b`)
- ~4 GB disk space for model weights (BGE-M3 + reranker, downloaded on first run)

### Python dependencies

```
pdfplumber==0.11.*
qdrant-client==1.9.*
FlagEmbedding==1.2.*
python-dotenv==1.0.*
rich==13.*
gradio
openai
```

---

## Setup

**1. Clone the repository**

```bash
git clone https://github.com/abinayaa0/irctc_menu_rag.git
cd irctc_menu_rag/irctc-menu-table-rag
```

**2. Create and activate a virtual environment**

```bash
python -m venv rag_env
# Windows
rag_env\Scripts\activate
# macOS / Linux
source rag_env/bin/activate
```

**3. Install dependencies**

```bash
pip install -r requirements.txt
```

**4. Configure environment variables**

```bash
cp .env.example .env
```

Edit `.env` to set your Ollama model and base URL:

```
OLLAMA_MODEL=gemma2:2b
OLLAMA_BASE_URL=http://localhost:11434/v1
QDRANT_PATH=./qdrant_local
```

**5. Place the source PDF**

Copy the IRCTC South Central menu PDF into the data directory:

```
data/South-Central.pdf
```

**6. Start Ollama**

```bash
ollama serve
ollama pull gemma2:2b
```

---

## Usage

### Full pipeline — parse, index, then query interactively

```bash
python main.py
```

### One-shot query

```bash
python main.py --query "What is served for breakfast on Set 3?"
```

### Run individual steps

```bash
# Step 1: Parse PDF into structured chunks
python parse_pdf_llm.py

# Step 2: Build Qdrant vector index
python build_index_llm.py

# Step 3: Start interactive terminal assistant
python query_llm.py

# Step 3 (single query): Run one query and exit
python query_llm.py --query "I have Rs.60, what can I get?"
```

### Launch the Gradio web UI

```bash
python app_llm.py
```

The web interface will be available at `http://localhost:7860`.

---

## Project Structure

```
irctc-menu-table-rag/
├── data/
│   ├── South-Central.pdf        # Source menu document
│   ├── chunks.json              # Chunks from rule-based parser
│   ├── chunks_llm.json          # Chunks from LLM-assisted parser
│   └── menu_descriptions.md     # Human-readable menu reference
├── app.py                       # Gradio UI (rule-based pipeline)
├── app_llm.py                   # Gradio UI (LLM pipeline)
├── build_index.py               # Qdrant indexer (rule-based)
├── build_index_llm.py           # Qdrant indexer (LLM chunking)
├── main.py                      # Unified pipeline entry point
├── parse_pdf.py                 # Rule-based PDF parser
├── parse_pdf_llm.py             # LLM-assisted PDF parser
├── query.py                     # Terminal query loop (rule-based)
├── query_llm.py                 # Terminal query loop (LLM)
├── check_spelling.py            # Menu spelling validator
├── test_handlers.py             # Unit tests for query handlers
├── requirements.txt
├── .env.example
└── README.md
```

---

## Notebooks

Three Jupyter notebooks are included for experimentation and Colab compatibility:

| Notebook | Description |
|---|---|
| `irctc_sc_rag.ipynb` | Rule-based RAG pipeline |
| `irctc_sc_rag_v2.ipynb` | Improved rule-based pipeline with hybrid retrieval |
| `irctc_sc_rag_llm_colab.ipynb` | LLM-assisted pipeline, Colab-ready |

---

## Technical Notes

- **Embeddings**: `BAAI/bge-m3` — multi-lingual, supports both dense and sparse (lexical) representations
- **Reranker**: `BAAI/bge-reranker-base` — cross-encoder reranking for precision
- **Vector store**: Qdrant (local on-disk, no server or Docker required)
- **Fusion**: Reciprocal Rank Fusion (RRF, k=60) combines dense and sparse rankings
- **LLM**: Any OpenAI-compatible model served via Ollama (default: `gemma2:2b`)
- **Retrieval routing**: Smart routing — pinpoint hybrid search for specific set+meal queries, full scroll for general queries
- **Reranker threshold**: Low-confidence chunks are filtered to reduce hallucination in smaller models

---

## License

This project is released for educational and research purposes. The IRCTC menu data is sourced from publicly available railway documents.
