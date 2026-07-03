## IRCTC Menu Table RAG

A **Hybrid Retrieval-Augmented Generation (RAG)** system for querying the **IRCTC South Central Zone food menu** served on Rajdhani and Duronto trains in **Sleeper Class (SL)**.

The system parses a structured menu PDF, indexes it into a local vector store, and answers natural language questions using **hybrid retrieval**, **cross-encoder reranking**, and a **locally hosted LLM**.

---

## Overview

Indian Railways (IRCTC) serves a **7-day rotating meal menu** on select express trains.

This project builds an end-to-end RAG pipeline over that menu, allowing passengers to ask questions about meal availability, prices, and dietary options without manually browsing the PDF.

**Official Menu Website:**  
https://menurates.irctc.co.in/

### Example Queries

- What is served for breakfast on Set 3?
- Which lunch sets include paneer?
- I have Rs.60 — what can I get?
- Is there a Jain meal option available?
- What is common across all evening snack sets?

---

## Architecture

```text
South-Central.pdf
        │
        ▼
parse_pdf_llm.py
        │
        ▼
data/chunks_llm.json
        │
        ▼
build_index_llm.py
        │
        ▼
qdrant_local/
        │
        ▼
query_llm.py
        │
        ▼
app_llm.py (Gradio UI)
```

### Retrieval Pipeline

1. Encode the user query using **BGE-M3** (dense + sparse embeddings).
2. Perform hybrid retrieval in **Qdrant**.
3. Fuse dense and sparse rankings using **Reciprocal Rank Fusion (RRF)**.
4. Rerank retrieved chunks using **bge-reranker-base**.
5. Send the highest-ranked chunks to a locally hosted LLM through **Ollama**.
6. Generate a grounded answer constrained to the retrieved context.

---

## Menu Structure

| Meal | Price (Inclusive of Tax) | Available Sets |
|------|--------------------------:|---------------|
| Morning Tea | Rs.15 | 1–7 |
| Breakfast | Rs.65 | 1–7 |
| Evening Snacks | Rs.50 | 1–7 |
| Lunch & Dinner | Rs.120 | 1–7 |

### Additional Information

- 7 rotating meal sets
- Jain meals available on request
- Diabetic meals available on request
- Ready-made Masala Tea available on demand

---

## Requirements

- Python 3.10+
- Ollama
- Approximately 4 GB of disk space

### Python Dependencies

```text
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

### 1. Clone the Repository

```bash
git clone https://github.com/abinayaa0/irctc_menu_rag.git
cd irctc_menu_rag/irctc-menu-table-rag
```

### 2. Create and Activate a Virtual Environment

**Windows**

```bash
python -m venv rag_env
rag_env\Scripts\activate
```

**macOS / Linux**

```bash
python -m venv rag_env
source rag_env/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables

Copy the example environment file:

```bash
cp .env.example .env
```

Edit `.env`:

```env
OLLAMA_MODEL=gemma2:2b
OLLAMA_BASE_URL=http://localhost:11434/v1
QDRANT_PATH=./qdrant_local
```

### 5. Add the Source PDF

Place the IRCTC South Central menu PDF in:

```text
data/South-Central.pdf
```

### 6. Start Ollama

```bash
ollama serve
ollama pull gemma2:2b
```

---

## Usage

### Run the Complete Pipeline

```bash
python main.py
```

### One-Shot Query

```bash
python main.py --query "What is served for breakfast on Set 3?"
```

### Run Individual Components

#### Parse the PDF

```bash
python parse_pdf_llm.py
```

#### Build the Vector Index

```bash
python build_index_llm.py
```

#### Interactive Terminal Assistant

```bash
python query_llm.py
```

#### Single Query

```bash
python query_llm.py --query "I have Rs.60, what can I get?"
```

### Launch the Gradio Web UI

```bash
python app_llm.py
```

The interface will be available at:

```text
http://localhost:7860
```

---

## Project Structure

```text
irctc-menu-table-rag/
├── data/
│   ├── South-Central.pdf        # Source menu document
│   ├── chunks.json              # Rule-based parser output
│   ├── chunks_llm.json          # LLM-assisted parser output
│   └── menu_descriptions.md     # Human-readable menu reference
├── app.py                       # Rule-based Gradio UI
├── app_llm.py                   # LLM-based Gradio UI
├── build_index.py               # Rule-based index builder
├── build_index_llm.py           # LLM index builder
├── main.py                      # Unified pipeline entry point
├── parse_pdf.py                 # Rule-based parser
├── parse_pdf_llm.py             # LLM-assisted parser
├── query.py                     # Rule-based terminal assistant
├── query_llm.py                 # LLM terminal assistant
├── check_spelling.py            # Menu spelling validator
├── test_handlers.py             # Unit tests
├── requirements.txt
├── .env.example
└── README.md
```

---

## Notebooks

| Notebook | Description |
|-----------|-------------|
| `irctc_sc_rag.ipynb` | Rule-based RAG pipeline |
| `irctc_sc_rag_v2.ipynb` | Hybrid retrieval pipeline |
| `irctc_sc_rag_llm_colab.ipynb` | LLM-assisted pipeline for Google Colab |

---

## Technical Details

| Component | Technology |
|-----------|------------|
| Embeddings | BAAI/bge-m3 |
| Reranker | BAAI/bge-reranker-base |
| Vector Database | Qdrant |
| Retrieval | Hybrid Dense + Sparse |
| Fusion | Reciprocal Rank Fusion (RRF, k=60) |
| LLM | Ollama (OpenAI-compatible) |
| Default Model | gemma2:2b |
| Query Routing | Smart routing (hybrid search or full scroll) |
| Hallucination Reduction | Low-confidence reranker threshold |

---

## License

This project is intended for **educational and research purposes**.

The IRCTC menu data is sourced from publicly available railway documents.
