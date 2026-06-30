"""
IRCTC South Central Menu RAG — Simple Gradio UI.
"""

import os
import sys
import time
import re

os.environ["USE_TF"] = "0"
os.environ["USE_TORCH"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["HF_HUB_OFFLINE"] = "1"

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from pathlib import Path
from dotenv import load_dotenv
import gradio as gr
from qdrant_client import QdrantClient
from qdrant_client.models import NamedVector, NamedSparseVector, SparseVector
from FlagEmbedding import BGEM3FlagModel, FlagReranker
from openai import OpenAI

load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
COLLECTION = "irctc_sc_menu"
QDRANT_PATH = os.getenv("QDRANT_PATH", os.path.join(BASE_DIR, "qdrant_local"))
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma2:2b")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
TOP_K_RETRIEVE = 20
TOP_K_RERANK = 15
RRF_K = 60

PRICE_TABLE = {"Morning Tea": 15, "Evening Snacks": 50, "Breakfast": 65, "Lunch & Dinner": 120}

SYSTEM_PROMPT = """You are an assistant for IRCTC Rajdhani/Duronto South Central Zone
Sleeper Class food menu queries.

RULES:
1. Answer ONLY using explicitly stated facts from the context. Do not infer, assume, or generalize.
2. Never state that an item is available in all sets unless the context explicitly says so. If an item appears only in specific sets, list those sets explicitly.
3. Prices: Morning Tea Rs.15 | Breakfast Rs.65 | Evening Snacks Rs.50 |
   Lunch & Dinner Rs.120 — all inclusive of taxes. Never invent items or prices.
4. There are 7 rotating sets (Set 1-7) served cyclically across journeys.
5. For specific set questions, give that set's items exactly from context. If a set is not present in the context, do not assume or invent its items.
6. For general questions (e.g. "what's for breakfast"), summarise all 7 sets.
7. Mention Jain/Diabetic food on request only when the question asks about
   dietary restrictions, special meals, or food options generally.
   Do NOT append it to every answer.
8. Mention Masala Tea on demand only when the question is about tea or
   beverages. Do NOT append it to every answer.
9. For budget/price questions (e.g. "I have Rs.60"), list ALL meal types that
   fit within that budget using the prices above. With Rs.60 the passenger can
   afford Morning Tea (Rs.15) and/or Evening Snacks (Rs.50), but not Breakfast
   (Rs.65) or Lunch & Dinner (Rs.120). Also mention what they can get for
   their specific budget and suggest combinations.
10. Be concise. No filler. Lead with the direct answer.
11. When answering about specific food items, always mention the meal category (e.g., Morning Tea, Breakfast, Evening Snacks, Lunch & Dinner) and price in which they are served.
12. Only say "I don't have that information" if the question is completely
    unrelated to the IRCTC food menu (e.g. train schedules, tickets), or if it asks about a set or item that is not present in the menu. """

BOT_INFO_TEXT = """This chatbot is the IRCTC South Central Zone Menu Assistant.
It helps passengers in Sleeper Class (SL) on Rajdhani and Duronto trains look up the 7-day rotating food menu, prices, and options.
It can answer questions about:
- What is served for Morning Tea (Rs.15), Breakfast (Rs.65), Evening Snacks (Rs.50), and Lunch/Dinner (Rs.120).
- Which dishes are served on specific menu sets (Set 1 to Set 7).
- Pricing and budget questions (e.g., what meals can be bought for a given price).
- Special meal options like Jain/diabetic food (available on request) and Ready Made Masala Tea (available on demand)."""

SAMPLE_QUESTIONS = [
    "What is served for breakfast on Set 3?",
    "I have Rs.50 what can I get?",
    "Is there a non-veg option for lunch?",
    "My mother is Jain and diabetic, what options does she have?",
    "What dal is served in Set 5 lunch?",
    "Tell me about Masala Tea",
]


def init_llm():
    return OpenAI(base_url=OLLAMA_BASE_URL, api_key="ollama")


def init_models():
    embedder = BGEM3FlagModel("BAAI/bge-m3", use_fp16=False, device="cpu")
    try:
        reranker = FlagReranker("BAAI/bge-reranker-base", use_fp16=False, device="cpu")
    except Exception:
        reranker = None
    return embedder, reranker


def init_qdrant():
    lock_file = os.path.join(QDRANT_PATH, ".lock")
    if os.path.exists(lock_file):
        try:
            os.remove(lock_file)
        except Exception:
            pass
    return QdrantClient(path=QDRANT_PATH)


def embed_query(model, query_text):
    output = model.encode(
        [query_text],
        return_dense=True,
        return_sparse=True,
        return_colbert_vecs=False,
        max_length=256,
    )
    dense = output["dense_vecs"][0].tolist()
    d = output["lexical_weights"][0]
    return {
        "dense": dense,
        "sparse_indices": [int(k) for k in d.keys()],
        "sparse_values": [float(v) for v in d.values()],
    }


def hybrid_search(client, query_emb, query_filter=None, top_k=TOP_K_RETRIEVE):
    dense_hits = client.search(
        collection_name=COLLECTION,
        query_vector=NamedVector(name="dense", vector=query_emb["dense"]),
        query_filter=query_filter,
        limit=top_k, with_payload=True,
    )
    sparse_hits = client.search(
        collection_name=COLLECTION,
        query_vector=NamedSparseVector(
            name="sparse",
            vector=SparseVector(indices=query_emb["sparse_indices"], values=query_emb["sparse_values"]),
        ),
        query_filter=query_filter,
        limit=top_k, with_payload=True,
    )
    chunk_map = {}
    for rank, hit in enumerate(dense_hits, 1):
        pid = str(hit.id)
        if pid not in chunk_map:
            chunk_map[pid] = {"id": pid, "text": hit.payload.get("chunk_text", ""), "payload": hit.payload, "rrf": 0.0}
        chunk_map[pid]["rrf"] += 1 / (RRF_K + rank)
    for rank, hit in enumerate(sparse_hits, 1):
        pid = str(hit.id)
        if pid not in chunk_map:
            chunk_map[pid] = {"id": pid, "text": hit.payload.get("chunk_text", ""), "payload": hit.payload, "rrf": 0.0}
        chunk_map[pid]["rrf"] += 1 / (RRF_K + rank)
    chunks = list(chunk_map.values())
    chunks.sort(key=lambda c: c["rrf"], reverse=True)
    return chunks[:top_k]


def rerank_chunks(reranker, query_text, chunks, top_n=TOP_K_RERANK):
    if not chunks:
        return []
    if reranker is None:
        for c in chunks:
            c["score"] = c["rrf"]
            c["score_label"] = f"RRF {c['rrf']:.4f}"
        return chunks[:top_n]
    pairs = [[query_text, c["text"]] for c in chunks]
    scores = reranker.compute_score(pairs, normalize=True)
    for i, c in enumerate(chunks):
        c["score"] = float(scores[i])
        c["score_label"] = f"CE {c['score']:.4f} | RRF {c['rrf']:.4f}"
    chunks.sort(key=lambda c: c["score"], reverse=True)
    
    # Filter out low-scoring chunks to prevent conflation/hallucination in small LLMs
    if chunks:
        top_score = chunks[0].get("score", 0)
        if top_score >= 0.3:
            threshold = top_score * 0.5
            chunks = [c for c in chunks if c.get("score", 0) >= threshold]
        else:
            chunks = [c for c in chunks if c.get("score", 0) >= 0.01]
        
    return chunks[:top_n]


def build_prompt(query_text, chunks):
    blocks = []
    for i, c in enumerate(chunks, 1):
        p = c["payload"]
        blocks.append(
            f"[{i}] {p.get('meal_type','?')} | "
            f"Set {p.get('set_number','Overview')} | "
            f"Rs.{p.get('price','?')}\n{c['text']}"
        )
    return (
        SYSTEM_PROMPT + "\n\n--- RETRIEVED CONTEXT ---\n"
        + "\n\n".join(blocks)
        + "\n--- END CONTEXT ---\n\n"
        f"Question: {query_text}\n\nAnswer:"
    )


def generate_answer(prompt_text):
    client = init_llm()
    response = client.chat.completions.create(
        model=OLLAMA_MODEL,
        messages=[{"role": "user", "content": prompt_text}],
        temperature=0.1,
        max_tokens=300,
    )
    return response.choices[0].message.content.strip()


def handle_special_queries(query_text):
    q = query_text.lower()

    # Check for invalid set numbers (outside 1-7)
    set_nums = [int(n) for n in re.findall(r"\bset\s*(\d+)\b", query_text, re.IGNORECASE)]
    invalid_sets = [n for n in set_nums if n < 1 or n > 7]
    if invalid_sets:
        if len(invalid_sets) == 1:
            return f"There are only 7 rotating sets (Set 1 to Set 7) on the menu. Set {invalid_sets[0]} does not exist."
        else:
            sets_str = ", ".join(f"Set {n}" for n in invalid_sets)
            return f"There are only 7 rotating sets (Set 1 to Set 7) on the menu. {sets_str} do not exist."

    if re.search(r"\b(jain|diabetic|special meal|dietary restriction)\b", q):
        return (
            "Jain and diabetic food can be provided on request. "
            "The menu document does not specify the exact items available for these options. "
            "Please request it from the train staff when your meal is served."
        )



    m = re.search(r"rs\.?\s*(\d+)", q)
    if m:
        budget = int(m.group(1))
        
        # Check if there is a specific time/meal restriction in the query
        has_morning = any(k in q for k in ["morning", "breakfast", "am"]) and not any(k in q for k in ["evening", "snacks", "pm", "dinner"])
        has_evening = any(k in q for k in ["evening", "snacks", "pm"]) and not any(k in q for k in ["morning", "breakfast", "am", "lunch"])
        has_lunch_dinner = any(k in q for k in ["lunch", "dinner", "night", "noon", "midday"])
        
        candidates = list(PRICE_TABLE.items())
        if has_morning:
            candidates = [(name, price) for name, price in candidates if name in ["Morning Tea", "Breakfast"]]
        elif has_evening:
            candidates = [(name, price) for name, price in candidates if name in ["Evening Snacks"]]
        elif has_lunch_dinner:
            candidates = [(name, price) for name, price in candidates if name in ["Lunch & Dinner"]]
            
        affordable = [(name, price) for name, price in sorted(candidates, key=lambda x: x[1]) if price <= budget]
        if affordable:
            lines = [f"With Rs.{budget} you can afford:"]
            for name, price in affordable:
                lines.append(f"  - {name} (Rs.{price})")
            lines.append("")
            
            if has_morning:
                if budget < PRICE_TABLE["Breakfast"]:
                    lines.append("Note: This is only enough for Morning Tea.")
                else:
                    lines.append("Tip: You can afford both Morning Tea and Breakfast (total Rs.80) if you increase your budget, or choose either.")
            elif has_evening:
                pass
            elif has_lunch_dinner:
                pass
            else:
                if budget < PRICE_TABLE["Evening Snacks"]:
                    lines.append("Note: This is only enough for Morning Tea.")
                elif budget < PRICE_TABLE["Breakfast"]:
                    lines.append("Tip: You can get Morning Tea AND Evening Snacks for Rs.65 (Rs.15 + Rs.50).")
            return "\n".join(lines)
        else:
            if has_morning:
                return f"With Rs.{budget}, you cannot afford any morning meals. Morning Tea costs Rs.15 and Breakfast costs Rs.65."
            elif has_evening:
                return f"With Rs.{budget}, you cannot afford evening snacks. Evening Snacks cost Rs.50."
            elif has_lunch_dinner:
                return f"With Rs.{budget}, you cannot afford lunch/dinner. Lunch & Dinner costs Rs.120."
            else:
                return f"With Rs.{budget}, you cannot afford any meal on the menu. The cheapest option is Morning Tea at Rs.15."

    return None


def run_query(query_text, embedder, reranker, client):
    special = handle_special_queries(query_text)
    if special:
        return special, [], 0

    from qdrant_client.models import Filter, FieldCondition, MatchValue
    conditions = []
    
    # Check for set number in the query (Set 1-7)
    set_match = re.search(r"\bset\s*([1-7])\b", query_text, re.IGNORECASE)
    if set_match:
        target_set = int(set_match.group(1))
        conditions.append(FieldCondition(key="set_number", match=MatchValue(value=target_set)))

    # Check for explicit meal types in the query
    ql = query_text.lower()
    if "morning tea" in ql:
        conditions.append(FieldCondition(key="meal_type", match=MatchValue(value="Morning Tea")))
    elif "evening snack" in ql or "evening snacks" in ql or "evening" in ql:
        conditions.append(FieldCondition(key="meal_type", match=MatchValue(value="Evening Snacks")))
    elif "breakfast" in ql:
        conditions.append(FieldCondition(key="meal_type", match=MatchValue(value="Breakfast")))
    elif "lunch" in ql or "dinner" in ql:
        conditions.append(FieldCondition(key="meal_type", match=MatchValue(value="Lunch & Dinner")))

    query_filter = Filter(must=conditions) if conditions else None

    t0 = time.time()
    qemb = embed_query(embedder, query_text)
    candidates = hybrid_search(client, qemb, query_filter=query_filter)
    chunks = rerank_chunks(reranker, query_text, candidates)
    ms = (time.time() - t0) * 1000

    if not chunks:
        chunks = [{
            "id": "bot_info",
            "text": BOT_INFO_TEXT,
            "payload": {
                "meal_type": "General Information",
                "set_number": None,
                "price": 0
            },
            "score": 0.0,
            "score_label": "Bot Info"
        }]

    prompt = build_prompt(query_text, chunks)
    try:
        answer = generate_answer(prompt)
    except Exception as e:
        answer = f"Error: {e}"

    return answer, chunks, ms


embedder, reranker = init_models()
client = init_qdrant()


def handle_query(query_text):
    answer, chunks, ms = run_query(query_text, embedder, reranker, client)
    rows = [[
        i,
        c["payload"].get("meal_type", "?"),
        str(c["payload"].get("set_number", "Overview")) if c["payload"].get("set_number") is not None else "Overview",
        f"Rs.{c['payload'].get('price', '?')}",
        f"{c.get('score', 0):.4f}",
        c["text"][:80] + "...",
    ] for i, c in enumerate(chunks, 1)]
    return answer, rows


with gr.Blocks(title="IRCTC Menu Assistant") as demo:
    gr.Markdown("# IRCTC South Central Zone - Menu Assistant")

    query_input = gr.Textbox(label="Your Question", placeholder="Ask about the menu...")
    query_btn = gr.Button("Ask", variant="primary")

    answer_output = gr.Markdown(label="Answer")

    chunks_table = gr.Dataframe(
        headers=["#", "Meal Type", "Set", "Price", "Score", "Preview"],
        label="Retrieved Chunks",
        column_widths=["5%", "12%", "8%", "8%", "10%", "57%"],
    )

    with gr.Row():
        for q in SAMPLE_QUESTIONS:
            btn = gr.Button(q, size="sm")
            btn.click(fn=lambda q=q: q, outputs=query_input).then(
                fn=handle_query, inputs=query_input, outputs=[answer_output, chunks_table]
            )

    query_btn.click(fn=handle_query, inputs=query_input, outputs=[answer_output, chunks_table])
    query_input.submit(fn=handle_query, inputs=query_input, outputs=[answer_output, chunks_table])

if __name__ == "__main__":
    demo.launch(share=True, theme=gr.themes.Soft())
