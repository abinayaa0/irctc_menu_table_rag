"""
IRCTC South Central Menu RAG — Simple Gradio UI (LLM Structured Chunking version).
"""

import os
import sys
import time
import re
import csv
import json
from datetime import datetime

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
from qdrant_client.models import NamedVector, NamedSparseVector, SparseVector, Filter, FieldCondition, MatchValue
from FlagEmbedding import BGEM3FlagModel, FlagReranker
from openai import OpenAI

load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
COLLECTION = "irctc_sc_menu_llm"
QDRANT_PATH = os.getenv("QDRANT_PATH", os.path.join(BASE_DIR, "qdrant_local"))
CHUNKS_PATH = os.path.join(BASE_DIR, "data", "chunks_llm.json")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
TOP_K_RETRIEVE = 20
TOP_K_RERANK = 15
RRF_K = 60

LOG_FILE_PATH = os.path.join(BASE_DIR, "data", "query_log.csv")

PRICE_TABLE = {"Morning Tea": 15, "Evening Snacks": 50, "Breakfast": 65, "Lunch & Dinner": 120}

SYSTEM_PROMPT = """You are an assistant for IRCTC Rajdhani/Duronto South Central Zone
Sleeper Class food menu queries.

RULES:
1. Answer ONLY using explicitly stated facts from the context. Do not infer, assume, or generalize.
2. CRITICAL — "All sets" accuracy rule: When asked whether ALL sets include an item, you MUST
   verify EVERY set in the context before answering. If even one set in the context does NOT list
   the item, the answer is NO. Respond with: "No, not all sets include [item]. Sets [X,Y,Z] have
   it but Sets [A,B] do not." Never say 'all sets' unless the context explicitly confirms it for
   every single set from 1 to 7.
3. Prices: Morning Tea Rs.15 | Breakfast Rs.65 | Evening Snacks Rs.50 |
   Lunch & Dinner Rs.120 — all inclusive of taxes. Never invent items or prices.
4. There are 7 rotating sets (Set 1-7) served cyclically across journeys.
5. For specific set questions, give that set's items exactly from context. If a set is not present
   in the context, do not assume or invent its items.
6. For general questions (e.g. "what's for breakfast"), summarise all 7 sets.
7. Mention Jain/Diabetic food on request only when the question asks about dietary restrictions,
   special meals, or food options generally. Do NOT append it to every answer.
8. Mention Masala Tea on demand only when the question is about tea or beverages.
   Do NOT append it to every answer.
9. For budget/price questions (e.g. "I have Rs.60"), list ALL meal types that fit within that
   budget using the prices above. Also mention what they can get for their specific budget and
   suggest combinations.
10. Be concise. No filler. Lead with the direct answer.
11. When answering about specific food items, always mention the meal category
    (e.g., Morning Tea, Breakfast, Evening Snacks, Lunch & Dinner) and price in which they are served.
12. If a specific food item is not found in the menu, say clearly that it is not served on this menu.
    Then look at the retrieved context and suggest the closest available alternative(s).
    Only say "I don't have that information" if the question is completely unrelated to the IRCTC
    food menu (e.g. train schedules, ticket booking, refunds)."""

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

# ---------------------------------------------------------------------------
# Food item index — built once at startup from chunks_llm.json
# Replaces fragile regex extraction with lookup against known menu items.
# ---------------------------------------------------------------------------

MEAL_TYPE_TERMS = {
    "breakfast", "lunch", "dinner", "tea", "snacks", "snack",
    "morning tea", "evening snacks", "morning", "evening",
}
STOPWORDS = {
    "and", "or", "the", "with", "in", "on", "at", "to", "a", "an", "of",
    "for", "is", "are", "was", "be", "by", "as", "from", "this", "that",
    "items", "item", "set", "sets", "menu", "meal", "meals", "serving",
    "served", "include", "includes", "have", "has", "does", "which", "what",
}


def _normalize(text: str) -> str:
    """Normalize Indian food spelling variants for fuzzy matching."""
    n = text.lower().strip()
    n = re.sub(r'bh', 'b', n)
    n = re.sub(r'dh', 'd', n)
    n = re.sub(r'gh', 'g', n)
    n = re.sub(r'sh', 's', n)
    n = re.sub(r'th', 't', n)
    n = re.sub(r'ph', 'f', n)
    n = re.sub(r'ly$', 'li', n)
    n = re.sub(r'ey$', 'i', n)
    n = re.sub(r'oo', 'u', n)
    n = re.sub(r'ee', 'i', n)
    return n


def build_food_index(chunks_path: str) -> dict:
    """
    Build a lookup: normalized_item_name -> {meal_type: [set_nums]}
    Indexes both full item names ("Medu Vada") and significant component words ("vada", "medu").
    """
    index = {}

    try:
        chunks = json.load(open(chunks_path, encoding="utf-8"))
    except Exception:
        return index

    for chunk in chunks:
        set_num = chunk.get("set_number")
        meal_type = chunk.get("meal_type", "")
        text = chunk.get("chunk_text", "")
        if not set_num or not meal_type:
            continue

        # Split chunk text into individual item strings
        # Delimiters: +, comma, newline, bullet, semicolon
        raw_parts = re.split(r'[+,\n•;\-]', text)
        for raw in raw_parts:
            # Strip labels like "Items:", "Breakfast:", etc.
            item = re.sub(r'^[^:]+:\s*', '', raw.strip())
            item = re.sub(r'\s+', ' ', item).strip(" .")
            if not item or len(item) < 3:
                continue

            # Skip if it's just a meal-type word
            if item.lower() in MEAL_TYPE_TERMS or item.lower() in STOPWORDS:
                continue

            # Index the full normalized name
            item_norm = _normalize(item)
            if len(item_norm) >= 3:
                _add_to_index(index, item_norm, item, meal_type, set_num)

            # Also index each significant word in multi-word items
            for word in item.split():
                word_norm = _normalize(word)
                if len(word_norm) >= 4 and word_norm not in STOPWORDS and word_norm not in MEAL_TYPE_TERMS:
                    _add_to_index(index, word_norm, word, meal_type, set_num)

    return index


def _add_to_index(index, key, display, meal_type, set_num):
    if key not in index:
        index[key] = {"display": display, "meal_sets": {}}
    if meal_type not in index[key]["meal_sets"]:
        index[key]["meal_sets"][meal_type] = []
    if set_num not in index[key]["meal_sets"][meal_type]:
        index[key]["meal_sets"][meal_type].append(set_num)


def lookup_item_in_query(query_text: str, food_index: dict) -> str | None:
    """
    Check if the query is asking about a specific food item's availability.
    Searches the query for known menu items (lookup, not extraction).
    Returns a deterministic answer string, or None if no match.
    """
    # Only handle queries that look like item-presence questions
    ql = query_text.lower()
    is_presence_query = bool(re.search(
        r'\b(which|what|where|does|do|is|are|have|has|include|contain|serve|served|available|find)\b',
        ql
    ))
    if not is_presence_query:
        return None

    # Skip if it's purely a budget question (numbers dominate)
    if re.search(r'\brs\.?\s*\d+|\d+\s*rs', ql):
        return None

    q_norm = _normalize(query_text)

    # Find all matching food items (sorted longest-first for specificity)
    matched = []
    for item_norm, data in food_index.items():
        if re.search(r'\b' + re.escape(item_norm) + r'\b', q_norm):
            matched.append((item_norm, data))

    if not matched:
        return None

    # Use the longest match (most specific term wins)
    matched.sort(key=lambda x: len(x[0]), reverse=True)
    item_norm, data = matched[0]
    item_display = data["display"].title()
    meal_sets = data["meal_sets"]

    lines = [f"{item_display} is served in:"]
    for meal_type, set_nums in sorted(meal_sets.items()):
        set_nums = sorted(set(set_nums))
        plural = "s" if len(set_nums) > 1 else ""
        sets_str = ", ".join(str(s) for s in set_nums)
        lines.append(f"  - {meal_type} (Rs.{PRICE_TABLE.get(meal_type, '?')}): Set{plural} {sets_str}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# LLM / model helpers
# ---------------------------------------------------------------------------

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
            vector=SparseVector(
                indices=query_emb["sparse_indices"],
                values=query_emb["sparse_values"],
            ),
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
        max_tokens=400,
    )
    return response.choices[0].message.content.strip()


# ---------------------------------------------------------------------------
# Special query handler (deterministic — never reaches LLM)
# ---------------------------------------------------------------------------

def handle_special_queries(query_text):
    q = query_text.lower()

    # Invalid set numbers
    set_nums = [int(n) for n in re.findall(r"\bset\s*(\d+)\b", query_text, re.IGNORECASE)]
    invalid_sets = [n for n in set_nums if n < 1 or n > 7]
    if invalid_sets:
        if len(invalid_sets) == 1:
            return f"There are only 7 rotating sets (Set 1 to Set 7) on the menu. Set {invalid_sets[0]} does not exist."
        sets_str = ", ".join(f"Set {n}" for n in invalid_sets)
        return f"There are only 7 rotating sets (Set 1 to Set 7) on the menu. {sets_str} do not exist."

    # Jain / diabetic
    if re.search(r"\b(jain|diabetic|special meal|dietary restriction)\b", q):
        return (
            "Jain and diabetic food can be provided on request. "
            "The menu document does not specify the exact items available for these options. "
            "Please request it from the train staff when your meal is served."
        )

    # Budget detection — try patterns in priority order
    BUDGET_PATTERN = (
        r"rs\.?\s*(\d+)"                                                     # rs.150 / rs 150
        r"|(\d+)\s*rs\.?"                                                    # 150rs / 150 rs
        r"|(?:with|have|got|for|get|budget|only|just|spend|using"
        r"|under|upto|up to|rupees?|inr|afford|pay|cost|costs)\s+(\d+)"     # keyword + number
    )
    m = re.search(BUDGET_PATTERN, q)
    budget = None
    if m:
        budget = int(next(g for g in m.groups() if g is not None))
    else:
        # Fallback: any standalone number in a budget-sounding question
        budget_words = ["get", "buy", "afford", "eat", "have", "pay", "cost", "order", "meal", "food"]
        if any(w in q for w in budget_words):
            set_nums_in_q = {str(n) for n in re.findall(r"\bset\s*(\d+)\b", q, re.IGNORECASE)}
            nums = [int(n) for n in re.findall(r"\b(\d+)\b", q)
                    if n not in set_nums_in_q and 1 <= int(n) <= 10000]
            if nums:
                budget = max(nums)

    if budget is not None:
        has_morning = (any(k in q for k in ["morning", "breakfast", "am"])
                       and not any(k in q for k in ["evening", "snacks", "pm", "dinner"]))
        has_evening = (any(k in q for k in ["evening", "snacks", "pm"])
                       and not any(k in q for k in ["morning", "breakfast", "am", "lunch"]))
        has_lunch_dinner = any(k in q for k in ["lunch", "dinner", "night", "noon", "midday"])

        candidates = list(PRICE_TABLE.items())
        if has_morning:
            candidates = [(n, p) for n, p in candidates if n in ["Morning Tea", "Breakfast"]]
        elif has_evening:
            candidates = [(n, p) for n, p in candidates if n in ["Evening Snacks"]]
        elif has_lunch_dinner:
            candidates = [(n, p) for n, p in candidates if n in ["Lunch & Dinner"]]

        affordable = [(n, p) for n, p in sorted(candidates, key=lambda x: x[1]) if p <= budget]
        if affordable:
            lines = [f"With Rs.{budget} you can afford:"]
            for name, price in affordable:
                lines.append(f"  - {name} (Rs.{price})")
            lines.append("")
            total_all = sum(p for _, p in affordable)
            if not has_morning and not has_evening and not has_lunch_dinner:
                if budget < PRICE_TABLE["Evening Snacks"]:
                    lines.append("Note: This is only enough for Morning Tea.")
                elif budget < PRICE_TABLE["Breakfast"]:
                    lines.append("Tip: You can get Morning Tea AND Evening Snacks for Rs.65 (Rs.15 + Rs.50).")
                elif budget >= total_all:
                    lines.append(f"Tip: You can buy all of the above in one journey for a total of Rs.{total_all}.")
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


# ---------------------------------------------------------------------------
# Retrieval routing
# ---------------------------------------------------------------------------

def scroll_all(client, scroll_filter=None):
    """Return all matching chunks from Qdrant (no top-K dropping)."""
    results, _ = client.scroll(
        collection_name=COLLECTION,
        scroll_filter=scroll_filter,
        limit=50,
        with_payload=True,
        with_vectors=False,
    )
    return [
        {"id": str(r.id), "text": r.payload.get("chunk_text", ""),
         "payload": r.payload, "score": 1.0, "score_label": "Full scan", "rrf": 1.0}
        for r in results
    ]


def run_query(query_text, embedder, reranker, client, food_index):
    # 1. Deterministic special queries (budget, jain, invalid set)
    special = handle_special_queries(query_text)
    if special:
        return special, [], 0

    # 2. Deterministic food item lookup (replaces regex extraction entirely)
    item_answer = lookup_item_in_query(query_text, food_index)
    if item_answer:
        # Still retrieve chunks for display purposes
        chunks = scroll_all(client)
        return item_answer, chunks[:5], 0

    # 3. Smart retrieval routing
    conditions = []
    set_match = re.search(r"\bset\s*([1-7])\b", query_text, re.IGNORECASE)
    if set_match:
        conditions.append(FieldCondition(key="set_number", match=MatchValue(value=int(set_match.group(1)))))

    ql = query_text.lower()
    meal_type_filter = None
    if "morning tea" in ql:
        meal_type_filter = "Morning Tea"
    elif "evening snack" in ql or "evening snacks" in ql or "evening" in ql:
        meal_type_filter = "Evening Snacks"
    elif "breakfast" in ql:
        meal_type_filter = "Breakfast"
    elif "lunch" in ql or "dinner" in ql:
        meal_type_filter = "Lunch & Dinner"

    if meal_type_filter:
        conditions.append(FieldCondition(key="meal_type", match=MatchValue(value=meal_type_filter)))

    has_specific_set = set_match is not None
    query_filter = Filter(must=conditions) if conditions else None

    t0 = time.time()

    if has_specific_set and meal_type_filter:
        # Case A: pinpoint query (Set N + meal type) — hybrid search
        qemb = embed_query(embedder, query_text)
        candidates = hybrid_search(client, qemb, query_filter=query_filter)
        chunks = rerank_chunks(reranker, query_text, candidates)
    elif meal_type_filter:
        # Case B: meal-type query — scroll ALL chunks for that meal (never drop sets)
        chunks = scroll_all(client, Filter(must=[
            FieldCondition(key="meal_type", match=MatchValue(value=meal_type_filter))
        ]))
    else:
        # Case C: general/cross-meal query — scroll everything
        chunks = scroll_all(client)

    ms = (time.time() - t0) * 1000

    if not chunks:
        chunks = [{
            "id": "bot_info", "text": BOT_INFO_TEXT,
            "payload": {"meal_type": "General Information", "set_number": None, "price": 0},
            "score": 0.0, "score_label": "Bot Info", "rrf": 0.0,
        }]

    prompt = build_prompt(query_text, chunks)
    try:
        answer = generate_answer(prompt)
    except Exception as e:
        answer = f"Error: {e}"

    return answer, chunks, ms


# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------

embedder, reranker = init_models()
client = init_qdrant()
food_index = build_food_index(CHUNKS_PATH)


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def log_query_to_csv(query_text, answer, chunks):
    file_exists = os.path.exists(LOG_FILE_PATH)
    chunks_desc_list = []
    for i, c in enumerate(chunks, 1):
        if c.get("id") == "bot_info":
            continue
        p = c.get("payload", {}) or {}
        set_str = f"Set {p.get('set_number')}" if p.get("set_number") is not None else "Overview"
        text_clean = c.get("text", "").replace("\n", " ")
        chunks_desc_list.append(
            f"[{i}] {p.get('meal_type','?')} | {set_str} | Rs.{p.get('price','?')} "
            f"| Score: {c.get('score', 0):.4f} | Content: {text_clean}"
        )
    chunks_str = "\n".join(chunks_desc_list)
    os.makedirs(os.path.dirname(LOG_FILE_PATH), exist_ok=True)
    with open(LOG_FILE_PATH, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["Timestamp", "User Query", "Bot Answer", "Retrieved Sources"])
        writer.writerow([datetime.now().strftime("%Y-%m-%d %H:%M:%S"), query_text, answer, chunks_str])
    return LOG_FILE_PATH


def handle_query(query_text):
    answer, chunks, ms = run_query(query_text, embedder, reranker, client, food_index)
    rows = [[
        i,
        c["payload"].get("meal_type", "?"),
        str(c["payload"].get("set_number", "Overview")) if c["payload"].get("set_number") is not None else "Overview",
        f"Rs.{c['payload'].get('price', '?')}",
        f"{c.get('score', 0):.4f}",
        c["text"][:80] + "...",
    ] for i, c in enumerate(chunks, 1)]
    log_file = log_query_to_csv(query_text, answer, chunks)
    return answer, rows, log_file


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

with gr.Blocks(title="IRCTC Menu Assistant (LLM)") as demo:
    gr.Markdown("# IRCTC South Central Zone - Menu Assistant (LLM Structured Chunks)")

    with gr.Row():
        with gr.Column(scale=3):
            query_input = gr.Textbox(label="Your Question", placeholder="Ask about the menu...")
            query_btn = gr.Button("Ask", variant="primary")
        with gr.Column(scale=1):
            initial_log_file = LOG_FILE_PATH if os.path.exists(LOG_FILE_PATH) else None
            log_file_output = gr.File(label="Download Query Log Sheet (CSV)", value=initial_log_file, interactive=False)

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
                fn=handle_query, inputs=query_input, outputs=[answer_output, chunks_table, log_file_output]
            )

    query_btn.click(fn=handle_query, inputs=query_input, outputs=[answer_output, chunks_table, log_file_output])
    query_input.submit(fn=handle_query, inputs=query_input, outputs=[answer_output, chunks_table, log_file_output])

if __name__ == "__main__":
    demo.launch(share=True, theme=gr.themes.Soft())
