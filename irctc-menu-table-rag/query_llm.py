"""
IRCTC South Central Menu RAG Query Assistant (LLM Structured Chunking version).

Terminal-based interactive loop with hybrid retrieval (dense + sparse BGE-M3),
RRF fusion, cross-encoder reranking, and local Gemma generation.
"""

import os
os.environ["USE_TF"] = "0"
os.environ["USE_TORCH"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["HF_HUB_OFFLINE"] = "1"

import sys
import time
import re
import argparse
from dataclasses import dataclass
from pathlib import Path
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import (
    NamedVector, NamedSparseVector, SparseVector,
)
from FlagEmbedding import BGEM3FlagModel, FlagReranker
from openai import OpenAI
from rich.console import Console
from rich.prompt import Prompt
from rich.panel import Panel
from rich.table import Table
from rich.markdown import Markdown

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
COLLECTION = "irctc_sc_menu_llm"
QDRANT_PATH = os.getenv("QDRANT_PATH", os.path.join(BASE_DIR, "qdrant_local"))
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
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
12. If a specific food item is not found in the menu, say clearly that it is not served on this menu.
    Then look at the retrieved context and suggest the closest available alternative(s) — for example
    if someone asks about poha and the context shows upma, say "Poha is not on the menu. The closest
    item available is Upma, served in [Set X] Breakfast (Rs.65)."
    Only say "I don't have that information" if the question is completely unrelated to the IRCTC food
    menu (e.g. train schedules, ticket booking, refunds)."""

BOT_INFO_TEXT = """This chatbot is the IRCTC South Central Zone Menu Assistant.
It helps passengers in Sleeper Class (SL) on Rajdhani and Duronto trains look up the 7-day rotating food menu, prices, and options.
It can answer questions about:
- What is served for Morning Tea (Rs.15), Breakfast (Rs.65), Evening Snacks (Rs.50), and Lunch/Dinner (Rs.120).
- Which dishes are served on specific menu sets (Set 1 to Set 7).
- Pricing and budget questions (e.g., what meals can be bought for a given price).
- Special meal options like Jain/diabetic food (available on request) and Ready Made Masala Tea (available on demand)."""


@dataclass
class RetrievedChunk:
    id: str
    chunk_text: str
    payload: dict
    dense_rank: int = 0
    sparse_rank: int = 0
    rrf_score: float = 0.0
    rerank_score: float = 0.0


def init_llm():
    return OpenAI(base_url=OLLAMA_BASE_URL, api_key="ollama")


def load_models() -> tuple[BGEM3FlagModel, FlagReranker | None]:
    console = Console()
    console.print("[bold blue]Loading BGE-M3 Embedder...[/bold blue]")
    embedder = BGEM3FlagModel("BAAI/bge-m3", use_fp16=False, device="cpu")

    console.print("[bold blue]Loading Reranker...[/bold blue]")
    try:
        reranker = FlagReranker("BAAI/bge-reranker-base", use_fp16=False, device="cpu")
    except Exception as e:
        console.print(f"[yellow]Reranker unavailable ({e}). Falling back to hybrid RRF ranking.[/yellow]")
        reranker = None

    return embedder, reranker


def embed_query(model: BGEM3FlagModel, query: str) -> dict:
    output = model.encode(
        [query],
        return_dense=True,
        return_sparse=True,
        return_colbert_vecs=False,
        max_length=256,
    )
    dense = output["dense_vecs"][0].tolist()
    d = output["lexical_weights"][0]
    sparse_indices = [int(k) for k in d.keys()]
    sparse_values = [float(v) for v in d.values()]
    return {"dense": dense, "sparse_indices": sparse_indices, "sparse_values": sparse_values}


def hybrid_search(
    client: QdrantClient,
    query_emb: dict,
    query_filter=None,
    top_k: int = TOP_K_RETRIEVE,
) -> list[RetrievedChunk]:
    dense_hits = client.search(
        collection_name=COLLECTION,
        query_vector=NamedVector(name="dense", vector=query_emb["dense"]),
        query_filter=query_filter,
        limit=top_k,
        with_payload=True,
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
        limit=top_k,
        with_payload=True,
    )

    chunk_map: dict[str, RetrievedChunk] = {}

    for rank, hit in enumerate(dense_hits, 1):
        point_id = str(hit.id)
        if point_id not in chunk_map:
            chunk_map[point_id] = RetrievedChunk(
                id=point_id,
                chunk_text=hit.payload.get("chunk_text", ""),
                payload=hit.payload,
            )
        chunk_map[point_id].dense_rank = rank
        chunk_map[point_id].rrf_score += 1 / (RRF_K + rank)

    for rank, hit in enumerate(sparse_hits, 1):
        point_id = str(hit.id)
        if point_id not in chunk_map:
            chunk_map[point_id] = RetrievedChunk(
                id=point_id,
                chunk_text=hit.payload.get("chunk_text", ""),
                payload=hit.payload,
            )
        chunk_map[point_id].sparse_rank = rank
        chunk_map[point_id].rrf_score += 1 / (RRF_K + rank)

    sorted_chunks = list(chunk_map.values())
    sorted_chunks.sort(key=lambda c: c.rrf_score, reverse=True)
    return sorted_chunks[:top_k]


def rerank(
    reranker: FlagReranker | None,
    query: str,
    chunks: list[RetrievedChunk],
    top_n: int = TOP_K_RERANK,
) -> list[RetrievedChunk]:
    if not chunks:
        return []

    if reranker is None:
        for chunk in chunks:
            chunk.rerank_score = chunk.rrf_score
        return chunks[:top_n]

    pairs = [[query, c.chunk_text] for c in chunks]
    scores = reranker.compute_score(pairs, normalize=True)

    for i, chunk in enumerate(chunks):
        chunk.rerank_score = float(scores[i])

    chunks.sort(key=lambda c: c.rerank_score, reverse=True)
    
    # Filter out low-scoring chunks to prevent conflation/hallucination in small LLMs
    if chunks:
        top_score = chunks[0].rerank_score
        if top_score >= 0.3:
            threshold = top_score * 0.5
            chunks = [c for c in chunks if c.rerank_score >= threshold]
        else:
            chunks = [c for c in chunks if c.rerank_score >= 0.01]
        
    return chunks[:top_n]


def build_prompt(query: str, chunks: list[RetrievedChunk]) -> str:
    context_blocks = []
    for i, chunk in enumerate(chunks, 1):
        p = chunk.payload
        block = (
            f"[{i}] {p.get('meal_type','?')} | "
            f"Set {p.get('set_number','Overview')} | "
            f"Rs.{p.get('price','?')}\n"
            f"{chunk.chunk_text}"
        )
        context_blocks.append(block)

    return (
        SYSTEM_PROMPT + "\n\n"
        "--- RETRIEVED CONTEXT ---\n"
        + "\n\n".join(context_blocks)
        + "\n--- END CONTEXT ---\n\n"
        f"Question: {query}\n\nAnswer:"
    )


def generate_answer(prompt: str) -> str:
    client = init_llm()
    response = client.chat.completions.create(
        model=OLLAMA_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
        max_tokens=300,
    )
    return response.choices[0].message.content.strip()


def handle_special_queries(query: str) -> str | None:
    q = query.lower()

    # Check for invalid set numbers (outside 1-7)
    set_nums = [int(n) for n in re.findall(r"\bset\s*(\d+)\b", query, re.IGNORECASE)]
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

    m = (re.search(r"rs\.?\s*(\d+)", q)
         or re.search(r"(\d+)\s*rs\.?", q)
         or re.search(r"(?:with|have|got|for|get|budget|only|just|spend|using|under|upto|up to|rupees?|inr)\s+(\d+)", q))
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


def print_sources(chunks: list[RetrievedChunk]) -> None:
    console = Console()
    table = Table(title="Sources used")
    table.add_column("#", style="cyan", justify="right")
    table.add_column("Meal Type", style="magenta")
    table.add_column("Set", style="green")
    table.add_column("Price", style="yellow")
    table.add_column("Score", style="blue")
    table.add_column("Preview", style="white")

    for i, chunk in enumerate(chunks, 1):
        p = chunk.payload
        set_str = str(p.get("set_number")) if p.get("set_number") is not None else "Overview"
        preview = chunk.chunk_text.replace("\n", " ")[:60] + "..."
        table.add_row(
            str(i),
            p.get("meal_type", "?"),
            set_str,
            f"Rs.{p.get('price', '?')}",
            f"{chunk.rerank_score:.3f}",
            preview,
        )

    console.print(table)


def run_query(embedder, reranker, client, query: str) -> tuple[str | None, list[RetrievedChunk]]:
    special = handle_special_queries(query)
    if special:
        return special, []

    from qdrant_client.models import Filter, FieldCondition, MatchValue
    conditions = []
    
    # Check for set number in the query (Set 1-7)
    set_match = re.search(r"\bset\s*([1-7])\b", query, re.IGNORECASE)
    if set_match:
        target_set = int(set_match.group(1))
        conditions.append(FieldCondition(key="set_number", match=MatchValue(value=target_set)))

    # Check for explicit meal types in the query
    ql = query.lower()
    if "morning tea" in ql:
        conditions.append(FieldCondition(key="meal_type", match=MatchValue(value="Morning Tea")))
    elif "evening snack" in ql or "evening snacks" in ql or "evening" in ql:
        conditions.append(FieldCondition(key="meal_type", match=MatchValue(value="Evening Snacks")))
    elif "breakfast" in ql:
        conditions.append(FieldCondition(key="meal_type", match=MatchValue(value="Breakfast")))
    elif "lunch" in ql or "dinner" in ql:
        conditions.append(FieldCondition(key="meal_type", match=MatchValue(value="Lunch & Dinner")))

    query_filter = Filter(must=conditions) if conditions else None

    has_specific_set = any(
        hasattr(c, 'key') and c.key == 'set_number' for c in conditions
    )
    meal_type_filter = None
    for cond in conditions:
        if hasattr(cond, 'key') and cond.key == 'meal_type':
            meal_type_filter = cond.match.value
            break

    console = Console()
    t0 = time.time()

    # --- Smart retrieval routing (dataset: 26 chunks total, 8 per meal type) ---
    # A) Specific set + meal  → hybrid+filter (pinpoint, 1 chunk)
    # B) Meal type only       → scroll ALL chunks for that meal (8 chunks, no drops)
    # C) No meal type         → scroll ALL 26 chunks (item/cross-meal search)

    def scroll_to_chunks(scroll_filter=None):
        results, _ = client.scroll(
            collection_name=COLLECTION,
            scroll_filter=scroll_filter,
            limit=30,
            with_payload=True,
            with_vectors=False,
        )
        return [
            RetrievedChunk(
                id=str(r.id),
                chunk_text=r.payload.get("chunk_text", ""),
                payload=r.payload,
                rerank_score=1.0,
            )
            for r in results
        ]

    if has_specific_set and meal_type_filter:
        # Case A: pinpoint query
        query_emb = embed_query(embedder, query)
        candidates = hybrid_search(client, query_emb, query_filter=query_filter)
        final_chunks = rerank(reranker, query, candidates)
    elif meal_type_filter:
        # Case B: general meal-type query — scroll all sets for this meal
        scroll_filter = Filter(must=[
            FieldCondition(key="meal_type", match=MatchValue(value=meal_type_filter))
        ])
        final_chunks = scroll_to_chunks(scroll_filter)
    else:
        # Case C: item or cross-meal search — scroll everything
        final_chunks = scroll_to_chunks()

    retrieval_ms = (time.time() - t0) * 1000

    if not final_chunks:
        final_chunks = [RetrievedChunk(
            id="bot_info",
            chunk_text=BOT_INFO_TEXT,
            payload={
                "meal_type": "General Information",
                "set_number": None,
                "price": 0
            }
        )]

    console.print("Generating answer...")
    prompt = build_prompt(query, final_chunks)
    try:
        answer = generate_answer(prompt)
    except Exception as e:
        answer = f"Error: {e}"
        return answer, final_chunks

    console.print(f"[dim]Retrieval: {retrieval_ms:.0f}ms | {len(final_chunks)} chunks[/dim]")
    return answer, final_chunks


def interactive_mode(embedder, reranker, client) -> None:
    console = Console()

    body = (
        "Region: South Central Zone | Class: Sleeper (SL)\n"
        "4 meal types - 7 rotating sets - Hybrid search + reranking\n\n"
        "[bold]Example questions:[/bold]\n"
        "  What is served for breakfast on Set 3?\n"
        "  Is there a non-veg option for lunch?\n"
        "  I have Rs.60 what can I get?\n"
        "  What dal is served in Set 5 lunch?\n"
        "  What is common across all breakfast sets?\n"
        "  Tell me about Jain food options\n\n"
        "Type [bold]exit[/bold] or [bold]quit[/bold] to end."
    )
    console.print(Panel(body, title="IRCTC South Central Menu - RAG Assistant (LLM)", border_style="blue", expand=False))

    while True:
        try:
            query = Prompt.ask("\n[bold green]You[/bold green]").strip()
        except (KeyboardInterrupt, EOFError):
            console.print("\nGoodbye!")
            break

        if not query:
            continue
        if query.lower() in ("exit", "quit", "q"):
            console.print("Goodbye!")
            break

        answer, chunks = run_query(embedder, reranker, client, query)
        if answer:
            console.print(Panel(answer, title="Answer", border_style="green"))
            print_sources(chunks)


def main() -> None:
    parser = argparse.ArgumentParser(description="IRCTC South Central Menu RAG Query (LLM)")
    parser.add_argument("--query", "-q", type=str, help="Single query and exit")
    args = parser.parse_args()

    console = Console()

    if not Path(QDRANT_PATH).exists():
        console.print(
            f"[red]Error: Qdrant index not found at '{QDRANT_PATH}'.[/red]\n"
            "Run [bold]python build_index_llm.py[/bold] first."
        )
        sys.exit(1)

    client = QdrantClient(path=QDRANT_PATH)
    embedder, reranker = load_models()

    if args.query:
        answer, chunks = run_query(embedder, reranker, client, args.query)
        if answer:
            print(answer)
            if chunks:
                print_sources(chunks)
    else:
        interactive_mode(embedder, reranker, client)


if __name__ == "__main__":
    main()
