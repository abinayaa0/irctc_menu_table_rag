"""
IRCTC South Central Menu Index Builder for LLM Tool-called Chunks.

This script reads the chunks from data/chunks_llm.json, embeds them using BGE-M3,
and stores them in a separate local Qdrant collection 'irctc_sc_menu_llm'.
"""

import os
import sys

# Ensure HuggingFace transformers bypasses TensorFlow and uses PyTorch directly
os.environ["USE_TF"] = "0"
os.environ["USE_TORCH"] = "1"

# Ensure UTF-8 output encoding for consoles (e.g. legacy Windows)
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import json
import uuid
from pathlib import Path
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, SparseVectorParams,
    SparseIndexParams, PointStruct, SparseVector,
)
from FlagEmbedding import BGEM3FlagModel
from rich.console import Console
from rich.panel import Panel
from rich.progress import track

load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Read from chunks_llm.json instead of chunks.json
CHUNKS_PATH = Path(os.path.join(BASE_DIR, "data", "chunks_llm.json"))
QDRANT_PATH = os.getenv("QDRANT_PATH", os.path.join(BASE_DIR, "qdrant_local"))
# Use a separate collection name
COLLECTION = "irctc_sc_menu_llm"
DENSE_DIM = 1024
BATCH_SIZE = 8


def get_client() -> QdrantClient:
    return QdrantClient(path=QDRANT_PATH)


def create_collection(client: QdrantClient) -> None:
    client.recreate_collection(
        collection_name=COLLECTION,
        vectors_config={
            "dense": VectorParams(size=DENSE_DIM, distance=Distance.COSINE)
        },
        sparse_vectors_config={
            "sparse": SparseVectorParams(
                index=SparseIndexParams(on_disk=False)
            )
        },
    )
    print(f" Collection '{COLLECTION}' created/recreated.")
    

def load_model() -> BGEM3FlagModel:
    console = Console()
    console.print("Loading BGE-M3 embedder...")
    model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)
    console.print(" BGE-M3 loaded successfully.")
    return model


def embed_texts(model: BGEM3FlagModel, texts: list[str]) -> list[dict]:
    embeddings = []
    for i in track(range(0, len(texts), BATCH_SIZE), description="Embedding LLM chunks..."):
        batch = texts[i:i + BATCH_SIZE]
        output = model.encode(
            batch,
            batch_size=len(batch),
            max_length=512,
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=False,
        )
        
        dense_vecs = output["dense_vecs"]
        lexical_weights = output["lexical_weights"]
        
        for idx in range(len(batch)):
            dense = dense_vecs[idx].tolist()
            d = lexical_weights[idx]
            
            sparse_indices = [int(k) for k in d.keys()]
            sparse_values = [float(v) for v in d.values()]
            
            embeddings.append({
                "dense": dense,
                "sparse_indices": sparse_indices,
                "sparse_values": sparse_values
            })
            
    return embeddings


def build_points(chunks: list[dict], embeddings: list[dict]) -> list[PointStruct]:
    points = []
    for chunk, emb in zip(chunks, embeddings):
        # Generate a deterministic UUID based on chunk_id so upserts are idempotent
        point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk["chunk_id"]))
        
        payload = {
            "chunk_id": chunk["chunk_id"],
            "meal_type": chunk["meal_type"],
            "set_number": chunk["set_number"],  # None for overview chunks
            "price": chunk["price"],
            "region": chunk["region"],
            "train_class": chunk["train_class"],
            "has_nonveg": chunk["has_nonveg_option"],
            "chunk_text": chunk["chunk_text"],
            "veg_items_str": " | ".join(chunk.get("veg_items", [])) if chunk.get("veg_items") else "",
            "common_items_str": " | ".join(chunk.get("common_items", [])) if chunk.get("common_items") else "",
        }
        
        point = PointStruct(
            id=point_id,
            vector={
                "dense": emb["dense"],
                "sparse": SparseVector(
                    indices=emb["sparse_indices"],
                    values=emb["sparse_values"],
                ),
            },
            payload=payload,
        )
        points.append(point)
        
    return points


def main() -> None:
    console = Console()
    
    if not CHUNKS_PATH.exists():
        console.print(f"[red]Error: {CHUNKS_PATH} not found. Run parse_pdf_llm.py first.[/red]")
        return
        
    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        chunks = json.load(f)
        
    client = get_client()
    create_collection(client)
    
    model = load_model()
    texts = [c["chunk_text"] for c in chunks]
    embeddings = embed_texts(model, texts)
    
    points = build_points(chunks, embeddings)
    
    console.print("Upserting points to Qdrant...")
    client.upsert(collection_name=COLLECTION, points=points, wait=True)
    
    info = client.get_collection(COLLECTION)
    
    panel_content = (
        f"Collection : {COLLECTION}\n"
        f"Points     : {info.points_count}\n"
        f"Storage    : {QDRANT_PATH}\n\n"
        "Next step  : python query_llm.py"
    )
    
    console.print(Panel(panel_content, title="LLM Index Built", expand=False, border_style="green"))


if __name__ == "__main__":
    main()
