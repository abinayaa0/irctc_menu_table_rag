"""
IRCTC South Central Zone Rajdhani/Duronto Menu Parser (Sleeper Class).

This module dynamically parses South-Central.pdf using pdfplumber to extract
the 7-day rotating cyclic menu into structured chunks suitable for RAG.
It creates 32 chunks in total: 28 set-specific chunks and 4 meal overviews.
"""

import json
import os
import re
import sys
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional
import pdfplumber
from rich.console import Console
from rich.table import Table
from rich.progress import track
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma2:2b")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")

# Ensure UTF-8 output encoding for consoles (e.g. legacy Windows)
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

PDF_PATH = Path("data/South-Central.pdf")
OUTPUT_PATH = Path("data/chunks.json")


@dataclass
class MealSet:
    meal_type: str              # "Morning Tea" / "Breakfast" / "Evening Snacks" / "Lunch & Dinner"
    set_number: int             # 1–7
    price: int                  # 15 / 65 / 50 / 120
    region: str                 # "South Central Zone"
    train_class: str            # "Sleeper Class (SL)"
    veg_items: List[str]        # list of veg item strings for this set
    nonveg_items: List[str]     # non-veg option items (empty list if none)
    common_items: List[str]     # items identical across all sets (tea, disposables)
    has_nonveg_option: bool     # True if an OR/nonveg row exists for this meal
    footnotes: List[str]        # global footnotes from rows 35-38
    chunk_id: str               # e.g. "morning_tea_set_1", "lunch_dinner_set_4"


def extract_table(pdf_path: Path) -> List[List[Optional[str]]]:
    """
    Open PDF with pdfplumber and extract the first table.
    """
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF file not found at: {pdf_path.absolute()}")
    
    with pdfplumber.open(pdf_path) as pdf:
        if not pdf.pages:
            raise ValueError("The PDF contains no pages.")
        
        tables = pdf.pages[0].extract_tables()
        if not tables:
            raise ValueError("No tables were found in the PDF page.")
        
        return tables[0]


def parse_price(cell: str) -> int:
    """
    Input: "15/-" or "65/-" or "120/-", returns the integer representation.
    """
    clean_val = cell.replace("/-", "").strip()
    # Strip any extra text just in case (e.g. currency symbol or spacing)
    clean_val = re.sub(r"[^\d]", "", clean_val)
    if clean_val.isdigit():
        return int(clean_val)
    return 0


def clean_cell(cell: Optional[str]) -> str:
    """
    Replace \n and other whitespaces with single spaces, strip, and return.
    """
    if cell is None:
        return ""
    return re.sub(r"\s+", " ", str(cell)).strip()


def is_header_row(row: List[Optional[str]]) -> bool:
    """
    Return True if clean_cell(row[1]) == "Type of Services"
    """
    if len(row) > 1:
        return clean_cell(row[1]) == "Type of Services"
    return False


def is_footnote_row(row: List[Optional[str]]) -> bool:
    """
    Return True if clean_cell(row[1]) starts with any footnote keywords.
    """
    if len(row) > 1:
        text = clean_cell(row[1])
        keywords = ("Service In", "Food to be", "Option of", "Ready Made")
        return any(text.startswith(kw) for kw in keywords)
    return False


def extract_footnotes(table: List[List[Optional[str]]]) -> List[str]:
    """
    Find all rows where is_footnote_row() is True and extract footnote text.
    """
    footnotes = []
    for row in table:
        if is_footnote_row(row):
            footnotes.append(clean_cell(row[1]))
    return footnotes


def _parse_meal_section(rows: List[List[Optional[str]]], meal_type: str, price: int, footnotes: List[str]) -> List[MealSet]:
    """
    Parse an accumulated list of rows representing a single meal section.
    """
    region = "South Central Zone"
    train_class = "Sleeper Class (SL)"

    # 1. Detect if any "Or" / "OR" row exists in the section and find its index
    or_row_idx = -1
    has_nonveg_option = False
    for idx, r in enumerate(rows):
        vals = [clean_cell(r[c]) for c in range(2, 9)]
        if any(v in ("Or", "OR") for v in vals):
            has_nonveg_option = True
            or_row_idx = idx
            break

    # 2. Detect common items in rows.
    # Exclude the "Or"/"OR" row and the immediate non-veg option row (or_row_idx + 1)
    common_rows = []
    for idx, r in enumerate(rows):
        if idx == or_row_idx or idx == or_row_idx + 1:
            continue
        vals = [clean_cell(r[c]) for c in range(2, 9)]
        # If all sets (columns 2-8) share the identical non-empty text, it is common
        if len(set(vals)) == 1 and vals[0] != "":
            common_rows.append(idx)

    common_items = [clean_cell(rows[idx][2]) for idx in common_rows]

    meal_sets = []
    # Columns 2 to 8 map to sets 1 to 7
    for s in range(7):
        col_index = s + 2
        veg_items = []
        nonveg_items = []

        for idx, r in enumerate(rows):
            if idx == or_row_idx or idx in common_rows:
                continue

            cell_val = clean_cell(r[col_index])
            if cell_val == "":
                continue

            if or_row_idx != -1 and idx > or_row_idx:
                # Rows after "Or" represent the non-veg option (e.g. Omelette, Chicken Curry)
                nonveg_items.append(cell_val)
            else:
                # Rows before "Or" (or all rows if no "Or") represent veg items
                veg_items.append(cell_val)

        # Build chunk_id
        meal_slug = meal_type.lower().replace(" ", "_").replace("&", "and")
        chunk_id = f"{meal_slug}_set_{s+1}"

        meal_sets.append(MealSet(
            meal_type=meal_type,
            set_number=s+1,
            price=price,
            region=region,
            train_class=train_class,
            veg_items=veg_items,
            nonveg_items=nonveg_items,
            common_items=common_items,
            has_nonveg_option=has_nonveg_option,
            footnotes=footnotes,
            chunk_id=chunk_id
        ))

    return meal_sets


def parse_table(table: List[List[Optional[str]]]) -> List[MealSet]:
    """
    Iterate over rows and group them into 4 meal sections.
    """
    footnotes = extract_footnotes(table)
    all_meal_sets: List[MealSet] = []
    current_meal_rows: List[List[Optional[str]]] = []

    # Row 0: Title row, Row 1: Header row -> start from Row 2
    for idx in range(2, len(table)):
        row = table[idx]

        if is_header_row(row):
            # Process accumulated meal section
            if current_meal_rows:
                # Find label row
                label_row = None
                for r in current_meal_rows:
                    if clean_cell(r[1]) != "" and parse_price(clean_cell(r[9])) > 0:
                        label_row = r
                        break
                if label_row:
                    meal_type = clean_cell(label_row[1])
                    price = parse_price(clean_cell(label_row[9]))
                    meal_sets = _parse_meal_section(current_meal_rows, meal_type, price, footnotes)
                    all_meal_sets.extend(meal_sets)
                current_meal_rows = []
            continue

        if is_footnote_row(row):
            # Process last accumulated meal section and stop
            if current_meal_rows:
                label_row = None
                for r in current_meal_rows:
                    if clean_cell(r[1]) != "" and parse_price(clean_cell(r[9])) > 0:
                        label_row = r
                        break
                if label_row:
                    meal_type = clean_cell(label_row[1])
                    price = parse_price(clean_cell(label_row[9]))
                    meal_sets = _parse_meal_section(current_meal_rows, meal_type, price, footnotes)
                    all_meal_sets.extend(meal_sets)
                current_meal_rows = []
            break

        current_meal_rows.append(row)

    # Defensive cleanup if table ends without hitting a footnote row
    if current_meal_rows:
        label_row = None
        for r in current_meal_rows:
            if clean_cell(r[1]) != "" and parse_price(clean_cell(r[9])) > 0:
                label_row = r
                break
        if label_row:
            meal_type = clean_cell(label_row[1])
            price = parse_price(clean_cell(label_row[9]))
            meal_sets = _parse_meal_section(current_meal_rows, meal_type, price, footnotes)
            all_meal_sets.extend(meal_sets)

    return all_meal_sets


def meal_set_to_chunk_text(ms: MealSet) -> str:
    lines = [f"{ms.meal_type} (Rs.{ms.price}) — Set {ms.set_number}"]
    lines.append("Includes:")
    for item in ms.common_items:
        lines.append(f"- {item}")
    for item in ms.veg_items:
        lines.append(f"- {item}")
    if ms.has_nonveg_option and ms.nonveg_items:
        lines.append("")
        lines.append("Non-veg substitute (choose instead of veg main):")
        for item in ms.nonveg_items:
            lines.append(f"- {item}")
    return "\n".join(lines)


def normalize_chunk_with_llm(ms: MealSet) -> str:
    """
    Call Ollama local LLM to normalize the raw menu text into a search-optimized semantic record.
    If the LLM call fails, it falls back to the local text representation.
    """
    fallback_text = meal_set_to_chunk_text(ms)
    try:
        client = OpenAI(base_url=OLLAMA_BASE_URL, api_key="ollama")
        
        # Structure the input for the LLM
        common_items_str = "\n".join(f"- {item}" for item in ms.common_items) if ms.common_items else "None"
        veg_items_str = "\n".join(f"- {item}" for item in ms.veg_items) if ms.veg_items else "None"
        nonveg_items_str = "\n".join(f"- {item}" for item in ms.nonveg_items) if ms.nonveg_items else "None"
        
        system_prompt = (
            "You are a precise data normalizer for an IRCTC railway catering menu.\n"
            "Your goal is to convert raw menu items into a highly readable, structured, "
            "and search-optimized semantic record.\n"
            "Follow these rules strictly:\n"
            "1. Do not summarize or omit any dish or details. Keep all quantities exactly (e.g., '3 nos', '2 sliced bread').\n"
            "2. Do not invent items or prices. Do not add promotional or marketing adjectives.\n"
            "3. Correctly classify the cuisine style (e.g. South Indian, North Indian, standard breakfast, tea/coffee, kachori/samosa snack) based strictly on the dishes present. Do not default all meals to South Indian.\n"
            "4. Only list keywords and spelling variations (like 'dosa' for 'dosai') if they are actually present or directly related to the items served in this specific menu. Do NOT include words like 'dosa' or 'dosai' if the menu does not contain dosa or dosai.\n"
            "5. Format the output EXACTLY matching the structure below."
        )

        user_prompt = f"""Convert the following raw menu details into a standalone semantic record:

Meal Type: {ms.meal_type}
Set Number: Set {ms.set_number}
Price: Rs.{ms.price}
Region: {ms.region}
Train Class: {ms.train_class}

Common Items:
{common_items_str}

Vegetarian Items:
{veg_items_str}

Non-Vegetarian Items:
{nonveg_items_str}

Desired Output Structure:
{ms.meal_type} menu for Set {ms.set_number}.

Vegetarian option:
* [Item 1]
* [Item 2]

Non-vegetarian option:
* [Item 1]
* [Item 2]

Price: Rs.{ms.price}.

Search & Metadata:
* Meal Type: {ms.meal_type}
* Set Number: Set {ms.set_number}
* Price: Rs.{ms.price}
* Cuisine/Cuisine Style: [Classify the cuisine style of the dishes in this specific menu. For example, use 'South Indian' only if the menu contains South Indian dishes like idli, vada, dosa, upma, pongal, bonda. Use 'North Indian' for dal, roti, pulao. Use 'Standard Breakfast' or 'Continental' for bread/omelette. Use 'Beverages' for Morning Tea.]
* Keywords: [comma-separated list of dishes and items actually served in this specific menu. You can include alternative names or spellings only for items present in this meal (e.g., if 'Medu Vada' is present, include 'vada'. If 'Dosai' is present, include 'dosa'. If neither is present, do NOT include 'dosa' or 'vada'.)]
"""
        
        response = client.chat.completions.create(
            model=OLLAMA_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.1,
            max_tokens=400
        )
        normalized_text = response.choices[0].message.content.strip()
        if normalized_text:
            return normalized_text
    except Exception as e:
        print(f"\n⚠️ Warning: LLM normalization failed for Set {ms.set_number} {ms.meal_type}. Falling back to default format. Error: {e}")
    
    return fallback_text


def build_chunks(meal_sets: List[MealSet]) -> List[Dict[str, Any]]:
    """
    Aggregates MealSet objects into set chunks and overview chunks.
    """
    chunks: List[Dict[str, Any]] = []

    # 1. Add specific set chunks (28 total)
    for ms in track(meal_sets, description="Normalizing specific meal set chunks via LLM..."):
        chunk = asdict(ms)
        chunk["chunk_text"] = normalize_chunk_with_llm(ms)
        chunks.append(chunk)


    # 3. Add a dedicated Service Information chunk for global footnotes
    if meal_sets:
        sample = meal_sets[0]
        footnote_lines = sample.footnotes if sample.footnotes else [
            "Option of Jain/diabetic Food to be provided.",
            "Ready Made Masala Tea to be provided on demand.",
            "Service in Good Quality Casseroles.",
            "Food to be served on a tray with menu details and IRCTC toll free number."
        ]
        lines = [
            "General Service Information",
            "",
            "The following policies apply to all meal types and sets:",
        ]
        for note in footnote_lines:
            lines.append(f"- {note}")

        chunks.append({
            "meal_type": "General Information",
            "set_number": None,
            "price": 0,
            "region": sample.region,
            "train_class": sample.train_class,
            "veg_items": [],
            "nonveg_items": [],
            "common_items": [],
            "has_nonveg_option": False,
            "footnotes": footnote_lines,
            "chunk_id": "service_notes",
            "chunk_text": "\n".join(lines)
        })

    return chunks


def main() -> None:
    """
    Main execution loop.
    """
    try:
        table = extract_table(PDF_PATH)
        meal_sets = parse_table(table)
        chunks = build_chunks(meal_sets)
        
        # Ensure target directory exists and save
        OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
            json.dump(chunks, f, indent=2, ensure_ascii=False)
            
        console = Console()
        rich_table = Table(title="Generated Menu Chunks Summary (pdfplumber)")
        rich_table.add_column("chunk_id", style="cyan", no_wrap=True)
        rich_table.add_column("meal_type", style="magenta")
        rich_table.add_column("set", style="green")
        rich_table.add_column("price", style="yellow")
        rich_table.add_column("text_preview", style="white")

        for chunk in chunks:
            set_str = str(chunk["set_number"]) if chunk["set_number"] is not None else "Overview"
            preview = chunk["chunk_text"].replace("\n", " ")[:80] + "..."
            rich_table.add_row(
                chunk["chunk_id"],
                chunk["meal_type"],
                set_str,
                f"₹{chunk['price']}",
                preview
            )
            
        console.print(rich_table)
        print(f"✅ {len(chunks)} chunks saved to {OUTPUT_PATH}")
        print(f"   → {len(meal_sets)} meal×set combinations")
        print(f"   → 4 overview chunks")
        print(f"   → {len(extract_footnotes(table))} footnotes captured")
        
    except Exception as e:
        print(f"❌ Error occurred during parsing: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
