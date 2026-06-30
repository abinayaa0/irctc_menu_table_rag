"""
IRCTC Menu Parser using LLM Tool Calling (Structured Ingestion).

This module extracts the raw table text from data/South-Central.pdf
and feeds it to the LLM forcing it to use a 'serialize_all_menu_tables' tool call.
Supports specifying local models (via Ollama) or Gemini API models (using GEMINI_API_KEY).
Outputs structured chunks matching the original chunks format to data/chunks_llm.json,
and the full unified menu description text to data/menu_descriptions.md.
"""

import os
import sys
import re
import json
import argparse
import pdfplumber
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv

# Ensure UTF-8 output encoding for consoles (e.g. legacy Windows)
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

load_dotenv()

PDF_PATH = Path("data/South-Central.pdf")
OUTPUT_PATH = Path("data/chunks_llm.json")
DEFAULT_OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma2:2b")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")

# Define the Tool Calling Schema wrapped in a top-level array with global footnotes added
serialize_all_menu_tables_tool = {
    "type": "function",
    "function": {
        "name": "serialize_all_menu_tables",
        "description": "Converts and saves all menu services and global catering notes into structured data fields.",
        "parameters": {
            "type": "object",
            "properties": {
                "services": {
                    "type": "array",
                    "description": "List of the four catering meal services extracted from the menu table",
                    "items": {
                        "type": "object",
                        "properties": {
                            "service_name": {
                                "type": "string",
                                "description": "Category label (e.g. Morning Tea, Breakfast, Evening Snacks, Lunch & Dinner)"
                            },
                            "price": {
                                "type": "integer",
                                "description": "Price in Rupees (Rs.) inclusive of taxes"
                            },
                            "common_items": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "List of items served identical across all sets (e.g. Tea bag, sugar, creamer, stirrer, cup)"
                            },
                            "has_nonveg_option": {
                                "type": "boolean",
                                "description": "True if a non-vegetarian substitute is available"
                            },
                            "nonveg_items": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "The specific items in the non-vegetarian substitute (e.g. Omelette, bread, butter)"
                            },
                            "sets": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "set_no": {"type": "integer"},
                                        "veg_items": {
                                            "type": "array",
                                            "items": {"type": "string"},
                                            "description": "Vegetarian items specific to this set number. Leave empty list [] if there are no set-specific items (like for Morning Tea)"
                                        }
                                    },
                                    "required": ["set_no", "veg_items"]
                                }
                            }
                        },
                        "required": ["service_name", "price", "common_items", "has_nonveg_option", "nonveg_items", "sets"]
                    }
                },
                "global_footnotes": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of global policy notes and service rules at the bottom of the table (e.g., options of Jain/diabetic food, ready made tea, toll-free numbers, tray service details)."
                }
            },
            "required": ["services", "global_footnotes"]
        }
    }
}

def extract_raw_table_text():
    with pdfplumber.open(PDF_PATH) as pdf:
        table = pdf.pages[0].extract_tables()[0]
    
    lines = []
    for row in table:
        cleaned_row = [str(cell).replace("\n", " ").strip() if cell is not None else "" for cell in row]
        if any(cleaned_row):
            lines.append(" | ".join(cleaned_row))
    return "\n".join(lines)

def run_tool_calling_ingestion(table_text, model_name):
    # Route model target to Gemini API if the model name contains "gemini"
    is_gemini = "gemini" in model_name.lower()
    
    if is_gemini:
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            api_key = os.getenv("GEMINI_API_KEY")
            
        if not api_key:
            print("[ERROR] GEMINI_API_KEY environment variable is not set. Please export GEMINI_API_KEY before running.")
            return None
        
        base_url = "https://generativelanguage.googleapis.com/v1beta/openai/"
        print(f"Routing extraction to Gemini API endpoint with model: '{model_name}'...")
    else:
        api_key = "ollama"
        base_url = OLLAMA_BASE_URL
        print(f"\nCalling Local LLM (Model: '{model_name}') via Ollama...")

    client = OpenAI(base_url=base_url, api_key=api_key)
    
    system_prompt = (
        "You are an ingestion agent extracting data from IRCTC rail menu tables.\n"
        "Analyze the provided raw table text and extract the details for ALL catering services "
        "and global footnotes by executing the `serialize_all_menu_tables` tool.\n"
        "Do not invent any items or sets. Make sure to extract all four services and all bottom footnotes."
    )
    
    user_prompt = f"Analyze and serialize the entire menu and footnotes from this table:\n\n{table_text}"
    
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            tools=[serialize_all_menu_tables_tool],
            tool_choice={"type": "function", "function": {"name": "serialize_all_menu_tables"}},
            temperature=0.1
        )
        
        message = response.choices[0].message
        tool_calls = message.tool_calls
        
        if not tool_calls:
            print(f"[ERROR] No tool calls returned by model '{model_name}'.")
            return None
            
        tool_call = tool_calls[0]
        arguments_json = json.loads(tool_call.function.arguments)
        return arguments_json
        
    except Exception as e:
        print(f"[ERROR] Error communicating with model '{model_name}': {e}")
        return None

def chunk_menu(menu_text):
    """Parses menu_descriptions.md into structured set and overview chunks using custom regex."""
    chunks = []

    # 1. Extract General Service Information if present
    gen_info_match = re.search(
        r"### General Service Information.*?\n(.*)", 
        menu_text, 
        re.DOTALL
    )
    common_footnotes = []
    if gen_info_match:
        footnote_text = gen_info_match.group(1).strip()
        common_footnotes = [
            line.strip("- ").strip() 
            for line in footnote_text.split("\n") 
            if line.strip()
        ]

    # Find each service section (stopping at next Service or General Information)
    service_pattern = r"### Service: (.*?)\n(.*?)(?=\n### Service:|\n### General Service Information|\Z)"
    services = re.findall(service_pattern, menu_text, re.DOTALL)

    for service_name, service_content in services:
        # Extract price
        price_match = re.search(r"Rs\.\s*(\d+)", service_name)
        price = int(price_match.group(1)) if price_match else None

        service_title = service_name.split("(Rs.")[0].strip()

        # Extract all sets
        set_pattern = rf"{re.escape(service_title)} Set (\d+) includes (.*?)(?=\n{re.escape(service_title)} Set \d+ includes|\nAll sets are served with:|\Z)"
        sets = re.findall(set_pattern, service_content, re.DOTALL)

        # Common information
        common_match = re.search(
            r"All sets are served with:(.*?)(?=\nAlternatively,|\Z)",
            service_content,
            re.DOTALL,
        )
        common_items = common_match.group(1).strip() if common_match else ""

        # Alternative option
        alt_match = re.search(
            r"Alternatively,(.*?)(?=\n### Service:|\Z)",
            service_content,
            re.DOTALL,
        )
        alt_option = alt_match.group(1).strip() if alt_match else ""

        # Create one chunk per set
        for set_no, menu_items in sets:
            set_num = int(set_no)
            slug = service_title.lower().replace(" ", "_").replace("&", "and")
            chunk_id = f"{slug}_set_{set_num}"

            veg_items = [item.strip() for item in menu_items.split(",") if item.strip()]
            common_items_list = [item.strip() for item in common_items.split(",") if item.strip()]
            
            chunk = {
                "meal_type": service_title,
                "set_number": set_num,
                "price": price,
                "region": "South Central Zone",
                "train_class": "Sleeper Class (SL)",
                "veg_items": veg_items,
                "nonveg_items": [item.strip() for item in alt_option.replace("any set can be replaced with the non-vegetarian option:", "").split(",") if item.strip()] if alt_option else [],
                "common_items": common_items_list,
                "has_nonveg_option": bool(alt_option),
                "chunk_id": chunk_id,
                "chunk_text": f"""
Service: {service_title}
Set: {set_no}
Items: {menu_items.strip()}

Common Items:
{common_items}

Alternative Option:
{alt_option}
""".strip()
            }
            chunks.append(chunk)

        # Service-level chunk (overview)
        slug = service_title.lower().replace(" ", "_").replace("&", "and")
        chunks.append({
            "meal_type": service_title,
            "set_number": None,
            "price": price,
            "region": "South Central Zone",
            "train_class": "Sleeper Class (SL)",
            "veg_items": [],
            "nonveg_items": [],
            "common_items": [item.strip() for item in common_items.split(",") if item.strip()] if common_items else [],
            "has_nonveg_option": bool(alt_option),
            "chunk_id": f"{slug}_overview",
            "chunk_text": f"""
Service: {service_title}

Common Items:
{common_items}

Alternative Option:
{alt_option}
""".strip()
        })

    # Add General Service Information chunk
    if common_footnotes:
        chunks.append({
            "meal_type": "General Information",
            "set_number": None,
            "price": 0,
            "region": "South Central Zone",
            "train_class": "Sleeper Class (SL)",
            "veg_items": [],
            "nonveg_items": [],
            "common_items": [],
            "has_nonveg_option": False,
            "chunk_id": "service_notes",
            "chunk_text": "General Service Information\n\nThe following policies apply to all meal types and sets:\n" + "\n".join(f"- {note}" for note in common_footnotes)
        })

    return chunks


def generate_nl_descriptions(extracted_data):
    """Generate search-optimized sentences from structured tool call fields, including extracted footnotes."""
    services = extracted_data.get("services", [])
    all_descriptions = []
    
    for service in services:
        name = service.get("service_name")
        price = service.get("price")
        common = service.get("common_items", [])
        has_nonveg = service.get("has_nonveg_option", False)
        nonveg = service.get("nonveg_items", [])
        
        output_lines = [
            f"### Service: {name} (Rs. {price})",
            f"The price of {name} is Rs. {price} inclusive of all taxes."
        ]
        
        # Construct set sentences
        for s in service.get("sets", []):
            set_no = s.get("set_no")
            veg_list = s.get("veg_items", [])
            if veg_list:
                veg = ", ".join(veg_list)
                output_lines.append(f"{name} Set {set_no} includes {veg}.")
            
        if common:
            common_str = ", ".join(common)
            output_lines.append(f"All sets are served with: {common_str}.")
            
        if has_nonveg and nonveg:
            nonveg_str = ", ".join(nonveg)
            output_lines.append(f"Alternatively, any set can be replaced with the non-vegetarian option: {nonveg_str}.")
            
        all_descriptions.append("\n".join(output_lines))
        
    # Append the dynamically extracted global service notes
    footnote_lines = extracted_data.get("global_footnotes", [])
    if footnote_lines:
        notes_str = "\n".join(f"- {note}" for note in footnote_lines)
        service_notes = (
            f"### General Service Information & Special Meals\n"
            f"The following policies apply across all services and rotating sets:\n"
            f"{notes_str}"
        )
        all_descriptions.append(service_notes)
        
    return "\n\n".join(all_descriptions)

def main():
    parser = argparse.ArgumentParser(description="Test tool-calling table serialization with different local models.")
    parser.add_argument(
        "--model", "-m",
        type=str,
        default=DEFAULT_OLLAMA_MODEL,
        help=f"The model to use for extraction (e.g. gemma2:2b, qwen2.5-coder:7b, gemini-1.5-flash)"
    )
    args = parser.parse_args()
    
    if not PDF_PATH.exists():
        print(f"Error: PDF not found at {PDF_PATH}")
        exit(1)
        
    table_text = extract_raw_table_text()
    
    extracted_fields = run_tool_calling_ingestion(table_text, args.model)
    if extracted_fields:
        print(f"\n[OK] Structured Tool Call Arguments received from '{args.model}':")
        print(json.dumps(extracted_fields, indent=2))
        
        # 1. Generate and save the full natural language descriptions (before chunking)
        descriptions_text = generate_nl_descriptions(extracted_fields)
        desc_path = OUTPUT_PATH.parent / "menu_descriptions.md"
        with open(desc_path, "w", encoding="utf-8") as f:
            f.write(descriptions_text)
        print(f"\n[OK] Successfully saved the full menu descriptions to: {desc_path}")
        
        # 2. Build standard RAG chunks and write to data/chunks_llm.json
        chunks = chunk_menu(descriptions_text)
        OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
            json.dump(chunks, f, indent=2, ensure_ascii=False)
            
        print(f"[OK] Successfully structured {len(chunks)} chunks and saved them to: {OUTPUT_PATH}")

if __name__ == "__main__":
    main()

