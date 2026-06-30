"""Verify food index after fix."""
import json, re

MEAL_TYPE_TERMS = {
    "breakfast", "lunch", "dinner", "tea", "snacks", "snack",
    "morning tea", "evening snacks", "morning", "evening",
}
STOPWORDS = {
    "and","or","the","with","in","on","at","to","a","an","of","for","is","are",
    "was","be","by","as","from","this","that","items","item","set","sets","menu",
    "meal","meals","serving","served","include","includes","have","has","does","which","what",
}
PRICE_TABLE = {"Morning Tea": 15, "Evening Snacks": 50, "Breakfast": 65, "Lunch & Dinner": 120}

def _normalize(text):
    n = text.lower().strip()
    for a, b in [('bh','b'),('dh','d'),('gh','g'),('sh','s'),('th','t'),('ph','f')]:
        n = n.replace(a, b)
    n = re.sub(r'ly$','li',n)
    n = re.sub(r'ey$','i',n)
    n = n.replace('oo','u').replace('ee','i')
    return n

def _add_to_index(index, key, display, meal_type, set_num):
    if key not in index:
        index[key] = {"display": display, "meal_sets": {}}
    if meal_type not in index[key]["meal_sets"]:
        index[key]["meal_sets"][meal_type] = []
    if set_num not in index[key]["meal_sets"][meal_type]:
        index[key]["meal_sets"][meal_type].append(set_num)

def build_food_index(path):
    index = {}
    chunks = json.load(open(path, encoding='utf-8'))
    for chunk in chunks:
        set_num = chunk.get('set_number')
        meal_type = chunk.get('meal_type','')
        text = chunk.get('chunk_text','')
        if not set_num or not meal_type:   # FIXED: no longer skipping by meal_type name
            continue
        raw_parts = re.split(r'[+,\n•;\-]', text)
        for raw in raw_parts:
            item = re.sub(r'^[^:]+:\s*','', raw.strip())
            item = re.sub(r'\s+',' ', item).strip(' .')
            if not item or len(item) < 3:
                continue
            if item.lower() in MEAL_TYPE_TERMS or item.lower() in STOPWORDS:
                continue
            item_norm = _normalize(item)
            if len(item_norm) >= 3:
                _add_to_index(index, item_norm, item, meal_type, set_num)
            for word in item.split():
                wn = _normalize(word)
                if len(wn) >= 4 and wn not in STOPWORDS and wn not in MEAL_TYPE_TERMS:
                    _add_to_index(index, wn, word, meal_type, set_num)
    return index

def lookup(query_text, food_index):
    ql = query_text.lower()
    if not re.search(r'\b(which|what|where|does|do|is|are|have|has|include|contain|serve|served|available|find)\b', ql):
        return None
    if re.search(r'\brs\.?\s*\d+|\d+\s*rs', ql):
        return None
    q_norm = _normalize(query_text)
    matched = [(k, v) for k,v in food_index.items() if re.search(r'\b'+re.escape(k)+r'\b', q_norm)]
    if not matched:
        return "NO MATCH"
    matched.sort(key=lambda x: len(x[0]), reverse=True)
    item_norm, data = matched[0]
    item_display = data['display'].title()
    lines = [f"{item_display} is served in:"]
    for mt, sets in sorted(data['meal_sets'].items()):
        sets = sorted(set(sets))
        lines.append(f"  - {mt} (Rs.{PRICE_TABLE.get(mt,'?')}): Sets {', '.join(map(str,sets))}")
    return '\n'.join(lines)

index = build_food_index('data/chunks_llm.json')
print(f"Index size: {len(index)} entries\n")

# Show sambhar entry
key = _normalize("sambhar")
print(f"Index entry for '{key}':")
if key in index:
    print(f"  display: {index[key]['display']}")
    print(f"  meal_sets: {index[key]['meal_sets']}")
else:
    print("  NOT FOUND")

print()
tests = [
    "Which meal includes sambar?",
    "Which sets have poha for breakfast?",
    "What sets include upma?",
    "What sets have medu vada?",
    "Is rice served in all dinner sets?",
]
for q in tests:
    result = lookup(q, index)
    print(f"Q: {q}")
    print(f"A: {result}\n")
