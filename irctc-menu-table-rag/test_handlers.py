"""
Full test of deterministic handlers — no models, no Qdrant needed.
Tests every query type that has been reported as wrong.
"""
import json, re, sys

# ── inline the functions from app_llm.py (no model/qdrant imports needed) ──

PRICE_TABLE = {"Morning Tea": 15, "Evening Snacks": 50, "Breakfast": 65, "Lunch & Dinner": 120}

MEAL_TYPE_TERMS = {
    "breakfast", "lunch", "dinner", "tea", "snacks", "snack",
    "morning tea", "evening snacks", "morning", "evening",
}
STOPWORDS = {
    "and","or","the","with","in","on","at","to","a","an","of","for","is","are",
    "was","be","by","as","from","this","that","items","item","set","sets","menu",
    "meal","meals","serving","served","include","includes","have","has","does","which","what",
}

def _normalize(text):
    n = text.lower().strip()
    for a, b in [('bh','b'),('dh','d'),('gh','g'),('sh','s'),('th','t'),('ph','f')]:
        n = n.replace(a, b)
    n = re.sub(r'ly$','li',n)
    n = re.sub(r'ey$','i',n)
    return n.replace('oo','u').replace('ee','i')

def _add_to_index(index, key, display, meal_type, set_num):
    if key not in index:
        index[key] = {"display": display, "meal_sets": {}}
    if meal_type not in index[key]["meal_sets"]:
        index[key]["meal_sets"][meal_type] = []
    if set_num not in index[key]["meal_sets"][meal_type]:
        index[key]["meal_sets"][meal_type].append(set_num)

def build_food_index(path):
    index = {}
    for chunk in json.load(open(path, encoding='utf-8')):
        set_num = chunk.get('set_number')
        meal_type = chunk.get('meal_type', '')
        if not set_num or not meal_type:          # FIXED: no meal_type name filter
            continue
        for raw in re.split(r'[+,\n•;\-]', chunk.get('chunk_text','')):
            item = re.sub(r'^[^:]+:\s*', '', raw.strip())
            item = re.sub(r'\s+', ' ', item).strip(' .')
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

def lookup_item_in_query(query_text, food_index):
    ql = query_text.lower()
    if not re.search(r'\b(which|what|where|does|do|is|are|have|has|include|contain|serve|served|available|find)\b', ql):
        return None
    if re.search(r'\brs\.?\s*\d+|\d+\s*rs', ql):
        return None
    q_norm = _normalize(query_text)
    matched = [(k, v) for k, v in food_index.items()
               if re.search(r'\b' + re.escape(k) + r'\b', q_norm)]
    if not matched:
        return None
    matched.sort(key=lambda x: len(x[0]), reverse=True)
    item_norm, data = matched[0]
    item_display = data['display'].title()
    lines = [f"{item_display} is served in:"]
    for mt, sets in sorted(data['meal_sets'].items()):
        sets = sorted(set(sets))
        s = "s" if len(sets) > 1 else ""
        lines.append(f"  - {mt} (Rs.{PRICE_TABLE.get(mt,'?')}): Set{s} {', '.join(map(str,sets))}")
    return '\n'.join(lines)

def handle_special_queries(query_text):
    q = query_text.lower()
    set_nums = [int(n) for n in re.findall(r"\bset\s*(\d+)\b", query_text, re.IGNORECASE)]
    invalid = [n for n in set_nums if n < 1 or n > 7]
    if invalid:
        return f"Only Sets 1-7 exist. Set {invalid[0]} does not exist."
    if re.search(r"\b(jain|diabetic)\b", q):
        return "Jain/diabetic food available on request from train staff."
    BUDGET_PATTERN = (
        r"rs\.?\s*(\d+)|(\d+)\s*rs\.?"
        r"|(?:with|have|got|for|get|budget|only|just|spend|using|under|upto|up to"
        r"|rupees?|inr|afford|pay|cost|costs)\s+(\d+)"
    )
    m = re.search(BUDGET_PATTERN, q)
    budget = None
    if m:
        budget = int(next(g for g in m.groups() if g is not None))
    else:
        bwords = ["get","buy","afford","eat","have","pay","cost","order","meal","food"]
        if any(w in q for w in bwords):
            set_strs = {str(n) for n in re.findall(r"\bset\s*(\d+)\b", q, re.IGNORECASE)}
            nums = [int(n) for n in re.findall(r"\b(\d+)\b", q)
                    if n not in set_strs and 1 <= int(n) <= 10000]
            if nums:
                budget = max(nums)
    if budget is not None:
        has_morning = any(k in q for k in ["morning","breakfast","am"]) and not any(k in q for k in ["evening","pm","dinner"])
        has_evening = any(k in q for k in ["evening","snacks","pm"]) and not any(k in q for k in ["morning","breakfast","lunch"])
        has_lunch   = any(k in q for k in ["lunch","dinner","night","noon"])
        cands = list(PRICE_TABLE.items())
        if has_morning: cands = [(n,p) for n,p in cands if n in ["Morning Tea","Breakfast"]]
        elif has_evening: cands = [(n,p) for n,p in cands if n == "Evening Snacks"]
        elif has_lunch:  cands = [(n,p) for n,p in cands if n == "Lunch & Dinner"]
        affordable = [(n,p) for n,p in sorted(cands, key=lambda x:x[1]) if p <= budget]
        if affordable:
            lines = [f"With Rs.{budget} you can afford:"]
            for name, price in affordable:
                lines.append(f"  - {name} (Rs.{price})")
            return "\n".join(lines)
        else:
            return f"With Rs.{budget}, you cannot afford any meal. Cheapest is Morning Tea at Rs.15."
    return None

# ── Build index and run all tests ──

food_index = build_food_index('data/chunks_llm.json')
print(f"Food index built: {len(food_index)} entries\n")
print("=" * 65)

TESTS = [
    # (query, expected_contains, should_use_deterministic)
    # Budget queries
    ("I have Rs.50 what can I get?",               "Evening Snacks",              True),
    ("what will you get for 150rs?",               "Morning Tea",                 True),
    ("what will you get for 125rs?",               "Morning Tea",                 True),
    ("what will you get for 10rs?",                "cannot afford",               True),
    # Item queries
    ("Which meal includes sambar?",                "Breakfast",                   True),
    ("Which meal includes sambhar?",               "Breakfast",                   True),
    ("Which sets includes sambar for lunch?",      "Sambhar",                     True),
    ("Which sets include upma?",                   "Set 1",                       True),
    ("What sets include upma?",                    "Set 1",                       True),
    ("Which sets have poha for breakfast?",        None,                          False),  # not on menu → None → LLM
    ("Is rice served in all dinner sets?",         "Lunch & Dinner",              True),
    ("What sets have medu vada?",                  "Sets 1, 3, 5, 6",             True),
    # Invalid set
    ("What is in Set 9 breakfast?",                "Set 9 does not exist",        True),
]

passed = 0
failed = 0
for query, expected, det in TESTS:
    special = handle_special_queries(query)
    item_ans = lookup_item_in_query(query, food_index)
    answer = special or item_ans

    det_fired = answer is not None
    correct = True

    if det and expected:
        correct = expected.lower() in (answer or "").lower()
    elif not det:
        correct = answer is None  # should NOT fire deterministic

    status = "PASS" if correct else "FAIL"
    if correct:
        passed += 1
    else:
        failed += 1

    print(f"{status} | {query}")
    print(f"       det_fired={det_fired} | answer={repr((answer or 'None')[:80])}")
    print()

print("=" * 65)
print(f"Results: {passed} passed, {failed} failed out of {len(TESTS)} tests")
