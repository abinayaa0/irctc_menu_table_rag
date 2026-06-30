"""
IRCTC South Central Menu — Unified RAG Pipeline.

Usage:
  python main.py            Parse PDF, build index, then start query loop
  python main.py --query "I have Rs.60 what can I get?"
  python main.py --parse-only
  python main.py --build-only
  python main.py --query-only
"""

import os
import sys
import argparse

os.environ["USE_TF"] = "0"
os.environ["USE_TORCH"] = "1"

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from rich.console import Console
from rich.panel import Panel

console = Console()


def step_parse():
    console.print(Panel("Step 1: Parsing PDF into chunks", border_style="blue"))
    from parse_pdf import main as parse_main
    parse_main()


def step_build():
    console.print(Panel("Step 2: Building Qdrant index from chunks", border_style="blue"))
    from build_index import main as build_main
    build_main()


def step_query(args_query: str | None = None):
    console.print(Panel("Step 3: Starting query assistant", border_style="blue"))
    from query import main as query_main
    sys.argv = [sys.argv[0]]
    if args_query:
        sys.argv += ["--query", args_query]
    query_main()


def main():
    parser = argparse.ArgumentParser(description="IRCTC South Central Menu RAG Pipeline")
    parser.add_argument("--query", "-q", type=str, help="Single query and exit")
    parser.add_argument("--parse-only", action="store_true", help="Only run PDF parsing")
    parser.add_argument("--build-only", action="store_true", help="Only run index building")
    parser.add_argument("--query-only", action="store_true", help="Only start query loop")
    args = parser.parse_args()

    if args.parse_only:
        step_parse()
    elif args.build_only:
        step_build()
    elif args.query_only:
        step_query(args.query)
    elif args.query:
        step_parse()
        step_build()
        step_query(args.query)
    else:
        step_parse()
        step_build()
        step_query()


if __name__ == "__main__":
    main()
