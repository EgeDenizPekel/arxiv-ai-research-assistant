"""
Generate an initial eval dataset of question/ground-truth pairs.

Samples random chunks from SQLite, asks GPT-4o-mini to produce a
question + ground-truth answer for each chunk, then saves to
eval_dataset.json for manual review.

Usage:
    poetry run python scripts/generate_eval_dataset.py
    poetry run python scripts/generate_eval_dataset.py --n 150 --out eval_dataset_raw.json
"""

import argparse
import json
import os
import random
import sqlite3
import sys
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger
from openai import OpenAI
from tqdm import tqdm

load_dotenv()

DB_PATH = Path("data/papers.db")
DEFAULT_N = 120          # generate more than needed so you can prune during review
DEFAULT_OUT = Path("eval_dataset.json")
MODEL = "gpt-4o-mini"

SYSTEM_PROMPT = """\
You are a research assistant creating an evaluation dataset for a RAG system over ArXiv ML/AI papers.

Given a passage from a research paper, generate:
1. A specific, answerable question that requires understanding the passage.
2. A concise ground-truth answer (2-4 sentences) that is fully supported by the passage.

Rules:
- The question must be answerable ONLY from the given passage - no outside knowledge required.
- Prefer factual questions about methods, results, definitions, or comparisons.
- Avoid overly vague questions like "What is this paper about?".
- The answer must be grounded in the passage text.

Respond with valid JSON only:
{"question": "...", "ground_truth": "..."}"""


def _sample_chunks(n: int) -> list[dict]:
    """Sample n random chunks from the SQLite DB."""
    if not DB_PATH.exists():
        logger.error(f"Database not found at {DB_PATH}. Run the pipeline first.")
        sys.exit(1)

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    cur.execute("SELECT COUNT(*) FROM chunks")
    total = cur.fetchone()[0]
    logger.info(f"Total chunks in DB: {total}. Sampling {n}.")

    cur.execute(
        """
        SELECT c.id, c.text, c.section, c.chunk_index,
               p.arxiv_id, p.title
        FROM chunks c
        JOIN papers p ON c.paper_id = p.id
        WHERE LENGTH(c.text) > 300
        ORDER BY RANDOM()
        LIMIT ?
        """,
        (n,),
    )
    rows = [dict(r) for r in cur.fetchall()]
    conn.close()
    return rows


def _generate_pair(client: OpenAI, chunk: dict) -> dict | None:
    """Ask GPT-4o-mini to generate a Q&A pair from the chunk. Returns None on failure."""
    passage = chunk["text"][:1500]  # cap to avoid excessive token use
    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Passage:\n{passage}"},
            ],
            response_format={"type": "json_object"},
            temperature=0.7,
            max_tokens=300,
        )
        data = json.loads(response.choices[0].message.content)
        return {
            "question": data["question"],
            "ground_truth": data["ground_truth"],
            "source_arxiv_id": chunk.get("arxiv_id", ""),
            "source_section": chunk.get("section", ""),
        }
    except Exception as e:
        logger.warning(f"Failed to generate pair for chunk {chunk['id']}: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Generate eval dataset candidates")
    parser.add_argument("--n", type=int, default=DEFAULT_N, help="Number of candidates to generate")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="Output JSON path")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    random.seed(args.seed)
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    chunks = _sample_chunks(args.n)
    pairs = []

    for chunk in tqdm(chunks, desc="Generating Q&A pairs", unit="chunk"):
        pair = _generate_pair(client, chunk)
        if pair:
            pairs.append(pair)

    logger.info(f"Generated {len(pairs)} pairs from {len(chunks)} chunks.")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(pairs, f, indent=2)

    logger.success(
        f"Saved to {args.out}.\n"
        f"Next step: manually review the file, remove low-quality pairs, "
        f"and keep 75-100 high-quality examples."
    )


if __name__ == "__main__":
    main()
