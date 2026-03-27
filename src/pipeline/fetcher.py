"""
Fetches paper text and metadata from the HuggingFace `scientific_papers` dataset
(arxiv config). This dataset contains ~200K ArXiv ML papers with full body text,
abstracts, and section names.

We use this instead of the ArXiv API directly because the legacy export.arxiv.org
API aggressively rate-limits category search queries.

Dataset fields: article (str), abstract (str), section_names (newline-separated str)
Note: no article_id field - we generate stable IDs from a hash of the abstract.
"""

import hashlib

from loguru import logger


def fetch_papers(max_papers: int = 100, skip_ids: set[str] | None = None) -> list[dict]:
    """
    Stream papers from the HuggingFace `scientific_papers` arxiv dataset.

    Each returned dict has keys compatible with the SQLite `papers` schema:
        arxiv_id, title, authors, abstract, year, categories, pdf_path
    Plus extra keys consumed by pipeline.py but not stored in DB:
        full_text, section_names

    Args:
        max_papers: Maximum number of new papers to return.
        skip_ids: Paper IDs already in the database - these are skipped.

    Returns:
        List of paper dicts.
    """
    from datasets import load_dataset

    skip_ids = skip_ids or set()
    papers: list[dict] = []

    logger.info("Loading scientific_papers (arxiv) dataset from HuggingFace (streaming)")

    ds = load_dataset(
        "scientific_papers",
        "arxiv",
        split="train",
        streaming=True,
        trust_remote_code=True,
    )

    for example in ds:
        if len(papers) >= max_papers:
            break

        article: str = (example.get("article") or "").strip()
        abstract: str = (example.get("abstract") or "").strip()

        if not article:
            continue

        # No article_id in this dataset - generate a stable ID from the abstract hash
        arxiv_id = "hf_" + hashlib.md5(abstract.encode()).hexdigest()[:12]

        if arxiv_id in skip_ids:
            logger.debug(f"Skipping {arxiv_id} (already in DB)")
            continue

        # section_names is a newline-separated string, not a list
        raw_sections = example.get("section_names") or ""
        section_names = [s.strip() for s in raw_sections.split("\n") if s.strip()]

        title = _extract_title(abstract)

        papers.append(
            {
                "arxiv_id": arxiv_id,
                "title": title,
                "authors": "",
                "abstract": abstract,
                "year": 0,
                "categories": "cs.AI/cs.LG/cs.CL",
                "pdf_path": None,
                "full_text": article,
                "section_names": section_names,
            }
        )

        logger.info(f"Loaded {len(papers)}/{max_papers} - {arxiv_id}: {title[:60]}")

    logger.info(f"Loaded {len(papers)} papers from HuggingFace dataset")
    return papers


def _extract_title(abstract: str) -> str:
    """Best-effort title from the first sentence of the abstract."""
    if abstract:
        first = abstract.split(".")[0].strip()
        if 10 < len(first) < 120:
            return first
    return "Untitled Paper"
