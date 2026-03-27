from loguru import logger

from .chunker import chunk_sections
from .database import get_existing_arxiv_ids, init_db, insert_chunks, insert_paper
from .fetcher import fetch_papers


def _build_sections(paper: dict) -> list[dict]:
    """
    Convert a paper dict from the HuggingFace dataset into the section format
    expected by chunk_sections(): [{"heading": str, "text": str}, ...]

    If section_names are available, we split the article text at each heading.
    Otherwise the entire article is treated as a single section.
    """
    full_text: str = paper.get("full_text", "").strip()
    section_names: list[str] = paper.get("section_names") or []

    if not full_text:
        return []

    if not section_names:
        return [{"heading": "body", "text": full_text}]

    # Split article text at section heading occurrences (case-insensitive)
    sections: list[dict] = []
    remaining = full_text

    for i, heading in enumerate(section_names):
        heading_stripped = heading.strip()
        idx = remaining.lower().find(heading_stripped.lower())

        if idx == -1:
            # Heading not found in remaining text - append as empty to preserve order
            sections.append({"heading": heading_stripped, "text": ""})
            continue

        # Text before this heading belongs to the previous section (or preamble)
        before = remaining[:idx].strip()
        if before:
            if sections:
                sections[-1]["text"] += " " + before
            else:
                sections.append({"heading": "preamble", "text": before})

        remaining = remaining[idx + len(heading_stripped):].strip()
        sections.append({"heading": heading_stripped, "text": ""})

    # Remaining text after the last heading goes into the last section
    if remaining and sections:
        sections[-1]["text"] += " " + remaining
    elif remaining:
        sections.append({"heading": "body", "text": remaining})

    # Drop sections with no text
    return [s for s in sections if s["text"].strip()]


def run_pipeline(max_papers: int = 100) -> dict:
    """
    Full data pipeline: load from HuggingFace -> chunk -> store.

    Args:
        max_papers: How many new papers to fetch and process.

    Returns:
        Summary dict with counts of papers and chunks processed.
    """
    logger.info("Initializing database")
    init_db()

    existing_ids = get_existing_arxiv_ids()
    logger.info(f"Found {len(existing_ids)} papers already in DB - will skip these")

    logger.info(f"Loading up to {max_papers} new papers from HuggingFace")
    papers = fetch_papers(max_papers=max_papers, skip_ids=existing_ids)

    total_chunks = 0
    failed = 0

    for paper in papers:
        arxiv_id = paper["arxiv_id"]

        sections = _build_sections(paper)
        if not sections:
            logger.warning(f"No text extracted for {arxiv_id} - skipping")
            failed += 1
            continue

        chunks = chunk_sections(sections)
        if not chunks:
            logger.warning(f"No chunks produced for {arxiv_id} - skipping")
            failed += 1
            continue

        # Strip the extra keys before inserting into DB
        db_paper = {k: v for k, v in paper.items() if k not in ("full_text", "section_names")}
        paper_id = insert_paper(db_paper)
        insert_chunks(paper_id, arxiv_id, chunks)

        logger.info(f"Stored {len(chunks)} chunks for {arxiv_id}")
        total_chunks += len(chunks)

    summary = {
        "papers_fetched": len(papers),
        "papers_stored": len(papers) - failed,
        "papers_failed": failed,
        "chunks_stored": total_chunks,
    }

    logger.info(
        f"Pipeline complete: {summary['papers_stored']} papers, "
        f"{summary['chunks_stored']} chunks stored"
    )
    return summary
