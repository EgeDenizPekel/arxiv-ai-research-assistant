from loguru import logger

from .chunker import chunk_sections
from .database import get_existing_arxiv_ids, init_db, insert_chunks, insert_paper
from .fetcher import fetch_papers
from .parser import parse_pdf


def run_pipeline(max_papers: int = 100) -> dict:
    """
    Full data pipeline: fetch -> parse -> chunk -> store.

    Args:
        max_papers: How many new papers to fetch and process.

    Returns:
        Summary dict with counts of papers and chunks processed.
    """
    logger.info("Initializing database")
    init_db()

    existing_ids = get_existing_arxiv_ids()
    logger.info(f"Found {len(existing_ids)} papers already in DB - will skip these")

    logger.info(f"Fetching up to {max_papers} new papers from ArXiv")
    papers = fetch_papers(max_papers=max_papers, skip_ids=existing_ids)

    total_chunks = 0
    failed_parse = 0

    for paper in papers:
        arxiv_id = paper["arxiv_id"]
        logger.info(f"Processing {arxiv_id}: {paper['title'][:60]}")

        parsed = parse_pdf(paper["pdf_path"])

        if not parsed["sections"]:
            logger.warning(f"No sections extracted from {arxiv_id} - skipping")
            failed_parse += 1
            continue

        chunks = chunk_sections(parsed["sections"])

        if not chunks:
            logger.warning(f"No chunks produced for {arxiv_id} - skipping")
            failed_parse += 1
            continue

        paper_id = insert_paper(paper)
        insert_chunks(paper_id, arxiv_id, chunks)

        logger.info(f"  Stored {len(chunks)} chunks for {arxiv_id}")
        total_chunks += len(chunks)

    summary = {
        "papers_fetched": len(papers),
        "papers_stored": len(papers) - failed_parse,
        "papers_failed_parse": failed_parse,
        "chunks_stored": total_chunks,
    }

    logger.info(
        f"Pipeline complete: {summary['papers_stored']} papers, "
        f"{summary['chunks_stored']} chunks stored"
    )
    return summary
