from pathlib import Path

import arxiv
from loguru import logger

PDF_DIR = Path("data/pdfs")

# Papers from 2020-2024 across the three ML/AI/NLP categories
SEARCH_QUERY = "cat:cs.LG OR cat:cs.AI OR cat:cs.CL"


def fetch_papers(max_papers: int = 100, skip_ids: set[str] | None = None) -> list[dict]:
    """
    Fetch papers from ArXiv and download their PDFs.

    Args:
        max_papers: Maximum number of new papers to fetch.
        skip_ids: ArXiv IDs already in the database - these are skipped.

    Returns:
        List of paper metadata dicts for successfully downloaded papers.
    """
    PDF_DIR.mkdir(parents=True, exist_ok=True)
    skip_ids = skip_ids or set()

    client = arxiv.Client(
        page_size=50,
        delay_seconds=3.0,  # respect ArXiv rate limits
        num_retries=3,
    )
    search = arxiv.Search(
        query=SEARCH_QUERY,
        max_results=max_papers + len(skip_ids),  # overfetch to account for skips
        sort_by=arxiv.SortCriterion.SubmittedDate,
        sort_order=arxiv.SortOrder.Descending,
    )

    papers = []
    fetched = 0

    for result in client.results(search):
        if fetched >= max_papers:
            break

        arxiv_id = result.entry_id.split("/")[-1]

        if arxiv_id in skip_ids:
            logger.debug(f"Skipping {arxiv_id} (already in DB)")
            continue

        pdf_path = PDF_DIR / f"{arxiv_id}.pdf"

        if not pdf_path.exists():
            try:
                result.download_pdf(dirpath=str(PDF_DIR), filename=f"{arxiv_id}.pdf")
                logger.info(f"Downloaded {arxiv_id}: {result.title[:60]}")
            except Exception as e:
                logger.warning(f"Failed to download {arxiv_id}: {e}")
                continue

        papers.append(
            {
                "arxiv_id": arxiv_id,
                "title": result.title.replace("\n", " ").strip(),
                "authors": ", ".join(a.name for a in result.authors[:10]),
                "abstract": result.summary.replace("\n", " ").strip(),
                "year": result.published.year,
                "categories": ", ".join(result.categories),
                "pdf_path": str(pdf_path),
            }
        )
        fetched += 1

    logger.info(f"Fetched {len(papers)} new papers")
    return papers
