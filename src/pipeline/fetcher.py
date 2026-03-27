"""
Fetches paper text and metadata from the HuggingFace `scientific_papers` dataset
(arxiv config). Papers are filtered to ML/AI topics via abstract keyword matching.

We use this instead of the ArXiv API directly because the legacy export.arxiv.org
API aggressively rate-limits category search queries, and the API only returns
abstracts (150-300 words), not full paper text.

Dataset fields: article (str), abstract (str), section_names (newline-separated str)
Note: no article_id field - we generate stable IDs from a hash of the abstract.
"""

import hashlib
import re

from loguru import logger

# ---------------------------------------------------------------------------
# ML/AI keyword filter
# ---------------------------------------------------------------------------

# At least one of these must appear in the abstract (case-insensitive) for a
# paper to be kept. Terms are chosen to cover all ML subfields while excluding
# pure physics, astronomy, chemistry, and biology papers that mention "neural"
# only in a biological context.
_ML_KEYWORDS = [
    "neural network",
    "deep learning",
    "machine learning",
    "transformer",
    "attention mechanism",
    "self-attention",
    "reinforcement learning",
    "convolutional neural",
    "recurrent neural",
    "language model",
    "natural language processing",
    "large language model",
    "computer vision",
    "generative adversarial",
    "variational autoencoder",
    "stochastic gradient",
    "backpropagation",
    "graph neural",
    "object detection",
    "image classification",
    "image segmentation",
    "text classification",
    "knowledge graph",
    "word embedding",
    "representation learning",
    "transfer learning",
    "semi-supervised learning",
    "self-supervised",
    "pre-trained model",
    "fine-tuning",
    "dropout regularization",
    "batch normalization",
    "question answering",
    "sentiment analysis",
    "named entity recognition",
    "speech recognition",
    "encoder-decoder",
    "seq2seq",
    "token classification",
    "masked language",
    "contrastive learning",
    "diffusion model",
    "score-based",
    "foundation model",
]

# Pre-compile for speed (streaming through 10K+ papers)
_ML_PATTERN = re.compile(
    "|".join(re.escape(kw) for kw in _ML_KEYWORDS),
    re.IGNORECASE,
)


def _is_ml_paper(abstract: str, full_text: str) -> bool:
    """Return True if the paper is ML/AI-related based on abstract keywords."""
    return bool(_ML_PATTERN.search(abstract) or _ML_PATTERN.search(full_text[:500]))


# ---------------------------------------------------------------------------
# LaTeX artifact cleaning
# ---------------------------------------------------------------------------

# The scientific_papers dataset replaces math with @xmath<N> and citations
# with @xcite (or @cite). These tokens are meaningless to the LLM and degrade
# retrieval quality. Strip them and collapse any resulting whitespace runs.
_LATEX_ARTIFACTS = re.compile(
    r"@xmath\d*"        # math placeholders: @xmath0, @xmath123, @xmath
    r"|@xcite\b"        # citation placeholders: @xcite
    r"|@cite\b"         # alternative citation form
    r"|\[ [^\]]{0,60} \]"  # short bracket refs like [ 1 ] or [ 15 , 16 ]
    r"|\s{2,}",         # collapse multiple spaces (cleanup after removal)
    re.IGNORECASE,
)


def _clean_text(text: str) -> str:
    """Strip LaTeX placeholder tokens and normalise whitespace."""
    cleaned = _LATEX_ARTIFACTS.sub(
        lambda m: " " if m.group(0).startswith(" ") or len(m.group(0)) > 3 else "",
        text,
    )
    # Final whitespace normalisation: collapse runs of spaces/newlines sensibly
    cleaned = re.sub(r" {2,}", " ", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


# ---------------------------------------------------------------------------
# Main fetch function
# ---------------------------------------------------------------------------

def fetch_papers(max_papers: int = 750, skip_ids: set[str] | None = None) -> list[dict]:
    """
    Stream papers from the HuggingFace `scientific_papers` arxiv dataset,
    keeping only ML/AI papers as determined by abstract keyword matching.

    Each returned dict has keys compatible with the SQLite `papers` schema:
        arxiv_id, title, authors, abstract, year, categories, pdf_path
    Plus extra keys consumed by pipeline.py but not stored in DB:
        full_text, section_names

    Args:
        max_papers: Maximum number of ML/AI papers to return.
        skip_ids:   Paper IDs already in the database - these are skipped.

    Returns:
        List of paper dicts, all ML/AI topic.
    """
    from datasets import load_dataset

    skip_ids = skip_ids or set()
    papers: list[dict] = []
    examined = 0

    logger.info("Loading scientific_papers (arxiv) dataset from HuggingFace (streaming)")
    logger.info(f"Will stream until {max_papers} ML/AI papers collected")

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

        examined += 1

        article: str = (example.get("article") or "").strip()
        abstract: str = (example.get("abstract") or "").strip()

        if not article or not abstract:
            continue

        # --- ML/AI filter ---
        if not _is_ml_paper(abstract, article):
            if examined % 500 == 0:
                logger.debug(f"Examined {examined} papers, collected {len(papers)} ML papers so far")
            continue

        # --- Stable ID from abstract hash ---
        arxiv_id = "hf_" + hashlib.md5(abstract.encode()).hexdigest()[:12]

        if arxiv_id in skip_ids:
            logger.debug(f"Skipping {arxiv_id} (already in DB)")
            continue

        # --- Clean LaTeX artifacts ---
        article_clean = _clean_text(article)
        abstract_clean = _clean_text(abstract)

        # section_names is a newline-separated string, not a list
        raw_sections = example.get("section_names") or ""
        section_names = [s.strip() for s in raw_sections.split("\n") if s.strip()]

        title = _extract_title(abstract_clean)

        papers.append(
            {
                "arxiv_id": arxiv_id,
                "title": title,
                "authors": "",
                "abstract": abstract_clean,
                "year": 0,
                "categories": "cs.AI/cs.LG/cs.CL",
                "pdf_path": None,
                "full_text": article_clean,
                "section_names": section_names,
            }
        )

        logger.info(
            f"[{len(papers)}/{max_papers}] {arxiv_id}: {title[:70]} "
            f"(examined {examined})"
        )

    logger.info(
        f"Loaded {len(papers)} ML/AI papers after examining {examined} total papers "
        f"({100 * len(papers) / max(examined, 1):.1f}% hit rate)"
    )
    return papers


def _extract_title(abstract: str) -> str:
    """Best-effort title from the first sentence of the abstract."""
    if abstract:
        first = abstract.split(".")[0].strip()
        if 10 < len(first) < 120:
            return first
    return "Untitled Paper"
