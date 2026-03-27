import statistics

import fitz  # pymupdf
from loguru import logger


def parse_pdf(pdf_path: str) -> dict:
    """
    Extract text and section structure from an ArXiv PDF.

    Strategy:
    - Compute the median font size across the document (body text baseline).
    - Treat any block that is >=10% larger than the median, short (<100 chars),
      and does not end with a period as a section heading.
    - Group body text under its nearest preceding heading.

    Returns:
        {
            "sections": [{"heading": str, "text": str}, ...],
            "full_text": str,   # concatenated heading + body for each section
        }
    """
    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        logger.error(f"Could not open {pdf_path}: {e}")
        return {"sections": [], "full_text": ""}

    # First pass: collect all font sizes to establish the body text baseline
    all_sizes: list[float] = []
    for page in doc:
        for block in page.get_text("dict")["blocks"]:
            if block["type"] != 0:
                continue
            for line in block["lines"]:
                for span in line["spans"]:
                    if span["text"].strip():
                        all_sizes.append(span["size"])

    if not all_sizes:
        doc.close()
        return {"sections": [], "full_text": ""}

    body_size = statistics.median(all_sizes)
    heading_threshold = body_size * 1.1

    # Second pass: build section list
    sections: list[dict] = []
    current: dict = {"heading": "preamble", "text": ""}

    for page in doc:
        for block in page.get_text("dict")["blocks"]:
            if block["type"] != 0:
                continue

            block_text = ""
            max_size = 0.0
            for line in block["lines"]:
                for span in line["spans"]:
                    text = span["text"].strip()
                    if text:
                        block_text += text + " "
                        max_size = max(max_size, span["size"])

            block_text = block_text.strip()
            if not block_text:
                continue

            if _is_heading(block_text, max_size, heading_threshold):
                if current["text"].strip():
                    sections.append(current)
                current = {"heading": block_text, "text": ""}
            else:
                current["text"] += " " + block_text

    if current["text"].strip():
        sections.append(current)

    doc.close()

    full_text = "\n\n".join(
        f"{s['heading']}\n{s['text'].strip()}" for s in sections
    )

    return {"sections": sections, "full_text": full_text}


def _is_heading(text: str, font_size: float, threshold: float) -> bool:
    """Heuristic: large font, short, does not end like a sentence."""
    if font_size < threshold:
        return False
    if len(text) > 100:
        return False
    if text.endswith("."):
        return False
    # Reject lines that are clearly references or URLs
    if text.startswith("http") or text.startswith("["):
        return False
    return True
