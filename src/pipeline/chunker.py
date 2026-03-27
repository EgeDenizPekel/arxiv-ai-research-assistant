import tiktoken

CHUNK_SIZE = 512   # tokens
CHUNK_OVERLAP = 50  # tokens

_enc = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    return len(_enc.encode(text))


def chunk_sections(sections: list[dict]) -> list[dict]:
    """
    Convert a parsed document's sections into retrieval-ready chunks.

    For sections that fit within CHUNK_SIZE tokens, keep them as a single chunk.
    For longer sections, split with a sliding window and CHUNK_OVERLAP token overlap.

    Each returned chunk dict has keys: text, section, token_count.
    """
    chunks: list[dict] = []

    for section in sections:
        text = section["text"].strip()
        if not text:
            continue

        heading = section.get("heading", "")
        token_count = count_tokens(text)

        if token_count <= CHUNK_SIZE:
            chunks.append(
                {
                    "text": text,
                    "section": heading,
                    "token_count": token_count,
                }
            )
        else:
            for piece in _split_by_tokens(text):
                chunks.append(
                    {
                        "text": piece,
                        "section": heading,
                        "token_count": count_tokens(piece),
                    }
                )

    return chunks


def _split_by_tokens(text: str) -> list[str]:
    """Split text into overlapping token windows."""
    tokens = _enc.encode(text)
    pieces: list[str] = []
    start = 0

    while start < len(tokens):
        end = min(start + CHUNK_SIZE, len(tokens))
        pieces.append(_enc.decode(tokens[start:end]))
        if end == len(tokens):
            break
        start += CHUNK_SIZE - CHUNK_OVERLAP

    return pieces
