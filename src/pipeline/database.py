import sqlite3
from pathlib import Path

DB_PATH = Path("data/papers.db")


def get_connection() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    with get_connection() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS papers (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                arxiv_id    TEXT UNIQUE NOT NULL,
                title       TEXT NOT NULL,
                authors     TEXT,
                abstract    TEXT,
                year        INTEGER,
                categories  TEXT,
                pdf_path    TEXT,
                parsed_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS chunks (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                paper_id    INTEGER NOT NULL REFERENCES papers(id),
                arxiv_id    TEXT NOT NULL,
                chunk_index INTEGER NOT NULL,
                text        TEXT NOT NULL,
                section     TEXT,
                token_count INTEGER,
                created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            CREATE INDEX IF NOT EXISTS idx_chunks_arxiv_id ON chunks(arxiv_id);
            CREATE INDEX IF NOT EXISTS idx_chunks_paper_id ON chunks(paper_id);
        """)


def get_existing_arxiv_ids() -> set[str]:
    with get_connection() as conn:
        rows = conn.execute("SELECT arxiv_id FROM papers").fetchall()
    return {row["arxiv_id"] for row in rows}


def insert_paper(paper: dict) -> int:
    with get_connection() as conn:
        cursor = conn.execute(
            """
            INSERT OR IGNORE INTO papers (arxiv_id, title, authors, abstract, year, categories, pdf_path)
            VALUES (:arxiv_id, :title, :authors, :abstract, :year, :categories, :pdf_path)
            """,
            paper,
        )
        conn.commit()
        if cursor.lastrowid:
            return cursor.lastrowid
        # Paper already existed - fetch its id
        row = conn.execute(
            "SELECT id FROM papers WHERE arxiv_id = ?", (paper["arxiv_id"],)
        ).fetchone()
        return row["id"]


def insert_chunks(paper_id: int, arxiv_id: str, chunks: list[dict]) -> None:
    with get_connection() as conn:
        conn.executemany(
            """
            INSERT INTO chunks (paper_id, arxiv_id, chunk_index, text, section, token_count)
            VALUES (:paper_id, :arxiv_id, :chunk_index, :text, :section, :token_count)
            """,
            [
                {
                    "paper_id": paper_id,
                    "arxiv_id": arxiv_id,
                    "chunk_index": i,
                    "text": chunk["text"],
                    "section": chunk["section"],
                    "token_count": chunk["token_count"],
                }
                for i, chunk in enumerate(chunks)
            ],
        )
        conn.commit()


def get_all_chunks() -> list[dict]:
    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT c.id, c.arxiv_id, c.chunk_index, c.text, c.section, c.token_count,
                   p.title, p.authors, p.year
            FROM chunks c
            JOIN papers p ON p.id = c.paper_id
            ORDER BY c.paper_id, c.chunk_index
            """
        ).fetchall()
    return [dict(row) for row in rows]
