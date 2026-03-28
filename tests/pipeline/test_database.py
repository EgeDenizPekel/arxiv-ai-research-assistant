"""Unit tests for SQLite database helpers."""

import sqlite3
from unittest.mock import patch

import pytest

from src.pipeline.database import (
    get_all_chunks,
    get_existing_arxiv_ids,
    init_db,
    insert_chunks,
    insert_paper,
)

_SAMPLE_PAPER = {
    "arxiv_id": "hf_test001",
    "title": "A Test Paper on Neural Networks",
    "authors": "Author A, Author B",
    "abstract": "We study neural network architectures.",
    "year": 2024,
    "categories": "cs.LG",
    "pdf_path": None,
}

_SAMPLE_CHUNKS = [
    {"text": "Introduction to neural networks.", "section": "Introduction", "token_count": 5},
    {"text": "Methods for training models.", "section": "Methods", "token_count": 5},
]


@pytest.fixture
def patched_db(tmp_path):
    """Fixture that redirects DB_PATH to a temporary file and initialises the schema."""
    db_file = tmp_path / "test.db"
    with patch("src.pipeline.database.DB_PATH", db_file):
        init_db()
        yield db_file


def test_init_creates_tables(patched_db):
    conn = sqlite3.connect(patched_db)
    tables = {
        row[0]
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
    }
    conn.close()
    assert "papers" in tables
    assert "chunks" in tables


def test_insert_paper_returns_positive_id(patched_db):
    with patch("src.pipeline.database.DB_PATH", patched_db):
        paper_id = insert_paper(_SAMPLE_PAPER)
    assert isinstance(paper_id, int)
    assert paper_id > 0


def test_get_existing_arxiv_ids_includes_inserted_paper(patched_db):
    with patch("src.pipeline.database.DB_PATH", patched_db):
        insert_paper(_SAMPLE_PAPER)
        ids = get_existing_arxiv_ids()
    assert "hf_test001" in ids


def test_insert_paper_idempotent(patched_db):
    """Inserting the same paper twice returns the same ID without raising."""
    with patch("src.pipeline.database.DB_PATH", patched_db):
        id1 = insert_paper(_SAMPLE_PAPER)
        id2 = insert_paper(_SAMPLE_PAPER)
    assert id1 == id2


def test_insert_chunks_and_retrieve(patched_db):
    with patch("src.pipeline.database.DB_PATH", patched_db):
        paper_id = insert_paper(_SAMPLE_PAPER)
        insert_chunks(paper_id, _SAMPLE_PAPER["arxiv_id"], _SAMPLE_CHUNKS)
        all_chunks = get_all_chunks()
    texts = [c["text"] for c in all_chunks]
    assert "Introduction to neural networks." in texts
    assert "Methods for training models." in texts
