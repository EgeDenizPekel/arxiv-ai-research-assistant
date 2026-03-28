"""Unit tests for the semantic chunker."""

import pytest
from src.pipeline.chunker import CHUNK_SIZE, chunk_sections, count_tokens


def test_short_section_becomes_single_chunk():
    sections = [{"heading": "Introduction", "text": "This is a short section about neural networks."}]
    chunks = chunk_sections(sections)
    assert len(chunks) == 1
    assert chunks[0]["section"] == "Introduction"
    assert chunks[0]["text"] == "This is a short section about neural networks."


def test_long_section_splits_into_multiple_chunks():
    # ~600 words, well over 512 tokens
    long_text = "The attention mechanism computes query key value representations. " * 100
    sections = [{"heading": "Methods", "text": long_text}]
    chunks = chunk_sections(sections)
    assert len(chunks) > 1


def test_all_chunks_within_token_limit():
    long_text = "transformer model training gradient descent epoch batch. " * 150
    sections = [{"heading": "Body", "text": long_text}]
    chunks = chunk_sections(sections)
    for chunk in chunks:
        assert count_tokens(chunk["text"]) <= CHUNK_SIZE


def test_empty_section_skipped():
    sections = [
        {"heading": "Empty", "text": "   "},
        {"heading": "Content", "text": "Some meaningful content about deep learning."},
    ]
    chunks = chunk_sections(sections)
    assert len(chunks) == 1
    assert chunks[0]["section"] == "Content"


def test_section_heading_preserved_across_all_chunks():
    long_text = "deep learning representation. " * 200
    sections = [{"heading": "Discussion", "text": long_text}]
    chunks = chunk_sections(sections)
    assert len(chunks) > 1
    assert all(c["section"] == "Discussion" for c in chunks)


def test_multiple_sections_produce_separate_chunks():
    sections = [
        {"heading": "Abstract", "text": "We propose a new neural network."},
        {"heading": "Conclusion", "text": "We evaluated our model on benchmarks."},
    ]
    chunks = chunk_sections(sections)
    assert len(chunks) == 2
    sections_seen = {c["section"] for c in chunks}
    assert sections_seen == {"Abstract", "Conclusion"}


def test_token_count_stored_in_chunk():
    sections = [{"heading": "Intro", "text": "A short text about machine learning models."}]
    chunks = chunk_sections(sections)
    assert "token_count" in chunks[0]
    assert chunks[0]["token_count"] == count_tokens(chunks[0]["text"])
