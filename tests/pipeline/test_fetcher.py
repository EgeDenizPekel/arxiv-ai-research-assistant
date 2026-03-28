"""Unit tests for the ML/AI keyword filter and LaTeX text cleaner in fetcher.py."""

from src.pipeline.fetcher import _clean_text, _is_ml_paper


# ---------------------------------------------------------------------------
# ML/AI keyword filter
# ---------------------------------------------------------------------------

def test_neural_network_abstract_accepted():
    abstract = "We propose a novel neural network architecture for image classification tasks."
    assert _is_ml_paper(abstract, "") is True


def test_transformer_abstract_accepted():
    abstract = "We introduce a new transformer model trained on large text corpora."
    assert _is_ml_paper(abstract, "") is True


def test_astronomy_abstract_rejected():
    abstract = "We study the orbital dynamics of binary star systems in the Milky Way galaxy."
    assert _is_ml_paper(abstract, "") is False


def test_biology_abstract_rejected():
    abstract = "We analysed the protein folding patterns across 2,000 species of bacteria."
    assert _is_ml_paper(abstract, "") is False


def test_ml_keyword_in_full_text_fallback():
    # Abstract has no ML keywords but first 500 chars of full text does
    abstract = "We present a theoretical framework for dynamic systems."
    full_text = "This paper applies deep learning to analyse the dynamical system."
    assert _is_ml_paper(abstract, full_text) is True


# ---------------------------------------------------------------------------
# LaTeX artifact cleaner
# ---------------------------------------------------------------------------

def test_clean_text_strips_xmath_tokens():
    text = "The matrix @xmath0 is multiplied by @xmath12 to produce the output vector."
    cleaned = _clean_text(text)
    assert "@xmath" not in cleaned


def test_clean_text_strips_xcite_tokens():
    text = "As shown in previous work @xcite, the results are consistent with theory."
    cleaned = _clean_text(text)
    assert "@xcite" not in cleaned


def test_clean_text_collapses_multiple_spaces():
    text = "word1   word2    word3     word4"
    cleaned = _clean_text(text)
    assert "  " not in cleaned


def test_clean_text_preserves_content():
    text = "The attention mechanism computes query key value representations."
    cleaned = _clean_text(text)
    assert "attention mechanism" in cleaned
    assert "query key value" in cleaned
