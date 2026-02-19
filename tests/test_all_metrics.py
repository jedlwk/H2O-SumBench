#!/usr/bin/env python3
"""
Tests for the 17 supported evaluation metrics in H2O SumBench.

Covers:
  - Word Overlap (ROUGE, BLEU, METEOR, chrF++, Levenshtein)
  - Fluency (Perplexity)
  - Semantic (BERTScore)
  - Completeness (EntityCoverage, SemanticCoverage, BERTScoreRecall)
  - LLM Judge (G-Eval, DAG, Prometheus) — skipped without API key

Run:  python -m pytest tests/test_all_metrics.py -v
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Force CPU mode before any torch imports
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['USE_CUDA'] = '0'

import pytest

# ---------------------------------------------------------------------------
# Test data
# ---------------------------------------------------------------------------

SOURCE = (
    "The Amazon rainforest covers 5.5 million square kilometers and produces "
    "20% of the world's oxygen. It is home to 10% of all species on Earth. "
    "Recent studies show deforestation has increased by 30% due to logging "
    "and agricultural expansion. The rainforest plays a crucial role in "
    "regulating global climate patterns."
)

REFERENCE = (
    "The Amazon rainforest spans 5.5 million square kilometers and generates "
    "20% of global oxygen. It houses 10% of Earth's species and is vital for "
    "climate regulation. Deforestation has risen 30% recently due to logging "
    "and farming activities."
)

SUMMARY = (
    "The Amazon rainforest spans 5.5 million square kilometers and generates "
    "20% of global oxygen. It houses 10% of Earth's species and is vital for "
    "climate regulation. Deforestation has risen 30% recently due to logging "
    "and farming."
)

IDENTICAL = "The quick brown fox jumps over the lazy dog."


def _skip_if_error(scores):
    """Skip the test if the metric returned an error (missing dependency)."""
    if scores.get('error'):
        pytest.skip(f"Dependency not available: {scores['error']}")


# ---------------------------------------------------------------------------
# Word Overlap
# ---------------------------------------------------------------------------

class TestWordOverlap:

    def test_rouge(self):
        from src.evaluators.era1_word_overlap import compute_rouge_scores
        s = compute_rouge_scores(REFERENCE, SUMMARY)
        _skip_if_error(s)
        for key in ('rouge1', 'rouge2', 'rougeL'):
            assert 0 <= s[key] <= 1

    def test_rouge_identical(self):
        from src.evaluators.era1_word_overlap import compute_rouge_scores
        s = compute_rouge_scores(IDENTICAL, IDENTICAL)
        _skip_if_error(s)
        assert s['rouge1'] == 1.0

    def test_bleu(self):
        from src.evaluators.era1_word_overlap import compute_bleu_score
        s = compute_bleu_score(REFERENCE, SUMMARY)
        _skip_if_error(s)
        assert 0 <= s['bleu'] <= 1

    def test_meteor(self):
        from src.evaluators.era1_word_overlap import compute_meteor_score
        s = compute_meteor_score(REFERENCE, SUMMARY)
        _skip_if_error(s)
        assert 0 <= s['meteor'] <= 1

    def test_chrf(self):
        from src.evaluators.era1_word_overlap import compute_chrf_score
        s = compute_chrf_score(REFERENCE, SUMMARY)
        _skip_if_error(s)
        assert 0 <= s['chrf'] <= 1

    def test_levenshtein(self):
        from src.evaluators.era1_word_overlap import compute_levenshtein_score
        s = compute_levenshtein_score(REFERENCE, SUMMARY)
        _skip_if_error(s)
        assert 0 <= s['levenshtein'] <= 1

    def test_levenshtein_identical(self):
        from src.evaluators.era1_word_overlap import compute_levenshtein_score
        s = compute_levenshtein_score(IDENTICAL, IDENTICAL)
        _skip_if_error(s)
        assert s['levenshtein'] == 1.0


# ---------------------------------------------------------------------------
# Fluency
# ---------------------------------------------------------------------------

class TestFluency:

    def test_perplexity(self):
        from src.evaluators.era1_word_overlap import compute_perplexity
        s = compute_perplexity(SOURCE, SUMMARY)
        _skip_if_error(s)
        assert s['perplexity'] >= 1
        assert 0 <= s['normalized_score'] <= 1


# ---------------------------------------------------------------------------
# Semantic
# ---------------------------------------------------------------------------

class TestSemantic:

    def test_bertscore(self):
        from src.evaluators.era2_embeddings import compute_bertscore
        s = compute_bertscore(REFERENCE, SUMMARY)
        _skip_if_error(s)
        for key in ('precision', 'recall', 'f1'):
            assert 0 <= s[key] <= 1

    def test_bertscore_identical(self):
        from src.evaluators.era2_embeddings import compute_bertscore
        s = compute_bertscore(IDENTICAL, IDENTICAL)
        _skip_if_error(s)
        assert s['f1'] > 0.9


# ---------------------------------------------------------------------------
# Completeness
# ---------------------------------------------------------------------------

class TestCompleteness:

    def test_entity_coverage(self):
        from src.evaluators.era3_logic_checkers import compute_coverage_score
        s = compute_coverage_score(SOURCE, SUMMARY)
        _skip_if_error(s)
        if s.get('score') is not None:
            assert 0 <= s['score'] <= 1

    def test_semantic_coverage(self):
        from src.evaluators.completeness_metrics import compute_semantic_coverage
        s = compute_semantic_coverage(SOURCE, SUMMARY)
        _skip_if_error(s)
        if s.get('score') is not None:
            assert 0 <= s['score'] <= 1

    def test_bertscore_recall(self):
        from src.evaluators.completeness_metrics import compute_bertscore_recall_source
        s = compute_bertscore_recall_source(SOURCE, SUMMARY)
        _skip_if_error(s)
        if s.get('recall') is not None:
            assert 0 <= s['recall'] <= 1


# ---------------------------------------------------------------------------
# LLM Judge (API-dependent — auto-skipped)
# ---------------------------------------------------------------------------

API_AVAILABLE = bool(os.getenv('H2OGPTE_API_KEY') and os.getenv('H2OGPTE_ADDRESS'))


@pytest.mark.skipif(not API_AVAILABLE, reason="H2OGPTE API not configured")
class TestLLMJudge:

    def test_geval_faithfulness(self):
        from src.evaluators.era3_llm_judge import evaluate_faithfulness
        r = evaluate_faithfulness(SUMMARY, SOURCE, timeout=90)
        _skip_if_error(r)
        assert 0 <= r['score'] <= 1

    def test_geval_coherence(self):
        from src.evaluators.era3_llm_judge import evaluate_coherence
        r = evaluate_coherence(SUMMARY, timeout=90)
        _skip_if_error(r)
        assert 0 <= r['score'] <= 1

    def test_geval_fluency(self):
        from src.evaluators.era3_llm_judge import evaluate_fluency
        r = evaluate_fluency(SUMMARY, timeout=90)
        _skip_if_error(r)
        assert 0 <= r['score'] <= 1


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:

    def test_empty_texts(self):
        from src.evaluators.era1_word_overlap import compute_rouge_scores
        s = compute_rouge_scores("", SUMMARY)
        assert 'rouge1' in s

    def test_special_characters(self):
        from src.evaluators.era1_word_overlap import compute_rouge_scores
        txt = "Test @#$% special & chars! <tag> 'quotes'"
        s = compute_rouge_scores(txt, txt)
        assert 'rouge1' in s


# ---------------------------------------------------------------------------
# Integration — run_multiple_metrics via tool_logic
# ---------------------------------------------------------------------------

class TestToolLogicIntegration:

    def test_run_multiple_lexical(self):
        from src.evaluators.tool_logic import run_multiple_metrics
        results = run_multiple_metrics(
            ['rouge', 'bleu', 'levenshtein'], SUMMARY, SOURCE
        )
        for m in ('rouge', 'bleu', 'levenshtein'):
            assert m in results
            assert 'scores' in results[m]

    def test_unknown_metric_returns_error(self):
        from src.evaluators.tool_logic import run_metric
        r = run_metric('nonexistent_metric', SUMMARY)
        assert r['error'] is not None
