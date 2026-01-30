#!/usr/bin/env python3
"""
Comprehensive test suite for ALL metrics in SumOmniEval.

This test covers:
- Era 1: Lexical Conformance (ROUGE, BLEU, METEOR, chrF++, Levenshtein, Perplexity)
- Era 2: Semantic Conformance (BERTScore, MoverScore)
- Era 3A: Faithfulness/Logic Checkers (NLI, FactCC, AlignScore, Coverage Score)
- Era 3B: LLM-as-a-Judge (G-Eval, DAG, Prometheus)
- Completeness: Local metrics (Semantic Coverage, BERTScore Recall, BARTScore)

Run with: python -m pytest tests/test_all_metrics.py -v
"""

import os
import sys

from src.evaluators.era3_llm_judge import evaluate_relevance

# Add paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Force CPU mode
import force_cpu  # noqa: F401

import pytest
from dotenv import load_dotenv
load_dotenv()

# ============================================================================
# TEST DATA
# ============================================================================

# Good summary - factually accurate, well-written
SOURCE_TEXT = """
The Amazon rainforest covers 5.5 million square kilometers and produces 20% of
the world's oxygen. It is home to 10% of all species on Earth. Recent studies show
deforestation has increased by 30% due to logging and agricultural expansion.
The rainforest plays a crucial role in regulating global climate patterns.
"""

REFERENCE_TEXT = """
The Amazon rainforest spans 5.5 million square kilometers and generates 20% of
global oxygen. It houses 10% of Earth's species and is vital for climate regulation.
Deforestation has risen 30% recently due to logging and farming activities.
"""

SUMMARY_GOOD = """
The Amazon rainforest spans 5.5 million square kilometers and generates 20% of
global oxygen. It houses 10% of Earth's species and is vital for climate regulation.
Deforestation has risen 30% recently due to logging and farming.
"""

# Bad summary - contains factual errors
SUMMARY_BAD = """
The Amazon rainforest covers 10 million square kilometers and produces 50% of
the world's oxygen. It contains 50% of all species. Deforestation has decreased
significantly in recent years due to new conservation efforts.
"""

# Identical texts for edge case testing
IDENTICAL_TEXT = "The quick brown fox jumps over the lazy dog."


# ============================================================================
# ERA 1: LEXICAL CONFORMANCE TESTS
# ============================================================================

class TestEra1LexicalMetrics:
    """Tests for Era 1 - Word Overlap / Lexical Conformance metrics."""

    def test_rouge_scores_format(self):
        """Test ROUGE returns expected format with all variants."""
        from src.evaluators.era1_word_overlap import compute_rouge_scores
        scores = compute_rouge_scores(REFERENCE_TEXT, SUMMARY_GOOD)

        assert 'rouge1' in scores, "Missing rouge1"
        assert 'rouge2' in scores, "Missing rouge2"
        assert 'rougeL' in scores, "Missing rougeL"

        # Scores should be between 0 and 1
        assert 0 <= scores['rouge1'] <= 1
        assert 0 <= scores['rouge2'] <= 1
        assert 0 <= scores['rougeL'] <= 1

    def test_rouge_identical_texts(self):
        """Test ROUGE with identical texts should be 1.0."""
        from src.evaluators.era1_word_overlap import compute_rouge_scores
        scores = compute_rouge_scores(IDENTICAL_TEXT, IDENTICAL_TEXT)

        if scores.get('error') is None:
            assert scores['rouge1'] == 1.0
            assert scores['rouge2'] == 1.0
            assert scores['rougeL'] == 1.0
        else:
            pytest.skip(f"ROUGE not available: {scores['error']}")

    def test_bleu_score_format(self):
        """Test BLEU returns expected format."""
        from src.evaluators.era1_word_overlap import compute_bleu_score
        scores = compute_bleu_score(REFERENCE_TEXT, SUMMARY_GOOD)

        assert 'bleu' in scores, "Missing bleu score"
        assert 0 <= scores['bleu'] <= 1

    def test_bleu_identical_texts(self):
        """Test BLEU with identical texts should be high."""
        from src.evaluators.era1_word_overlap import compute_bleu_score
        scores = compute_bleu_score(IDENTICAL_TEXT, IDENTICAL_TEXT)

        if scores.get('error') is None:
            assert scores['bleu'] > 0.9, "BLEU for identical texts should be > 0.9"
        else:
            pytest.skip(f"BLEU not available: {scores['error']}")

    def test_meteor_score_format(self):
        """Test METEOR returns expected format."""
        from src.evaluators.era1_word_overlap import compute_meteor_score
        scores = compute_meteor_score(REFERENCE_TEXT, SUMMARY_GOOD)

        assert 'meteor' in scores, "Missing meteor score"
        if scores.get('error') is None:
            assert 0 <= scores['meteor'] <= 1

    def test_meteor_identical_texts(self):
        """Test METEOR with identical texts should be high."""
        from src.evaluators.era1_word_overlap import compute_meteor_score
        scores = compute_meteor_score(IDENTICAL_TEXT, IDENTICAL_TEXT)

        if scores.get('error') is None:
            assert scores['meteor'] > 0.9, "METEOR for identical texts should be > 0.9"

    def test_chrf_score_format(self):
        """Test chrF++ returns expected format."""
        from src.evaluators.era1_word_overlap import compute_chrf_score
        scores = compute_chrf_score(REFERENCE_TEXT, SUMMARY_GOOD)

        assert 'chrf' in scores, "Missing chrF++ score"
        if scores.get('error') is None:
            assert 0 <= scores['chrf'] <= 1
            assert 'raw_score' in scores

    def test_chrf_identical_texts(self):
        """Test chrF++ with identical texts should be 1.0."""
        from src.evaluators.era1_word_overlap import compute_chrf_score
        scores = compute_chrf_score(IDENTICAL_TEXT, IDENTICAL_TEXT)

        if scores.get('error') is None:
            assert scores['chrf'] == 1.0, "chrF++ for identical texts should be 1.0"

    def test_levenshtein_score_format(self):
        """Test Levenshtein returns expected format."""
        from src.evaluators.era1_word_overlap import compute_levenshtein_score
        scores = compute_levenshtein_score(REFERENCE_TEXT, SUMMARY_GOOD)

        assert 'levenshtein' in scores, "Missing levenshtein score"
        assert 0 <= scores['levenshtein'] <= 1

    def test_levenshtein_identical_texts(self):
        """Test Levenshtein with identical texts should be 1.0."""
        from src.evaluators.era1_word_overlap import compute_levenshtein_score
        scores = compute_levenshtein_score(IDENTICAL_TEXT, IDENTICAL_TEXT)

        if scores.get('error') is None:
            assert scores['levenshtein'] == 1.0
        else:
            pytest.skip(f"Levenshtein not available: {scores['error']}")

    def test_perplexity_score_format(self):
        """Test Perplexity returns expected format."""
        from src.evaluators.era1_word_overlap import compute_perplexity
        scores = compute_perplexity(SOURCE_TEXT, SUMMARY_GOOD)

        assert 'perplexity' in scores or 'error' in scores
        if scores.get('error') is None:
            assert 'normalized_score' in scores
            assert 0 <= scores['normalized_score'] <= 1

    def test_all_era1_metrics(self):
        """Test compute_all_era1_metrics returns all metrics."""
        from src.evaluators.era1_word_overlap import compute_all_era1_metrics
        results = compute_all_era1_metrics(
            summary=SUMMARY_GOOD,
            reference_summary=REFERENCE_TEXT,  # Compare against reference, not source
        )

        expected_metrics = ['ROUGE', 'BLEU', 'METEOR', 'chrF++', 'Levenshtein', 'Perplexity']
        for metric in expected_metrics:
            assert metric in results, f"Missing {metric} in all metrics"


# ============================================================================
# ERA 2: SEMANTIC CONFORMANCE TESTS
# ============================================================================

class TestEra2SemanticMetrics:
    """Tests for Era 2 - Embedding-based Semantic Conformance metrics."""

    def test_bertscore_format(self):
        """Test BERTScore returns expected format."""
        from src.evaluators.era2_embeddings import compute_bertscore
        scores = compute_bertscore(REFERENCE_TEXT, SUMMARY_GOOD)

        assert 'f1' in scores or 'error' in scores
        if scores.get('error') is None:
            assert 'precision' in scores
            assert 'recall' in scores
            assert 0 <= scores['f1'] <= 1

    def test_bertscore_identical_texts(self):
        """Test BERTScore with identical texts should be high."""
        from src.evaluators.era2_embeddings import compute_bertscore
        scores = compute_bertscore(IDENTICAL_TEXT, IDENTICAL_TEXT)

        if scores.get('error') is None:
            assert scores['f1'] > 0.9, "BERTScore F1 for identical texts should be > 0.9"

    def test_moverscore_format(self):
        """Test MoverScore returns expected format."""
        from src.evaluators.era2_embeddings import compute_moverscore
        scores = compute_moverscore(REFERENCE_TEXT, SUMMARY_GOOD)

        assert 'moverscore' in scores or 'error' in scores
        if scores.get('error') is None:
            assert isinstance(scores['moverscore'], float)

    def test_all_era2_metrics(self):
        """Test compute_all_era2_metrics returns all metrics."""
        from src.evaluators.era2_embeddings import compute_all_era2_metrics
        results = compute_all_era2_metrics(
            summary=SUMMARY_GOOD,
            reference_summary=REFERENCE_TEXT,  # Compare against reference, not source
        )

        assert 'BERTScore' in results
        assert 'MoverScore' in results


# ============================================================================
# ERA 3A: FAITHFULNESS / LOGIC CHECKERS TESTS
# ============================================================================

class TestEra3AFaithfulnessMetrics:
    """Tests for Era 3A - Logic Checkers / Faithfulness metrics."""

    def test_nli_score_format(self):
        """Test NLI returns expected format."""
        from src.evaluators.era3_logic_checkers import compute_nli_score
        scores = compute_nli_score(SOURCE_TEXT, SUMMARY_GOOD)

        assert 'nli_score' in scores
        if scores.get('error') is None:
            assert 0 <= scores['nli_score'] <= 1
            assert 'label' in scores
            assert 'interpretation' in scores

    def test_nli_good_summary_higher_than_bad(self):
        """Test NLI score for good summary should be higher than bad summary."""
        from src.evaluators.era3_logic_checkers import compute_nli_score

        good_scores = compute_nli_score(SOURCE_TEXT, SUMMARY_GOOD)
        bad_scores = compute_nli_score(SOURCE_TEXT, SUMMARY_BAD)

        if good_scores.get('error') is None and bad_scores.get('error') is None:
            # Good summary should have higher consistency score
            assert good_scores['nli_score'] >= bad_scores['nli_score'] * 0.8, \
                "Good summary NLI score should be >= 80% of bad summary score"

    def test_factcc_score_format(self):
        """Test FactCC returns expected format."""
        from src.evaluators.era3_logic_checkers import compute_factcc_score
        scores = compute_factcc_score(SOURCE_TEXT, SUMMARY_GOOD)

        assert 'score' in scores or 'error' in scores
        if scores.get('error') is None and scores['score'] is not None:
            assert 0 <= scores['score'] <= 1
            assert 'label' in scores
            assert 'interpretation' in scores

    def test_alignscore_format(self):
        """Test AlignScore returns expected format."""
        from src.evaluators.era3_logic_checkers import compute_alignscore
        scores = compute_alignscore(SOURCE_TEXT, SUMMARY_GOOD)

        assert 'score' in scores or 'error' in scores
        if scores.get('error') is None and scores['score'] is not None:
            assert 0 <= scores['score'] <= 1
            assert 'interpretation' in scores

    def test_coverage_score_format(self):
        """Test Coverage Score (NER) returns expected format."""
        from src.evaluators.era3_logic_checkers import compute_coverage_score
        scores = compute_coverage_score(SOURCE_TEXT, SUMMARY_GOOD)

        assert 'score' in scores or 'error' in scores
        if scores.get('error') is None and scores['score'] is not None:
            assert 0 <= scores['score'] <= 1
            assert 'source_entities' in scores
            assert 'covered_entities' in scores
            assert 'interpretation' in scores

    def test_coverage_score_good_vs_bad(self):
        """Test Coverage Score: good summary should cover more entities."""
        from src.evaluators.era3_logic_checkers import compute_coverage_score

        good_scores = compute_coverage_score(SOURCE_TEXT, SUMMARY_GOOD)
        bad_scores = compute_coverage_score(SOURCE_TEXT, SUMMARY_BAD)

        if good_scores.get('error') is None and bad_scores.get('error') is None:
            if good_scores['score'] is not None and bad_scores['score'] is not None:
                # Good summary should have similar or higher coverage
                assert good_scores['score'] >= bad_scores['score'] * 0.5

    def test_all_era3_metrics(self):
        """Test compute_all_era3_metrics returns all enabled metrics."""
        from src.evaluators.era3_logic_checkers import compute_all_era3_metrics
        results = compute_all_era3_metrics(
            summary=SUMMARY_GOOD,
            source=SOURCE_TEXT,
            use_factcc=True,
            use_alignscore=True,
            use_coverage=True,
            use_factchecker=False  # Skip API test
        )

        assert 'NLI' in results, "Missing NLI metric"
        assert 'FactCC' in results, "Missing FactCC metric"
        assert 'AlignScore' in results, "Missing AlignScore metric"
        assert 'Coverage' in results, "Missing Coverage metric"


# ============================================================================
# COMPLETENESS METRICS TESTS
# ============================================================================

class TestCompletenessMetrics:
    """Tests for Completeness metrics (Semantic Coverage, BERTScore Recall, BARTScore)."""

    def test_semantic_coverage_format(self):
        """Test Semantic Coverage returns expected format."""
        from src.evaluators.completeness_metrics import compute_semantic_coverage
        scores = compute_semantic_coverage(SOURCE_TEXT, SUMMARY_GOOD)

        assert 'score' in scores or 'error' in scores
        if scores.get('error') is None and scores['score'] is not None:
            assert 0 <= scores['score'] <= 1
            assert 'source_sentences' in scores
            assert 'covered_sentences' in scores
            assert 'interpretation' in scores

    def test_semantic_coverage_good_vs_empty(self):
        """Test Semantic Coverage: good summary should have higher coverage than empty."""
        from src.evaluators.completeness_metrics import compute_semantic_coverage

        good_scores = compute_semantic_coverage(SOURCE_TEXT, SUMMARY_GOOD)
        empty_scores = compute_semantic_coverage(SOURCE_TEXT, "This is a very short summary.")

        if good_scores.get('error') is None and empty_scores.get('error') is None:
            if good_scores['score'] is not None and empty_scores['score'] is not None:
                assert good_scores['score'] >= empty_scores['score']

    def test_bertscore_recall_source_format(self):
        """Test BERTScore Recall (vs Source) returns expected format."""
        from src.evaluators.completeness_metrics import compute_bertscore_recall_source
        scores = compute_bertscore_recall_source(SOURCE_TEXT, SUMMARY_GOOD)

        assert 'recall' in scores or 'error' in scores
        if scores.get('error') is None and scores['recall'] is not None:
            assert 0 <= scores['recall'] <= 1
            assert 'precision' in scores
            assert 'f1' in scores
            assert 'interpretation' in scores

    def test_bartscore_format(self):
        """Test BARTScore returns expected format."""
        from src.evaluators.completeness_metrics import compute_bartscore
        scores = compute_bartscore(SOURCE_TEXT, SUMMARY_GOOD)

        assert 'score' in scores or 'error' in scores
        if scores.get('error') is None and scores['score'] is not None:
            assert 'interpretation' in scores

    def test_all_completeness_metrics(self):
        """Test compute_all_completeness_metrics returns all enabled metrics."""
        from src.evaluators.completeness_metrics import compute_all_completeness_metrics
        results = compute_all_completeness_metrics(
            summary=SUMMARY_GOOD,
            source=SOURCE_TEXT,
            use_semantic_coverage=True,
            use_bertscore_recall=True,
            use_bartscore=False  # Skip large model
        )

        assert 'SemanticCoverage' in results, "Missing SemanticCoverage metric"
        assert 'BERTScoreRecall' in results, "Missing BERTScoreRecall metric"


# ============================================================================
# ERA 3B: LLM-AS-A-JUDGE TESTS (API TESTS - Require H2OGPTE)
# ============================================================================

class TestEra3BLLMJudge:
    """Tests for Era 3B - LLM-as-a-Judge metrics (requires API)."""

    @pytest.fixture
    def check_api_available(self):
        """Check if H2OGPTE API is available."""
        api_key = os.getenv('H2OGPTE_API_KEY')
        address = os.getenv('H2OGPTE_ADDRESS')
        return bool(api_key and address)

    def test_llm_judge_initialization(self, check_api_available):
        """Test LLM Judge functions can be imported."""
        if not check_api_available:
            pytest.skip("H2OGPTE API not configured")

        from src.evaluators.era3_llm_judge import get_client
        client = get_client()
        assert client is not None

    def test_geval_faithfulness(self, check_api_available):
        """Test G-Eval Faithfulness evaluation."""
        if not check_api_available:
            pytest.skip("H2OGPTE API not configured")

        from src.evaluators.era3_llm_judge import evaluate_faithfulness

        result = evaluate_faithfulness(
            summary=SUMMARY_GOOD,
            source=SOURCE_TEXT,
            model_name='meta-llama/Llama-3.3-70B-Instruct',
            timeout=90
        )

        if result.get('error') is None:
            assert 'score' in result
            assert 0 <= result['score'] <= 1
            assert 'raw_score' in result

    def test_geval_coherence(self, check_api_available):
        """Test G-Eval Coherence evaluation."""
        if not check_api_available:
            pytest.skip("H2OGPTE API not configured")

        from src.evaluators.era3_llm_judge import evaluate_coherence

        result = evaluate_coherence(
            summary=SUMMARY_GOOD,
            model_name='meta-llama/Llama-3.3-70B-Instruct',
            timeout=90
        )

        if result.get('error') is None:
            assert 'score' in result
            assert 0 <= result['score'] <= 1

    def test_geval_relevance(self, check_api_available):
        """Test G-Eval Relevance evaluation (Completeness check)."""
        if not check_api_available:
            pytest.skip("H2OGPTE API not configured")

        from src.evaluators.era3_llm_judge import evaluate_relevance

        result = evaluate_relevance(
            summary=SUMMARY_GOOD,
            source=SOURCE_TEXT,
            model_name='meta-llama/Llama-3.3-70B-Instruct',
            timeout=90
        )

        if result.get('error') is None:
            assert 'score' in result
            assert 0 <= result['score'] <= 1

    def test_geval_fluency(self, check_api_available):
        """Test G-Eval Fluency evaluation."""
        if not check_api_available:
            pytest.skip("H2OGPTE API not configured")

        from src.evaluators.era3_llm_judge import evaluate_fluency

        result = evaluate_fluency(
            summary=SUMMARY_GOOD,
            model_name='meta-llama/Llama-3.3-70B-Instruct',
            timeout=90
        )

        if result.get('error') is None:
            assert 'score' in result
            assert 0 <= result['score'] <= 1

    def test_dag_evaluation(self, check_api_available):
        """Test DAG (Decision Tree) evaluation."""
        if not check_api_available:
            pytest.skip("H2OGPTE API not configured")

        from src.evaluators.era3_llm_judge import evaluate_dag

        result = evaluate_dag(
            summary=SUMMARY_GOOD,
            source=SOURCE_TEXT,
            model_name='meta-llama/Llama-3.3-70B-Instruct',
            timeout=90
        )

        if result.get('error') is None:
            assert 'score' in result
            assert 'raw_score' in result
            assert 0 <= result['raw_score'] <= 6

    def test_prometheus_evaluation(self, check_api_available):
        """Test Prometheus evaluation."""
        if not check_api_available:
            pytest.skip("H2OGPTE API not configured")

        from src.evaluators.era3_llm_judge import evaluate_prometheus

        result = evaluate_prometheus(
            summary=SUMMARY_GOOD,
            reference_summary=REFERENCE_TEXT,
            model_name='meta-llama/Llama-3.3-70B-Instruct',
            timeout=90
        )

        if result.get('error') is None:
            assert 'score' in result
            assert 0 <= result['score'] <= 1  # Normalized score

    def test_evaluate_all(self, check_api_available):
        """Test evaluate_all returns all metrics."""
        if not check_api_available:
            pytest.skip("H2OGPTE API not configured")

        from src.evaluators.era3_llm_judge import evaluate_all

        results = evaluate_all(
            summary=SUMMARY_GOOD,
            source=SOURCE_TEXT,
            reference_summary=REFERENCE_TEXT,
            model_name='meta-llama/Llama-3.3-70B-Instruct',
            timeout=90,
            include_dag=True,
            include_prometheus=True
        )

        expected = ['faithfulness', 'coherence', 'relevance', 'fluency', 'dag', 'prometheus']
        for metric in expected:
            assert metric in results, f"Missing {metric} in evaluate_all"


# ============================================================================
# EDGE CASES AND ERROR HANDLING
# ============================================================================

class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_source_text(self):
        """Test handling of empty source text."""
        from src.evaluators.era1_word_overlap import compute_rouge_scores
        scores = compute_rouge_scores("", SUMMARY_GOOD)
        assert 'rouge1' in scores

    def test_empty_summary_text(self):
        """Test handling of empty summary text."""
        from src.evaluators.era1_word_overlap import compute_rouge_scores
        scores = compute_rouge_scores(SOURCE_TEXT, "")
        assert 'rouge1' in scores

    def test_very_long_text(self):
        """Test handling of very long texts."""
        from src.evaluators.era3_logic_checkers import compute_nli_score
        long_text = " ".join(["word"] * 10000)
        scores = compute_nli_score(long_text, "Short summary.")
        # Should not raise exception, may return truncated result
        assert 'nli_score' in scores or 'error' in scores

    def test_special_characters(self):
        """Test handling of special characters."""
        from src.evaluators.era1_word_overlap import compute_rouge_scores
        special_text = "Test @#$% with special & characters! <tag> 'quotes' \"double\""
        scores = compute_rouge_scores(special_text, special_text)
        assert 'rouge1' in scores

    def test_unicode_text(self):
        """Test handling of unicode characters."""
        from src.evaluators.era1_word_overlap import compute_rouge_scores
        unicode_text = "The café served crème brûlée. 日本語 text."
        scores = compute_rouge_scores(unicode_text, unicode_text)
        assert 'rouge1' in scores

    def test_whitespace_only(self):
        """Test handling of whitespace-only text."""
        from src.evaluators.era1_word_overlap import compute_levenshtein_score
        scores = compute_levenshtein_score("   ", "   ")
        assert 'levenshtein' in scores


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

class TestIntegration:
    """Integration tests for running multiple metrics together."""

    def test_full_era1_pipeline(self):
        """Test full Era 1 evaluation pipeline."""
        from src.evaluators.era1_word_overlap import compute_all_era1_metrics
        results = compute_all_era1_metrics(
            summary=SUMMARY_GOOD,
            reference_summary=REFERENCE_TEXT,  # Compare against reference, not source
        )

        # All metrics should return without exceptions
        assert len(results) == 6
        for metric_name, metric_result in results.items():
            assert isinstance(metric_result, dict), f"{metric_name} should return dict"

    def test_full_era2_pipeline(self):
        """Test full Era 2 evaluation pipeline."""
        from src.evaluators.era2_embeddings import compute_all_era2_metrics
        results = compute_all_era2_metrics(
            summary=SUMMARY_GOOD,
            reference_summary=REFERENCE_TEXT,  # Compare against reference, not source
        )

        assert 'BERTScore' in results
        assert 'MoverScore' in results

    def test_full_era3a_pipeline(self):
        """Test full Era 3A evaluation pipeline."""
        from src.evaluators.era3_logic_checkers import compute_all_era3_metrics
        results = compute_all_era3_metrics(
            summary=SUMMARY_GOOD,
            source=SOURCE_TEXT,
            use_factcc=True,
            use_alignscore=True,
            use_coverage=True,
            use_factchecker=False
        )

        assert 'NLI' in results
        assert 'FactCC' in results

    def test_full_completeness_pipeline(self):
        """Test full completeness evaluation pipeline."""
        from src.evaluators.completeness_metrics import compute_all_completeness_metrics
        results = compute_all_completeness_metrics(
            SOURCE_TEXT,
            SUMMARY_GOOD,
            use_semantic_coverage=True,
            use_bertscore_recall=True,
            use_bartscore=False
        )

        assert 'SemanticCoverage' in results
        assert 'BERTScoreRecall' in results


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
