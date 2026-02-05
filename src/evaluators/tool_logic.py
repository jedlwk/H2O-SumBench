"""
Agent-Ready Wrapper Functions for Summary Evaluation Metrics.

**IMPORTANT FOR H2OGPTE AGENTS:**
This module must be called as a CLI tool, NOT imported as a Python module.
H2OGPTE agents should execute commands like:
    python -m src.evaluators.tool_logic run_multiple --metrics "rouge,bertscore" --summary "..." --source "..."

The CLI interface returns JSON to stdout for agent consumption.
Direct Python imports are supported for non-agent use cases only.

---

This module provides wrapper functions for all evaluation metrics, designed
for tool/function calling by AI agents. Each function has a standardized
interface with clear input/output schemas suitable for LLM function calling.

All wrapper functions:
- Accept summary (required), source (optional), and reference_summary (optional)
- Return a standardized dict with 'metric_name', 'scores', 'interpretation', 'error'
- Include comprehensive docstrings for automatic schema generation
- Do not modify the underlying evaluation functions

Usage (Non-Agent Python Import):
    from src.evaluators.tool_logic import evaluate_rouge, evaluate_bertscore, list_available_metrics

    # Run a single metric
    result = evaluate_rouge(summary="Generated text", source="Original text")

    # List all available metrics
    metrics = list_available_metrics()

    # Run a metric by name
    result = run_metric("rouge", summary="...", source="...")

Usage (H2OGPTE Agent CLI):
    python -m src.evaluators.tool_logic list_metrics
    python -m src.evaluators.tool_logic recommend --has-source --has-reference
    python -m src.evaluators.tool_logic run_multiple --metrics "rouge,bertscore" --summary "..." --source "..."
"""

import json
import argparse
from typing import Dict, Any, List, Optional, Callable

# Import all evaluation functions from existing modules
from .era1_word_overlap import (
    compute_rouge_scores,
    compute_bleu_score,
    compute_meteor_score,
    compute_levenshtein_score,
    compute_perplexity,
    compute_chrf_score,
    compute_all_era1_metrics
)
from .era2_embeddings import (
    compute_bertscore,
    compute_moverscore,
    compute_all_era2_metrics
)
from .era3_logic_checkers import (
    compute_nli_score,
    compute_factchecker_score,
    compute_factcc_score,
    compute_alignscore,
    compute_coverage_score,
    compute_all_era3_metrics
)
from .era3_llm_judge import (
    evaluate_faithfulness as _evaluate_faithfulness,
    evaluate_coherence as _evaluate_coherence,
    evaluate_relevance as _evaluate_relevance,
    evaluate_fluency as _evaluate_fluency,
    evaluate_dag as _evaluate_dag,
    evaluate_prometheus as _evaluate_prometheus,
    evaluate_all as _evaluate_all_llm
)
from .era3_unieval import (
    compute_unieval,
    compute_all_unieval_metrics
)
from .completeness_metrics import (
    compute_semantic_coverage,
    compute_bertscore_recall_source,
    compute_bartscore,
    compute_all_completeness_metrics
)


# =============================================================================
# ERA 1: WORD OVERLAP METRICS
# =============================================================================

def evaluate_rouge(
    summary: str,
    source: Optional[str] = None,
    reference_summary: Optional[str] = None
) -> Dict[str, Any]:
    """
    Evaluate word and phrase overlap between summary and reference using ROUGE metrics.

    ROUGE (Recall-Oriented Understudy for Gisting Evaluation) measures how many
    words and phrases from the reference appear in the summary. Higher scores
    indicate better lexical overlap.

    Args:
        summary: The generated summary text to evaluate. Required.
        source: The original source document text. Either source or reference_summary required.
        reference_summary: A reference summary for comparison. Either source or reference_summary required.

    Returns:
        Dictionary containing:
        - metric_name: "ROUGE"
        - scores: Dict with rouge1 (unigram), rouge2 (bigram), rougeL (longest common subsequence)
        - interpretation: Human-readable quality assessment
        - error: Error message if evaluation failed, None otherwise

    Score ranges: 0.0 to 1.0 for each metric. Higher is better.
    - rouge1 > 0.5: Good single-word overlap
    - rouge2 > 0.3: Good phrase overlap
    - rougeL > 0.4: Good sequence similarity
    """
    result = compute_rouge_scores(summary, source, reference_summary)

    interpretation = "Unable to compute"
    if result.get('rouge1') is not None:
        r1 = result['rouge1']
        if r1 >= 0.6:
            interpretation = "Excellent word overlap"
        elif r1 >= 0.4:
            interpretation = "Good word overlap"
        elif r1 >= 0.2:
            interpretation = "Moderate word overlap"
        else:
            interpretation = "Low word overlap"

    return {
        'metric_name': 'ROUGE',
        'scores': {
            'rouge1': result.get('rouge1'),
            'rouge2': result.get('rouge2'),
            'rougeL': result.get('rougeL')
        },
        'interpretation': interpretation,
        'error': result.get('error')
    }


def evaluate_bleu(
    summary: str,
    source: Optional[str] = None,
    reference_summary: Optional[str] = None
) -> Dict[str, Any]:
    """
    Evaluate n-gram precision between summary and reference using BLEU score.

    BLEU (Bilingual Evaluation Understudy) measures what percentage of words/phrases
    in the summary appear in the reference. Originally designed for machine translation,
    it focuses on precision rather than recall.

    Args:
        summary: The generated summary text to evaluate. Required.
        source: The original source document text. Either source or reference_summary required.
        reference_summary: A reference summary for comparison. Either source or reference_summary required.

    Returns:
        Dictionary containing:
        - metric_name: "BLEU"
        - scores: Dict with bleu (normalized 0-1 score)
        - interpretation: Human-readable quality assessment
        - error: Error message if evaluation failed, None otherwise

    Score range: 0.0 to 1.0. Higher is better.
    - bleu > 0.3: Good for summaries
    - bleu > 0.5: Excellent match
    """
    result = compute_bleu_score(summary, source, reference_summary)

    interpretation = "Unable to compute"
    if result.get('bleu') is not None:
        bleu = result['bleu']
        if bleu >= 0.5:
            interpretation = "Excellent n-gram precision"
        elif bleu >= 0.3:
            interpretation = "Good n-gram precision"
        elif bleu >= 0.15:
            interpretation = "Moderate n-gram precision"
        else:
            interpretation = "Low n-gram precision"

    return {
        'metric_name': 'BLEU',
        'scores': {'bleu': result.get('bleu')},
        'interpretation': interpretation,
        'error': result.get('error')
    }


def evaluate_meteor(
    summary: str,
    source: Optional[str] = None,
    reference_summary: Optional[str] = None
) -> Dict[str, Any]:
    """
    Evaluate semantic word overlap using METEOR score with synonym and stem matching.

    METEOR (Metric for Evaluation of Translation with Explicit ORdering) is more
    flexible than BLEU/ROUGE: it recognizes synonyms ("car"/"automobile") and
    word stems ("running"/"runs"). Good for evaluating paraphrasing quality.

    Args:
        summary: The generated summary text to evaluate. Required.
        source: The original source document text. Either source or reference_summary required.
        reference_summary: A reference summary for comparison. Either source or reference_summary required.

    Returns:
        Dictionary containing:
        - metric_name: "METEOR"
        - scores: Dict with meteor score (0-1)
        - interpretation: Human-readable quality assessment
        - error: Error message if evaluation failed, None otherwise

    Score range: 0.0 to 1.0. Higher is better. Typically higher than BLEU.
    """
    result = compute_meteor_score(summary, source, reference_summary)

    interpretation = "Unable to compute"
    if result.get('meteor') is not None:
        meteor = result['meteor']
        if meteor >= 0.5:
            interpretation = "Excellent semantic alignment"
        elif meteor >= 0.3:
            interpretation = "Good semantic alignment"
        elif meteor >= 0.15:
            interpretation = "Moderate semantic alignment"
        else:
            interpretation = "Low semantic alignment"

    return {
        'metric_name': 'METEOR',
        'scores': {'meteor': result.get('meteor')},
        'interpretation': interpretation,
        'error': result.get('error')
    }


def evaluate_levenshtein(
    summary: str,
    source: Optional[str] = None,
    reference_summary: Optional[str] = None
) -> Dict[str, Any]:
    """
    Evaluate character-level edit distance similarity between texts.

    Levenshtein distance measures how many character insertions, deletions, or
    substitutions are needed to transform one text into another. The similarity
    score is normalized to 0-1 where 1.0 means identical strings.

    Args:
        summary: The generated summary text to evaluate. Required.
        source: The original source document text. Either source or reference_summary required.
        reference_summary: A reference summary for comparison. Either source or reference_summary required.

    Returns:
        Dictionary containing:
        - metric_name: "Levenshtein"
        - scores: Dict with levenshtein similarity (0-1)
        - interpretation: Human-readable quality assessment
        - error: Error message if evaluation failed, None otherwise

    Score range: 0.0 to 1.0. 1.0 means identical strings.
    """
    result = compute_levenshtein_score(summary, source, reference_summary)

    interpretation = "Unable to compute"
    if result.get('levenshtein') is not None:
        lev = result['levenshtein']
        if lev >= 0.8:
            interpretation = "Very similar strings"
        elif lev >= 0.6:
            interpretation = "Moderately similar"
        elif lev >= 0.4:
            interpretation = "Some similarity"
        else:
            interpretation = "Low string similarity"

    return {
        'metric_name': 'Levenshtein',
        'scores': {'levenshtein': result.get('levenshtein')},
        'interpretation': interpretation,
        'error': result.get('error')
    }


def evaluate_perplexity(
    summary: str,
    source: Optional[str] = None,
    reference_summary: Optional[str] = None
) -> Dict[str, Any]:
    """
    Evaluate text fluency and naturalness using GPT-2 language model perplexity.

    Perplexity measures how "surprised" a language model is by the text. Lower
    perplexity indicates more natural, fluent writing. This metric only checks
    fluency, NOT factual accuracy or meaning.

    Args:
        summary: The generated summary text to evaluate for fluency. Required.
        source: Not used for perplexity (kept for API consistency).
        reference_summary: Not used for perplexity (kept for API consistency).

    Returns:
        Dictionary containing:
        - metric_name: "Perplexity"
        - scores: Dict with perplexity (raw) and normalized_score (0-1, higher=more fluent)
        - interpretation: Human-readable quality assessment
        - error: Error message if evaluation failed, None otherwise

    Perplexity ranges: Lower is better.
    - < 30: Very fluent
    - 30-100: Normal text
    - > 100: Potentially unnatural
    """
    result = compute_perplexity(summary, source, reference_summary)

    interpretation = "Unable to compute"
    if result.get('perplexity') is not None:
        ppl = result['perplexity']
        if ppl < 30:
            interpretation = "Very fluent and natural"
        elif ppl < 60:
            interpretation = "Fluent text"
        elif ppl < 100:
            interpretation = "Moderately fluent"
        else:
            interpretation = "Potentially unnatural phrasing"

    return {
        'metric_name': 'Perplexity',
        'scores': {
            'perplexity': result.get('perplexity'),
            'normalized_score': result.get('normalized_score')
        },
        'interpretation': interpretation,
        'error': result.get('error')
    }


def evaluate_chrf(
    summary: str,
    source: Optional[str] = None,
    reference_summary: Optional[str] = None
) -> Dict[str, Any]:
    """
    Evaluate character n-gram overlap using chrF++ score.

    chrF++ measures character-level n-gram overlap with word order consideration.
    More forgiving than word-level metrics: handles typos, morphology variations,
    and compound words better. Useful for morphologically rich languages.

    Args:
        summary: The generated summary text to evaluate. Required.
        source: The original source document text. Either source or reference_summary required.
        reference_summary: A reference summary for comparison. Either source or reference_summary required.

    Returns:
        Dictionary containing:
        - metric_name: "chrF++"
        - scores: Dict with chrf (normalized 0-1) and raw_score (0-100)
        - interpretation: Human-readable quality assessment
        - error: Error message if evaluation failed, None otherwise

    Score range: 0.0 to 1.0. Higher is better.
    """
    result = compute_chrf_score(summary, source, reference_summary)

    interpretation = "Unable to compute"
    if result.get('chrf') is not None:
        chrf = result['chrf']
        if chrf >= 0.7:
            interpretation = "Excellent character-level match"
        elif chrf >= 0.5:
            interpretation = "Good character-level match"
        elif chrf >= 0.3:
            interpretation = "Moderate character-level match"
        else:
            interpretation = "Low character-level match"

    return {
        'metric_name': 'chrF++',
        'scores': {
            'chrf': result.get('chrf'),
            'raw_score': result.get('raw_score')
        },
        'interpretation': interpretation,
        'error': result.get('error')
    }


def evaluate_all_word_overlap(
    summary: str,
    source: Optional[str] = None,
    reference_summary: Optional[str] = None
) -> Dict[str, Any]:
    """
    Run all word overlap and lexical metrics (Era 1) in a single call.

    Computes ROUGE, BLEU, METEOR, chrF++, Levenshtein, and Perplexity metrics
    together. Useful for comprehensive lexical comparison when you have a
    reference summary.

    Args:
        summary: The generated summary text to evaluate. Required.
        source: The original source document text. Either source or reference_summary required.
        reference_summary: A reference summary for comparison. Either source or reference_summary required.

    Returns:
        Dictionary containing:
        - metric_name: "Era1_WordOverlap"
        - scores: Dict with all Era 1 metrics (ROUGE, BLEU, METEOR, chrF++, Levenshtein, Perplexity)
        - interpretation: Overall quality assessment
        - error: Error message if any metric failed, None otherwise
    """
    results = compute_all_era1_metrics(summary, source, reference_summary)

    # Check for errors
    errors = []
    for metric_name, metric_result in results.items():
        if metric_result.get('error'):
            errors.append(f"{metric_name}: {metric_result['error']}")

    return {
        'metric_name': 'Era1_WordOverlap',
        'scores': results,
        'interpretation': "Comprehensive lexical comparison completed",
        'error': '; '.join(errors) if errors else None
    }


# =============================================================================
# ERA 2: EMBEDDING-BASED METRICS
# =============================================================================

def evaluate_bertscore(
    summary: str,
    source: Optional[str] = None,
    reference_summary: Optional[str] = None
) -> Dict[str, Any]:
    """
    Evaluate semantic similarity using contextualized BERT embeddings.

    BERTScore measures if texts have similar meanings even with different wording.
    Uses BERT to understand context ("bank" near "river" vs "bank" near "money").
    Excellent for evaluating paraphrases and semantic equivalence.

    Args:
        summary: The generated summary text to evaluate. Required.
        source: The original source document text. Either source or reference_summary required.
        reference_summary: A reference summary for comparison. Either source or reference_summary required.

    Returns:
        Dictionary containing:
        - metric_name: "BERTScore"
        - scores: Dict with precision, recall, f1 (each 0-1)
        - interpretation: Human-readable quality assessment
        - error: Error message if evaluation failed, None otherwise

    Score ranges: 0.0 to 1.0. Higher is better.
    - f1 > 0.8: Excellent semantic similarity
    - f1 > 0.7: Good semantic similarity
    """
    result = compute_bertscore(summary, source, reference_summary)

    interpretation = "Unable to compute"
    if result.get('f1') is not None:
        f1 = result['f1']
        if f1 >= 0.85:
            interpretation = "Excellent semantic similarity"
        elif f1 >= 0.75:
            interpretation = "Good semantic similarity"
        elif f1 >= 0.65:
            interpretation = "Moderate semantic similarity"
        else:
            interpretation = "Low semantic similarity"

    return {
        'metric_name': 'BERTScore',
        'scores': {
            'precision': result.get('precision'),
            'recall': result.get('recall'),
            'f1': result.get('f1')
        },
        'interpretation': interpretation,
        'error': result.get('error')
    }


def evaluate_moverscore(
    summary: str,
    source: Optional[str] = None,
    reference_summary: Optional[str] = None
) -> Dict[str, Any]:
    """
    Evaluate semantic alignment using optimal transport distance between word embeddings.

    MoverScore uses Earth Mover's Distance to find the minimum "cost" to transform
    the summary's word meanings into the source's word meanings. More sophisticated
    than token-by-token matching.

    Args:
        summary: The generated summary text to evaluate. Required.
        source: The original source document text. Either source or reference_summary required.
        reference_summary: A reference summary for comparison. Either source or reference_summary required.

    Returns:
        Dictionary containing:
        - metric_name: "MoverScore"
        - scores: Dict with moverscore (0-1)
        - interpretation: Human-readable quality assessment
        - error: Error message if evaluation failed, None otherwise

    Score range: 0.0 to 1.0. Higher is better. Typically 0.4-0.6 for good matches.
    Note: May have issues with CPU-only PyTorch configurations.
    """
    result = compute_moverscore(summary, source, reference_summary)

    interpretation = "Unable to compute"
    if result.get('moverscore') is not None:
        ms = result['moverscore']
        if ms >= 0.6:
            interpretation = "Excellent semantic alignment"
        elif ms >= 0.5:
            interpretation = "Good semantic alignment"
        elif ms >= 0.4:
            interpretation = "Moderate semantic alignment"
        else:
            interpretation = "Low semantic alignment"

    return {
        'metric_name': 'MoverScore',
        'scores': {'moverscore': result.get('moverscore')},
        'interpretation': interpretation,
        'error': result.get('error')
    }


def evaluate_all_embeddings(
    summary: str,
    source: Optional[str] = None,
    reference_summary: Optional[str] = None
) -> Dict[str, Any]:
    """
    Run all embedding-based semantic metrics (Era 2) in a single call.

    Computes BERTScore and MoverScore together. Both use BERT-based contextual
    embeddings to understand semantic content beyond exact word matches.

    Args:
        summary: The generated summary text to evaluate. Required.
        source: The original source document text. Either source or reference_summary required.
        reference_summary: A reference summary for comparison. Either source or reference_summary required.

    Returns:
        Dictionary containing:
        - metric_name: "Era2_Embeddings"
        - scores: Dict with BERTScore and MoverScore results
        - interpretation: Overall quality assessment
        - error: Error message if any metric failed, None otherwise
    """
    results = compute_all_era2_metrics(summary, source, reference_summary)

    errors = []
    for metric_name, metric_result in results.items():
        if metric_result.get('error'):
            errors.append(f"{metric_name}: {metric_result['error']}")

    return {
        'metric_name': 'Era2_Embeddings',
        'scores': results,
        'interpretation': "Semantic similarity analysis completed",
        'error': '; '.join(errors) if errors else None
    }


# =============================================================================
# ERA 3: LOGIC CHECKERS & FACTUALITY
# =============================================================================

def evaluate_nli(
    summary: str,
    source: str,
    reference_summary: Optional[str] = None
) -> Dict[str, Any]:
    """
    Check if summary claims are logically supported by source using Natural Language Inference.

    NLI classifies the relationship between source and summary as entailment, neutral,
    or contradiction. High scores indicate the summary is well-supported by the source.
    Use this for detecting hallucinations and unsupported claims.

    Args:
        summary: The generated summary text to check for factual consistency. Required.
        source: The original source document that should support the summary. Required.
        reference_summary: Not used for NLI (kept for API consistency).

    Returns:
        Dictionary containing:
        - metric_name: "NLI"
        - scores: Dict with nli_score (0-1), label (classification)
        - interpretation: Human-readable consistency assessment
        - error: Error message if evaluation failed, None otherwise

    Score range: 0.0 to 1.0. Higher is better.
    - > 0.7: Well-supported by source
    - < 0.4: Potential contradictions/hallucinations
    """
    result = compute_nli_score(summary, source, reference_summary)

    return {
        'metric_name': 'NLI',
        'scores': {
            'nli_score': result.get('nli_score'),
            'label': result.get('label')
        },
        'interpretation': result.get('interpretation', 'Unable to compute'),
        'error': result.get('error')
    }


def evaluate_factcc(
    summary: str,
    source: str,
    reference_summary: Optional[str] = None
) -> Dict[str, Any]:
    """
    Check factual consistency using a BERT model trained specifically for summarization.

    FactCC is a specialized fact-checking model trained on summarization data.
    Detects factual errors common in generated summaries like entity swaps,
    incorrect relations, and unsupported claims.

    Args:
        summary: The generated summary text to fact-check. Required.
        source: The original source document to verify claims against. Required.
        reference_summary: Not used for FactCC (kept for API consistency).

    Returns:
        Dictionary containing:
        - metric_name: "FactCC"
        - scores: Dict with score (0-1), label (Consistent/Inconsistent)
        - interpretation: Human-readable consistency assessment
        - error: Error message if evaluation failed, None otherwise

    Score range: 0.0 to 1.0. Higher is better.
    - > 0.7: Factually consistent
    - < 0.4: Likely inconsistencies
    """
    result = compute_factcc_score(summary, source, reference_summary)

    return {
        'metric_name': 'FactCC',
        'scores': {
            'score': result.get('score'),
            'label': result.get('label'),
            'raw_label': result.get('raw_label')
        },
        'interpretation': result.get('interpretation', 'Unable to compute'),
        'error': result.get('error')
    }


def evaluate_alignscore(
    summary: str,
    source: str,
    reference_summary: Optional[str] = None
) -> Dict[str, Any]:
    """
    Evaluate factual consistency using state-of-the-art unified alignment model (RECOMMENDED).

    AlignScore is the most robust single metric for factual accuracy. Uses RoBERTa-large
    fine-tuned on 7 diverse alignment tasks. Recommended for production fact-checking.

    Args:
        summary: The generated summary (claim) to fact-check. Required.
        source: The original source document (premise) to verify against. Required.
        reference_summary: Not used for AlignScore (kept for API consistency).

    Returns:
        Dictionary containing:
        - metric_name: "AlignScore"
        - scores: Dict with score (0-1)
        - interpretation: Human-readable consistency assessment
        - error: Error message if evaluation failed, None otherwise

    Score range: 0.0 to 1.0. Higher is better.
    - > 0.9: Fully consistent
    - > 0.7: Highly consistent
    - < 0.5: Potentially inconsistent
    """
    result = compute_alignscore(summary, source, reference_summary)

    return {
        'metric_name': 'AlignScore',
        'scores': {'score': result.get('score')},
        'interpretation': result.get('interpretation', 'Unable to compute'),
        'error': result.get('error')
    }


def evaluate_entity_coverage(
    summary: str,
    source: str,
    reference_summary: Optional[str] = None
) -> Dict[str, Any]:
    """
    Calculate what percentage of named entities from source appear in summary.

    Uses spaCy NER to extract entities (people, places, organizations, dates)
    from both texts and measures coverage. Ensures summaries preserve key
    factual details.

    Args:
        summary: The generated summary to check for entity coverage. Required.
        source: The original source document containing entities. Required.
        reference_summary: Not used for coverage (kept for API consistency).

    Returns:
        Dictionary containing:
        - metric_name: "EntityCoverage"
        - scores: Dict with score (0-1), source_entities, covered_entities, missing_entities
        - interpretation: Human-readable coverage assessment
        - error: Error message if evaluation failed, None otherwise

    Score range: 0.0 to 1.0. Higher is better.
    - > 0.7: Good entity coverage
    - < 0.5: Many entities missing
    """
    result = compute_coverage_score(summary, source, reference_summary)

    return {
        'metric_name': 'EntityCoverage',
        'scores': {
            'score': result.get('score'),
            'source_entities': result.get('source_entities'),
            'covered_entities': result.get('covered_entities'),
            'missing_entities': result.get('missing_entities')
        },
        'interpretation': result.get('interpretation', 'Unable to compute'),
        'error': result.get('error')
    }


def evaluate_factchecker_api(
    summary: str,
    source: str,
    reference_summary: Optional[str] = None,
    model_name: Optional[str] = None
) -> Dict[str, Any]:
    """
    Perform detailed claim-by-claim fact-checking using LLM API.

    Uses a large language model (default: Llama-3.3-70B) to extract and verify
    each factual claim in the summary against the source. Provides detailed
    breakdown of claims checked, issues found, and explanations.

    Requires H2OGPTE_API_KEY and H2OGPTE_ADDRESS environment variables.

    Args:
        summary: The generated summary with claims to fact-check. Required.
        source: The original source document to verify claims against. Required.
        reference_summary: Not used for fact-checking (kept for API consistency).
        model_name: LLM model name. Default "meta-llama/Llama-3.3-70B-Instruct".

    Returns:
        Dictionary containing:
        - metric_name: "FactChecker_API"
        - scores: Dict with score (0-1), raw_score (1-10), claims_checked, issues_found
        - interpretation: Human-readable factuality assessment
        - error: Error message if evaluation failed, None otherwise

    Score range: 0.0 to 1.0 (normalized from 1-10). Higher is better.
    """
    result = compute_factchecker_score(summary, source, reference_summary, model_name)

    return {
        'metric_name': 'FactChecker_API',
        'scores': {
            'score': result.get('score'),
            'raw_score': result.get('raw_score'),
            'claims_checked': result.get('claims_checked'),
            'issues_found': result.get('issues_found'),
            'explanation': result.get('explanation')
        },
        'interpretation': result.get('interpretation', 'Unable to compute'),
        'error': result.get('error')
    }


def evaluate_all_factuality(
    summary: str,
    source: str,
    reference_summary: Optional[str] = None,
    use_alignscore: bool = True,
    use_factcc: bool = False,
    use_coverage: bool = True,
    use_factchecker_api: bool = False
) -> Dict[str, Any]:
    """
    Run multiple factuality and consistency metrics (Era 3) to detect hallucinations.

    Always includes NLI. Optionally includes AlignScore (recommended), FactCC,
    EntityCoverage, and API-based FactChecker. Multiple approaches detect
    different types of factual errors.

    Args:
        summary: The generated summary to fact-check. Required.
        source: The original source document to verify against. Required.
        reference_summary: Not used for Era 3 metrics (kept for API consistency).
        use_alignscore: Enable AlignScore (recommended, ~1.3GB model). Default True.
        use_factcc: Enable FactCC BERT model (~600MB). Default False.
        use_coverage: Enable entity coverage check (requires spaCy). Default True.
        use_factchecker_api: Enable API-based LLM fact-checker (slow, requires API). Default False.

    Returns:
        Dictionary containing:
        - metric_name: "Era3_Factuality"
        - scores: Dict with NLI and optionally AlignScore, FactCC, Coverage, FactChecker
        - interpretation: Overall factuality assessment
        - error: Error message if any metric failed, None otherwise
    """
    results = compute_all_era3_metrics(
        summary, source, reference_summary,
        use_factchecker=use_factchecker_api,
        use_factcc=use_factcc,
        use_alignscore=use_alignscore,
        use_coverage=use_coverage
    )

    if 'error' in results:
        return {
            'metric_name': 'Era3_Factuality',
            'scores': {},
            'interpretation': 'Unable to compute',
            'error': results['error']
        }

    errors = []
    for metric_name, metric_result in results.items():
        if isinstance(metric_result, dict) and metric_result.get('error'):
            errors.append(f"{metric_name}: {metric_result['error']}")

    return {
        'metric_name': 'Era3_Factuality',
        'scores': results,
        'interpretation': "Factuality analysis completed",
        'error': '; '.join(errors) if errors else None
    }


# =============================================================================
# ERA 3: LLM-AS-JUDGE EVALUATORS
# =============================================================================

def evaluate_llm_faithfulness(
    summary: str,
    source: str,
    reference_summary: Optional[str] = None,
    model_name: str = 'meta-llama/Llama-3.3-70B-Instruct',
    timeout: int = 60
) -> Dict[str, Any]:
    """
    Use LLM to evaluate if summary claims are supported by source (G-Eval framework).

    An LLM reads both texts and judges faithfulness on a 1-10 scale with explanation.
    Detects hallucinations, contradictions, and unsupported claims. No token limits.
    Requires H2OGPTE API.

    Args:
        summary: The generated summary text to evaluate. Required.
        source: The source document to verify claims against. Required.
        reference_summary: Not used for faithfulness (kept for API consistency).
        model_name: H2OGPTE LLM model name. Default "meta-llama/Llama-3.3-70B-Instruct".
        timeout: API timeout in seconds. Default 60.

    Returns:
        Dictionary containing:
        - metric_name: "LLM_Faithfulness"
        - scores: Dict with score (0-1), raw_score (1-10), explanation
        - interpretation: LLM's reasoning for the score
        - error: Error message if evaluation failed, None otherwise
    """
    result = _evaluate_faithfulness(summary, source, reference_summary, model_name, timeout)

    return {
        'metric_name': 'LLM_Faithfulness',
        'scores': {
            'score': result.get('score'),
            'raw_score': result.get('raw_score')
        },
        'interpretation': result.get('explanation', 'Unable to compute'),
        'error': result.get('error')
    }


def evaluate_llm_coherence(
    summary: str,
    source: Optional[str] = None,
    reference_summary: Optional[str] = None,
    model_name: str = 'meta-llama/Llama-3.3-70B-Instruct',
    timeout: int = 60
) -> Dict[str, Any]:
    """
    Use LLM to evaluate if the summary flows logically with clear structure (G-Eval framework).

    An LLM judges coherence on a 1-10 scale, checking for logical progression,
    clear topic transitions, and overall structural quality. Requires H2OGPTE API.

    Args:
        summary: The generated summary text to evaluate. Required.
        source: Not used for coherence (kept for API consistency).
        reference_summary: Not used for coherence (kept for API consistency).
        model_name: H2OGPTE LLM model name. Default "meta-llama/Llama-3.3-70B-Instruct".
        timeout: API timeout in seconds. Default 60.

    Returns:
        Dictionary containing:
        - metric_name: "LLM_Coherence"
        - scores: Dict with score (0-1), raw_score (1-10)
        - interpretation: LLM's reasoning about logical structure
        - error: Error message if evaluation failed, None otherwise
    """
    result = _evaluate_coherence(summary, source, reference_summary, model_name, timeout)

    return {
        'metric_name': 'LLM_Coherence',
        'scores': {
            'score': result.get('score'),
            'raw_score': result.get('raw_score')
        },
        'interpretation': result.get('explanation', 'Unable to compute'),
        'error': result.get('error')
    }


def evaluate_llm_relevance(
    summary: str,
    source: str,
    reference_summary: Optional[str] = None,
    model_name: str = 'meta-llama/Llama-3.3-70B-Instruct',
    timeout: int = 60
) -> Dict[str, Any]:
    """
    Use LLM to evaluate if summary captures key source information (G-Eval framework).

    An LLM judges relevance on a 1-10 scale, checking if key information is
    captured while avoiding trivial details. Requires H2OGPTE API.

    Args:
        summary: The generated summary text to evaluate. Required.
        source: The source document to check coverage against. Required.
        reference_summary: Not used for relevance (kept for API consistency).
        model_name: H2OGPTE LLM model name. Default "meta-llama/Llama-3.3-70B-Instruct".
        timeout: API timeout in seconds. Default 60.

    Returns:
        Dictionary containing:
        - metric_name: "LLM_Relevance"
        - scores: Dict with score (0-1), raw_score (1-10)
        - interpretation: LLM's reasoning about information selection
        - error: Error message if evaluation failed, None otherwise
    """
    result = _evaluate_relevance(summary, source, reference_summary, model_name, timeout)

    return {
        'metric_name': 'LLM_Relevance',
        'scores': {
            'score': result.get('score'),
            'raw_score': result.get('raw_score')
        },
        'interpretation': result.get('explanation', 'Unable to compute'),
        'error': result.get('error')
    }


def evaluate_llm_fluency(
    summary: str,
    source: Optional[str] = None,
    reference_summary: Optional[str] = None,
    model_name: str = 'meta-llama/Llama-3.3-70B-Instruct',
    timeout: int = 60
) -> Dict[str, Any]:
    """
    Use LLM to evaluate grammatical correctness and writing quality (G-Eval framework).

    An LLM judges fluency on a 1-10 scale, checking for grammar errors, awkward
    phrasing, and overall readability. Requires H2OGPTE API.

    Args:
        summary: The generated summary text to evaluate. Required.
        source: Not used for fluency (kept for API consistency).
        reference_summary: Not used for fluency (kept for API consistency).
        model_name: H2OGPTE LLM model name. Default "meta-llama/Llama-3.3-70B-Instruct".
        timeout: API timeout in seconds. Default 60.

    Returns:
        Dictionary containing:
        - metric_name: "LLM_Fluency"
        - scores: Dict with score (0-1), raw_score (1-10)
        - interpretation: LLM's reasoning about writing quality
        - error: Error message if evaluation failed, None otherwise
    """
    result = _evaluate_fluency(summary, source, reference_summary, model_name, timeout)

    return {
        'metric_name': 'LLM_Fluency',
        'scores': {
            'score': result.get('score'),
            'raw_score': result.get('raw_score')
        },
        'interpretation': result.get('explanation', 'Unable to compute'),
        'error': result.get('error')
    }


def evaluate_llm_dag(
    summary: str,
    source: str,
    reference_summary: Optional[str] = None,
    model_name: str = 'meta-llama/Llama-3.3-70B-Instruct',
    timeout: int = 60
) -> Dict[str, Any]:
    """
    Use LLM with decision tree approach for multi-dimensional evaluation (DAG framework).

    Evaluates summary using a 3-step decision tree: Factual Accuracy (0-2),
    Completeness (0-2), Clarity (0-2). Total score 0-6. Requires H2OGPTE API.

    Args:
        summary: The generated summary text to evaluate. Required.
        source: The source document to verify against. Required.
        reference_summary: Not used for DAG (kept for API consistency).
        model_name: H2OGPTE LLM model name. Default "meta-llama/Llama-3.3-70B-Instruct".
        timeout: API timeout in seconds. Default 60.

    Returns:
        Dictionary containing:
        - metric_name: "LLM_DAG"
        - scores: Dict with score (0-1), raw_score (0-6), step1_factual, step2_completeness, step3_clarity
        - interpretation: LLM's reasoning for the decision path
        - error: Error message if evaluation failed, None otherwise
    """
    result = _evaluate_dag(summary, source, reference_summary, model_name, timeout)

    return {
        'metric_name': 'LLM_DAG',
        'scores': {
            'score': result.get('score'),
            'raw_score': result.get('raw_score'),
            'step1_factual': result.get('step1_factual'),
            'step2_completeness': result.get('step2_completeness'),
            'step3_clarity': result.get('step3_clarity')
        },
        'interpretation': result.get('explanation', 'Unable to compute'),
        'error': result.get('error')
    }


def evaluate_llm_prometheus(
    summary: str,
    source: Optional[str] = None,
    reference_summary: str = None,
    model_name: str = 'meta-llama/Llama-3.3-70B-Instruct',
    timeout: int = 60
) -> Dict[str, Any]:
    """
    Use LLM to compare summary against reference using Prometheus framework.

    Evaluates summary on a 1-5 scale against a reference summary, focusing on
    information density. Requires a reference summary. Requires H2OGPTE API.

    Args:
        summary: The generated summary text to evaluate. Required.
        source: Not used for Prometheus (kept for API consistency).
        reference_summary: Reference summary for comparison. Required.
        model_name: H2OGPTE LLM model name. Default "meta-llama/Llama-3.3-70B-Instruct".
        timeout: API timeout in seconds. Default 60.

    Returns:
        Dictionary containing:
        - metric_name: "LLM_Prometheus"
        - scores: Dict with score (0-1), raw_score (1-5)
        - interpretation: LLM's reasoning using the rubric
        - error: Error message if evaluation failed, None otherwise
    """
    result = _evaluate_prometheus(summary, source, reference_summary, model_name, timeout)

    return {
        'metric_name': 'LLM_Prometheus',
        'scores': {
            'score': result.get('score'),
            'raw_score': result.get('raw_score')
        },
        'interpretation': result.get('explanation', 'Unable to compute'),
        'error': result.get('error')
    }


def evaluate_all_llm_judge(
    summary: str,
    source: Optional[str] = None,
    reference_summary: Optional[str] = None,
    model_name: str = 'meta-llama/Llama-3.3-70B-Instruct',
    timeout: int = 60,
    include_dag: bool = False,
    include_prometheus: bool = False
) -> Dict[str, Any]:
    """
    Run comprehensive LLM-based evaluation across multiple dimensions (G-Eval + DAG/Prometheus).

    Always includes Faithfulness, Coherence, Relevance, and Fluency. Optionally
    includes DAG decision tree and Prometheus reference comparison.
    Requires H2OGPTE API.

    Args:
        summary: The generated summary to evaluate. Required.
        source: Source document (required for faithfulness, relevance, DAG).
        reference_summary: Reference summary (required for Prometheus if enabled).
        model_name: H2OGPTE LLM model name. Default "meta-llama/Llama-3.3-70B-Instruct".
        timeout: API timeout per call in seconds. Default 60.
        include_dag: Enable DAG decision tree evaluation. Default False.
        include_prometheus: Enable Prometheus reference comparison. Default False.

    Returns:
        Dictionary containing:
        - metric_name: "LLM_Judge_All"
        - scores: Dict with faithfulness, coherence, relevance, fluency, optionally dag, prometheus
        - interpretation: Overall quality assessment
        - error: Error message if any metric failed, None otherwise
    """
    results = _evaluate_all_llm(
        summary, source, reference_summary, model_name, timeout,
        include_dag, include_prometheus
    )

    errors = []
    for metric_name, metric_result in results.items():
        if isinstance(metric_result, dict) and metric_result.get('error'):
            errors.append(f"{metric_name}: {metric_result['error']}")

    return {
        'metric_name': 'LLM_Judge_All',
        'scores': results,
        'interpretation': "LLM-based evaluation completed",
        'error': '; '.join(errors) if errors else None
    }


# =============================================================================
# ERA 3: UNIEVAL
# =============================================================================

def evaluate_unieval(
    summary: str,
    source: str,
    reference_summary: Optional[str] = None,
    dimensions: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Evaluate summary across multiple dimensions using UniEval's Boolean QA framework.

    UniEval converts evaluation into yes/no questions for coherence, consistency,
    and fluency. Provides multi-dimensional quality assessment from a unified model.

    Args:
        summary: The generated summary text to evaluate. Required.
        source: The original source document text. Required.
        reference_summary: Not used for UniEval (kept for API consistency).
        dimensions: List of dimensions to evaluate. Options: ["coherence", "consistency", "fluency"].
                   Default None evaluates all three.

    Returns:
        Dictionary containing:
        - metric_name: "UniEval"
        - scores: Dict with coherence, consistency, fluency scores (each 0-1)
        - interpretation: Human-readable assessment for each dimension
        - error: Error message if evaluation failed, None otherwise

    Score ranges: 0.0 to 1.0 for each dimension. Higher is better.
    """
    result = compute_unieval(summary, source, reference_summary, dimensions)

    return {
        'metric_name': 'UniEval',
        'scores': {
            'coherence': result.get('coherence'),
            'consistency': result.get('consistency'),
            'fluency': result.get('fluency')
        },
        'interpretation': result.get('interpretations', {}),
        'error': result.get('error')
    }


# =============================================================================
# COMPLETENESS METRICS
# =============================================================================

def evaluate_semantic_coverage(
    summary: str,
    source: Optional[str] = None,
    reference_summary: Optional[str] = None,
    threshold: float = 0.7
) -> Dict[str, Any]:
    """
    Calculate what percentage of source sentences are semantically represented in summary.

    Compares sentence embeddings to find semantic matches. High score means most
    source sentences have similar meaning in the summary. Use this to check if
    the summary captures all key points.

    Args:
        summary: The generated summary text to evaluate. Required.
        source: The original source document text. Either source or reference_summary required.
        reference_summary: A reference summary for comparison. Either source or reference_summary required.
        threshold: Cosine similarity threshold (0-1) to count as "covered". Default 0.7.

    Returns:
        Dictionary containing:
        - metric_name: "SemanticCoverage"
        - scores: Dict with score (0-1), source_sentences, covered_sentences, threshold
        - interpretation: Human-readable coverage assessment
        - error: Error message if evaluation failed, None otherwise

    Score range: 0.0 to 1.0. Higher is better.
    - > 0.7: Good coverage
    - < 0.5: Missing important content
    """
    result = compute_semantic_coverage(summary, source, reference_summary, threshold=threshold)

    return {
        'metric_name': 'SemanticCoverage',
        'scores': {
            'score': result.get('score'),
            'source_sentences': result.get('source_sentences'),
            'covered_sentences': result.get('covered_sentences'),
            'threshold': result.get('threshold')
        },
        'interpretation': result.get('interpretation', 'Unable to compute'),
        'error': result.get('error')
    }


def evaluate_bertscore_recall(
    summary: str,
    source: Optional[str] = None,
    reference_summary: Optional[str] = None
) -> Dict[str, Any]:
    """
    Measure what fraction of source meaning appears in summary using BERT embeddings.

    Uses contextualized BERT embeddings for token-level semantic matching.
    Recall score close to 1.0 means summary represents most of source's meaning.

    Args:
        summary: The generated summary text to evaluate. Required.
        source: The original source document text. Either source or reference_summary required.
        reference_summary: A reference summary for comparison. Either source or reference_summary required.

    Returns:
        Dictionary containing:
        - metric_name: "BERTScore_Recall"
        - scores: Dict with recall (0-1), precision (0-1), f1 (0-1)
        - interpretation: Human-readable coverage assessment
        - error: Error message if evaluation failed, None otherwise

    Score range: 0.0 to 1.0. Higher is better.
    - recall > 0.5: High coverage
    - recall < 0.3: Low coverage
    """
    result = compute_bertscore_recall_source(summary, source, reference_summary)

    return {
        'metric_name': 'BERTScore_Recall',
        'scores': {
            'recall': result.get('recall'),
            'precision': result.get('precision'),
            'f1': result.get('f1')
        },
        'interpretation': result.get('interpretation', 'Unable to compute'),
        'error': result.get('error')
    }


def evaluate_bartscore(
    summary: str,
    source: Optional[str] = None,
    reference_summary: Optional[str] = None
) -> Dict[str, Any]:
    """
    Calculate how likely BART would generate this summary from the source.

    Computes the log-probability that BART would generate the summary given
    the source. Higher scores (closer to 0) indicate better alignment.
    Requires ~1.6GB model download.

    Args:
        summary: The generated summary text to evaluate. Required.
        source: The original source document text. Either source or reference_summary required.
        reference_summary: A reference summary for comparison. Either source or reference_summary required.

    Returns:
        Dictionary containing:
        - metric_name: "BARTScore"
        - scores: Dict with score (negative log-likelihood, higher is better)
        - interpretation: Human-readable quality assessment
        - error: Error message if evaluation failed, None otherwise

    Score range: Typically -3 to -1. Higher (less negative) is better.
    - > -1.0: Excellent
    - > -2.0: Good
    - > -3.0: Moderate
    """
    result = compute_bartscore(summary, source, reference_summary)

    return {
        'metric_name': 'BARTScore',
        'scores': {'score': result.get('score')},
        'interpretation': result.get('interpretation', 'Unable to compute'),
        'error': result.get('error')
    }


def evaluate_all_completeness(
    summary: str,
    source: Optional[str] = None,
    reference_summary: Optional[str] = None,
    use_semantic_coverage: bool = True,
    use_bertscore_recall: bool = True,
    use_bartscore: bool = False
) -> Dict[str, Any]:
    """
    Run all completeness metrics to check if summary captures key source information.

    Each metric uses a different approach: Semantic Coverage (sentence embeddings),
    BERTScore Recall (token embeddings), BARTScore (generative probability).

    Args:
        summary: The generated summary text to evaluate. Required.
        source: The original source document text. Either source or reference_summary required.
        reference_summary: A reference summary for comparison. Either source or reference_summary required.
        use_semantic_coverage: Enable sentence-level coverage check (~80MB model). Default True.
        use_bertscore_recall: Enable BERTScore recall (~500MB model). Default True.
        use_bartscore: Enable BARTScore (~1.6GB model). Default False.

    Returns:
        Dictionary containing:
        - metric_name: "Completeness_All"
        - scores: Dict with SemanticCoverage, BERTScoreRecall, optionally BARTScore
        - interpretation: Overall completeness assessment
        - error: Error message if any metric failed, None otherwise
    """
    results = compute_all_completeness_metrics(
        summary, source, reference_summary,
        use_semantic_coverage, use_bertscore_recall, use_bartscore
    )

    errors = []
    for metric_name, metric_result in results.items():
        if metric_result.get('error'):
            errors.append(f"{metric_name}: {metric_result['error']}")

    return {
        'metric_name': 'Completeness_All',
        'scores': results,
        'interpretation': "Completeness analysis completed",
        'error': '; '.join(errors) if errors else None
    }


# =============================================================================
# UTILITY FUNCTIONS FOR AGENTS
# =============================================================================

# Registry of all available metrics with metadata
METRIC_REGISTRY: Dict[str, Dict[str, Any]] = {
    # Era 1: Word Overlap
    'rouge': {
        'function': evaluate_rouge,
        'category': 'word_overlap',
        'requires_source': False,
        'requires_reference': False,
        'requires_either': True,
        'description': 'Word and phrase overlap using ROUGE metrics'
    },
    'bleu': {
        'function': evaluate_bleu,
        'category': 'word_overlap',
        'requires_source': False,
        'requires_reference': False,
        'requires_either': True,
        'description': 'N-gram precision using BLEU score'
    },
    'meteor': {
        'function': evaluate_meteor,
        'category': 'word_overlap',
        'requires_source': False,
        'requires_reference': False,
        'requires_either': True,
        'description': 'Semantic word overlap with synonyms using METEOR'
    },
    'levenshtein': {
        'function': evaluate_levenshtein,
        'category': 'word_overlap',
        'requires_source': False,
        'requires_reference': False,
        'requires_either': True,
        'description': 'Character-level edit distance similarity'
    },
    'perplexity': {
        'function': evaluate_perplexity,
        'category': 'fluency',
        'requires_source': False,
        'requires_reference': False,
        'requires_either': False,
        'description': 'Text fluency using GPT-2 perplexity'
    },
    'chrf': {
        'function': evaluate_chrf,
        'category': 'word_overlap',
        'requires_source': False,
        'requires_reference': False,
        'requires_either': True,
        'description': 'Character n-gram overlap using chrF++'
    },
    'all_word_overlap': {
        'function': evaluate_all_word_overlap,
        'category': 'word_overlap',
        'requires_source': False,
        'requires_reference': False,
        'requires_either': True,
        'description': 'All Era 1 word overlap metrics combined'
    },

    # Era 2: Embeddings
    'bertscore': {
        'function': evaluate_bertscore,
        'category': 'semantic',
        'requires_source': False,
        'requires_reference': False,
        'requires_either': True,
        'description': 'Semantic similarity using BERT embeddings'
    },
    'moverscore': {
        'function': evaluate_moverscore,
        'category': 'semantic',
        'requires_source': False,
        'requires_reference': False,
        'requires_either': True,
        'description': 'Semantic alignment using optimal transport'
    },
    'all_embeddings': {
        'function': evaluate_all_embeddings,
        'category': 'semantic',
        'requires_source': False,
        'requires_reference': False,
        'requires_either': True,
        'description': 'All Era 2 embedding metrics combined'
    },

    # Era 3: Factuality
    'nli': {
        'function': evaluate_nli,
        'category': 'factuality',
        'requires_source': True,
        'requires_reference': False,
        'requires_either': False,
        'description': 'Natural Language Inference for logical consistency'
    },
    'factcc': {
        'function': evaluate_factcc,
        'category': 'factuality',
        'requires_source': True,
        'requires_reference': False,
        'requires_either': False,
        'description': 'FactCC BERT-based fact checking'
    },
    'alignscore': {
        'function': evaluate_alignscore,
        'category': 'factuality',
        'requires_source': True,
        'requires_reference': False,
        'requires_either': False,
        'description': 'State-of-the-art unified alignment model (RECOMMENDED)'
    },
    'entity_coverage': {
        'function': evaluate_entity_coverage,
        'category': 'factuality',
        'requires_source': True,
        'requires_reference': False,
        'requires_either': False,
        'description': 'Named entity coverage check'
    },
    'factchecker_api': {
        'function': evaluate_factchecker_api,
        'category': 'factuality',
        'requires_source': True,
        'requires_reference': False,
        'requires_either': False,
        'description': 'LLM API-based claim-by-claim fact checking'
    },
    'all_factuality': {
        'function': evaluate_all_factuality,
        'category': 'factuality',
        'requires_source': True,
        'requires_reference': False,
        'requires_either': False,
        'description': 'All factuality metrics combined'
    },

    # Era 3: LLM Judge
    'llm_faithfulness': {
        'function': evaluate_llm_faithfulness,
        'category': 'llm_judge',
        'requires_source': True,
        'requires_reference': False,
        'requires_either': False,
        'description': 'LLM-based faithfulness evaluation (G-Eval)'
    },
    'llm_coherence': {
        'function': evaluate_llm_coherence,
        'category': 'llm_judge',
        'requires_source': False,
        'requires_reference': False,
        'requires_either': False,
        'description': 'LLM-based coherence evaluation (G-Eval)'
    },
    'llm_relevance': {
        'function': evaluate_llm_relevance,
        'category': 'llm_judge',
        'requires_source': True,
        'requires_reference': False,
        'requires_either': False,
        'description': 'LLM-based relevance evaluation (G-Eval)'
    },
    'llm_fluency': {
        'function': evaluate_llm_fluency,
        'category': 'llm_judge',
        'requires_source': False,
        'requires_reference': False,
        'requires_either': False,
        'description': 'LLM-based fluency evaluation (G-Eval)'
    },
    'llm_dag': {
        'function': evaluate_llm_dag,
        'category': 'llm_judge',
        'requires_source': True,
        'requires_reference': False,
        'requires_either': False,
        'description': 'LLM-based multi-dimensional evaluation (DAG)'
    },
    'llm_prometheus': {
        'function': evaluate_llm_prometheus,
        'category': 'llm_judge',
        'requires_source': False,
        'requires_reference': True,
        'requires_either': False,
        'description': 'LLM-based reference comparison (Prometheus)'
    },
    'all_llm_judge': {
        'function': evaluate_all_llm_judge,
        'category': 'llm_judge',
        'requires_source': True,
        'requires_reference': False,
        'requires_either': False,
        'description': 'All LLM judge metrics combined'
    },

    # UniEval
    'unieval': {
        'function': evaluate_unieval,
        'category': 'multi_dimensional',
        'requires_source': True,
        'requires_reference': False,
        'requires_either': False,
        'description': 'Multi-dimensional evaluation (coherence, consistency, fluency)'
    },

    # Completeness
    'semantic_coverage': {
        'function': evaluate_semantic_coverage,
        'category': 'completeness',
        'requires_source': False,
        'requires_reference': False,
        'requires_either': True,
        'description': 'Sentence-level semantic coverage'
    },
    'bertscore_recall': {
        'function': evaluate_bertscore_recall,
        'category': 'completeness',
        'requires_source': False,
        'requires_reference': False,
        'requires_either': True,
        'description': 'BERTScore recall for content coverage'
    },
    'bartscore': {
        'function': evaluate_bartscore,
        'category': 'completeness',
        'requires_source': False,
        'requires_reference': False,
        'requires_either': True,
        'description': 'BART generative probability score'
    },
    'all_completeness': {
        'function': evaluate_all_completeness,
        'category': 'completeness',
        'requires_source': False,
        'requires_reference': False,
        'requires_either': True,
        'description': 'All completeness metrics combined'
    },
}


def list_available_metrics() -> List[Dict[str, Any]]:
    """
    List all available evaluation metrics with their metadata.

    Returns a list of dictionaries, each containing:
    - name: Metric identifier for use with run_metric()
    - category: Grouping (word_overlap, semantic, factuality, llm_judge, etc.)
    - requires_source: Whether source document is required
    - requires_reference: Whether reference summary is required
    - requires_either: Whether either source or reference is required
    - description: Brief description of what the metric measures

    Returns:
        List of metric metadata dictionaries.

    Example:
        >>> metrics = list_available_metrics()
        >>> for m in metrics:
        ...     print(f"{m['name']}: {m['description']}")
    """
    return [
        {
            'name': name,
            'category': info['category'],
            'requires_source': info['requires_source'],
            'requires_reference': info['requires_reference'],
            'requires_either': info['requires_either'],
            'description': info['description']
        }
        for name, info in METRIC_REGISTRY.items()
    ]


def list_metrics_by_category(category: str) -> List[Dict[str, Any]]:
    """
    List available metrics filtered by category.

    Categories:
    - word_overlap: ROUGE, BLEU, METEOR, etc.
    - semantic: BERTScore, MoverScore
    - factuality: NLI, FactCC, AlignScore, etc.
    - llm_judge: LLM-based G-Eval, DAG, Prometheus
    - multi_dimensional: UniEval
    - completeness: Semantic Coverage, BERTScore Recall, BARTScore
    - fluency: Perplexity

    Args:
        category: Category name to filter by.

    Returns:
        List of metric metadata dictionaries matching the category.
    """
    return [
        {
            'name': name,
            'category': info['category'],
            'requires_source': info['requires_source'],
            'requires_reference': info['requires_reference'],
            'requires_either': info['requires_either'],
            'description': info['description']
        }
        for name, info in METRIC_REGISTRY.items()
        if info['category'] == category
    ]


def get_metric_info(metric_name: str) -> Optional[Dict[str, Any]]:
    """
    Get detailed information about a specific metric.

    Args:
        metric_name: Name of the metric (e.g., 'rouge', 'bertscore', 'alignscore').

    Returns:
        Dictionary with metric metadata, or None if metric not found.
    """
    if metric_name not in METRIC_REGISTRY:
        return None

    info = METRIC_REGISTRY[metric_name]
    return {
        'name': metric_name,
        'category': info['category'],
        'requires_source': info['requires_source'],
        'requires_reference': info['requires_reference'],
        'requires_either': info['requires_either'],
        'description': info['description']
    }


def run_metric(
    metric_name: str,
    summary: str,
    source: Optional[str] = None,
    reference_summary: Optional[str] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Run a specific evaluation metric by name.

    This is the main entry point for agents to run evaluation metrics dynamically.
    The metric_name should match one of the keys from list_available_metrics().

    Args:
        metric_name: Name of the metric to run (e.g., 'rouge', 'bertscore', 'alignscore').
        summary: The generated summary text to evaluate. Required.
        source: The original source document text. Required for some metrics.
        reference_summary: A reference summary for comparison. Required for some metrics.
        **kwargs: Additional arguments passed to the metric function.

    Returns:
        Dictionary containing:
        - metric_name: Name of the metric run
        - scores: Metric-specific scores
        - interpretation: Human-readable assessment
        - error: Error message if evaluation failed, None otherwise

    Example:
        >>> result = run_metric('rouge', summary="Generated text", source="Original text")
        >>> result = run_metric('alignscore', summary="Text", source="Source")
        >>> result = run_metric('perplexity', summary="Some text")
    """
    if metric_name not in METRIC_REGISTRY:
        return {
            'metric_name': metric_name,
            'scores': {},
            'interpretation': 'Unknown metric',
            'error': f"Unknown metric: {metric_name}. Use list_available_metrics() to see available options."
        }

    metric_info = METRIC_REGISTRY[metric_name]
    metric_func = metric_info['function']

    # Validate required inputs
    if metric_info['requires_source'] and source is None:
        return {
            'metric_name': metric_name,
            'scores': {},
            'interpretation': 'Missing required input',
            'error': f"Metric '{metric_name}' requires source document"
        }

    if metric_info['requires_reference'] and reference_summary is None:
        return {
            'metric_name': metric_name,
            'scores': {},
            'interpretation': 'Missing required input',
            'error': f"Metric '{metric_name}' requires reference summary"
        }

    if metric_info['requires_either'] and source is None and reference_summary is None:
        return {
            'metric_name': metric_name,
            'scores': {},
            'interpretation': 'Missing required input',
            'error': f"Metric '{metric_name}' requires either source or reference_summary"
        }

    # Run the metric
    try:
        return metric_func(summary, source, reference_summary, **kwargs)
    except Exception as e:
        return {
            'metric_name': metric_name,
            'scores': {},
            'interpretation': 'Execution error',
            'error': str(e)
        }


def run_multiple_metrics(
    metric_names: List[str],
    summary: str,
    source: Optional[str] = None,
    reference_summary: Optional[str] = None,
    **kwargs
) -> Dict[str, Dict[str, Any]]:
    """
    Run multiple evaluation metrics in a single call.

    Args:
        metric_names: List of metric names to run.
        summary: The generated summary text to evaluate. Required.
        source: The original source document text. Required for some metrics.
        reference_summary: A reference summary for comparison. Required for some metrics.
        **kwargs: Additional arguments passed to all metric functions.

    Returns:
        Dictionary mapping metric names to their results.

    Example:
        >>> results = run_multiple_metrics(
        ...     ['rouge', 'bertscore', 'nli'],
        ...     summary="Generated text",
        ...     source="Original text"
        ... )
        >>> results['rouge']['scores']
        >>> results['bertscore']['scores']
    """
    results = {}
    for metric_name in metric_names:
        results[metric_name] = run_metric(
            metric_name, summary, source, reference_summary, **kwargs
        )
    return results


# =============================================================================
# CLI INTERFACE FOR AGENTS
# =============================================================================

def _cmd_list_metrics(args):
    """List all available metrics grouped by category."""
    metrics = list_available_metrics()

    by_category = {}
    for m in metrics:
        cat = m['category']
        if cat not in by_category:
            by_category[cat] = []
        by_category[cat].append({
            "name": m["name"],
            "description": m["description"],
            "requires_source": m["requires_source"],
            "requires_reference": m["requires_reference"]
        })

    print(json.dumps({"metrics_by_category": by_category}, indent=2))


def _cmd_run(args):
    """Run a specific metric."""
    result = run_metric(
        metric_name=args.metric,
        summary=args.summary,
        source=args.source,
        reference_summary=args.reference
    )
    print(json.dumps(result, indent=2))


def _cmd_run_multiple(args):
    """Run multiple metrics at once."""
    metrics = args.metrics.split(',')
    results = run_multiple_metrics(
        metric_names=metrics,
        summary=args.summary,
        source=args.source,
        reference_summary=args.reference
    )
    print(json.dumps(results, indent=2))


def _cmd_info(args):
    """Get info about a specific metric."""
    info = get_metric_info(args.metric)
    if info is None:
        print(json.dumps({"error": f"Unknown metric: {args.metric}"}))
    else:
        print(json.dumps(info, indent=2))


def main():
    """
    CLI entry point for H2OGPTE agent tool calling.

    This is the primary interface for H2OGPTE agents. Agents execute this
    script via shell commands as a module:
        python -m src.evaluators.tool_logic <command> <args>

    All commands return JSON to stdout for parsing.
    """
    parser = argparse.ArgumentParser(
        description="Summary Evaluation Tool for AI Agents"
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # list_metrics
    list_parser = subparsers.add_parser("list_metrics", help="List all available metrics")
    list_parser.set_defaults(func=_cmd_list_metrics)

    # run
    run_parser = subparsers.add_parser("run", help="Run a specific metric")
    run_parser.add_argument("--metric", required=True)
    run_parser.add_argument("--summary", required=True)
    run_parser.add_argument("--source")
    run_parser.add_argument("--reference")
    run_parser.set_defaults(func=_cmd_run)

    # run_multiple
    multi_parser = subparsers.add_parser("run_multiple", help="Run multiple metrics")
    multi_parser.add_argument("--metrics", required=True, help="Comma-separated metric names")
    multi_parser.add_argument("--summary", required=True)
    multi_parser.add_argument("--source")
    multi_parser.add_argument("--reference")
    multi_parser.set_defaults(func=_cmd_run_multiple)

    # info
    info_parser = subparsers.add_parser("info", help="Get info about a metric")
    info_parser.add_argument("--metric", required=True)
    info_parser.set_defaults(func=_cmd_info)

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return

    args.func(args)


if __name__ == "__main__":
    main()
