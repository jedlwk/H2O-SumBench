"""
Part 1 - Completeness Metrics (Substance Check).

These metrics ensure key information and main points are captured.
Comparison: Generated Summary ↔ Source Text

Metrics:
- Semantic Coverage: Sentence-level embedding similarity
- BERTScore Recall (vs Source): What fraction of source is in summary
- BARTScore: Probability-based coverage P(Summary|Source)
"""

import os
from typing import Dict, List, Optional
import warnings

# Force CPU mode
os.environ['CUDA_VISIBLE_DEVICES'] = ''
warnings.filterwarnings('ignore')


def compute_semantic_coverage(
    summary: str,
    source: Optional[str] = None,
    reference_summary: Optional[str] = None,
    model_name: str = "all-MiniLM-L6-v2",
    threshold: float = 0.7
) -> Dict:
    """
    Calculate what percentage of source sentences are semantically represented in the summary.

    This metric answers: "How many of the source's key points made it into the summary?"
    Compares sentence embeddings to find semantic matches. High score (>0.7) means most
    source sentences have a similar meaning in the summary. Low score means the summary
    misses important information.

    Use this when: You want to check if the summary is complete and captures all key points.

    Args:
        summary (str): Generated summary text to evaluate
        source (str, optional): Original source document text to summarize
        reference_summary (str, optional): Reference summary that represents a score of 5 (ideal quality)
        model_name (str, optional): Sentence embedding model name. Default "all-MiniLM-L6-v2" (~80MB)
        threshold (float, optional): Cosine similarity threshold (0-1) to count a sentence as "covered".
                                     Default 0.7 means 70% semantic similarity required.

    Returns:
        Dict: Result dictionary with keys:
            - score (float): Coverage ratio from 0.0 to 1.0 (e.g., 0.75 = 75% of source sentences covered)
            - source_sentences (int): Total number of sentences in source
            - covered_sentences (int): Number of source sentences found in summary (above threshold)
            - threshold (float): The similarity threshold used
            - interpretation (str): Human-readable label like "Good Coverage" or "Low Coverage"
            - error (str, optional): Error message if computation failed

    Example:
        >>> result = compute_semantic_coverage(
        ...     summary="A cat was sitting.",
        ...     source="The cat sat. The dog ran."
        ... )
        >>> result['score']  # e.g., 0.5 (1 out of 2 sentences covered)
        >>> result['covered_sentences']  # 1
    """
    try:
        from sentence_transformers import SentenceTransformer
        import numpy as np

        # Validate required parameters
        if source is None and reference_summary is None:
            return {
                'score': None,
                'error': 'Either source or reference_summary must be provided'
            }

        # Use source if provided, otherwise use reference_summary
        comparison_text = source if source is not None else reference_summary

        # Split into sentences
        def split_sentences(text):
            """Simple sentence splitting."""
            import re
            sentences = re.split(r'(?<=[.!?])\s+', text.strip())
            return [s.strip() for s in sentences if s.strip()]

        source_sentences = split_sentences(comparison_text)
        summary_sentences = split_sentences(summary)

        if not source_sentences:
            return {
                'score': 1.0,
                'source_sentences': 0,
                'covered_sentences': 0,
                'interpretation': 'No sentences in source',
                'error': None
            }

        if not summary_sentences:
            return {
                'score': 0.0,
                'source_sentences': len(source_sentences),
                'covered_sentences': 0,
                'interpretation': 'No sentences in summary',
                'error': None
            }

        # Load model
        model = SentenceTransformer(model_name)

        # Encode sentences
        source_embeddings = model.encode(source_sentences)
        summary_embeddings = model.encode(summary_sentences)

        # Compute coverage: for each source sentence, find max similarity to any summary sentence
        covered_count = 0
        coverage_details = []

        for i, src_emb in enumerate(source_embeddings):
            # Cosine similarity with all summary sentences
            similarities = np.dot(summary_embeddings, src_emb) / (
                np.linalg.norm(summary_embeddings, axis=1) * np.linalg.norm(src_emb)
            )
            max_sim = float(np.max(similarities))
            is_covered = max_sim >= threshold

            if is_covered:
                covered_count += 1

            coverage_details.append({
                'sentence': source_sentences[i][:50] + '...' if len(source_sentences[i]) > 50 else source_sentences[i],
                'max_similarity': round(max_sim, 3),
                'covered': is_covered
            })

        coverage_score = covered_count / len(source_sentences)

        return {
            'score': round(coverage_score, 4),
            'source_sentences': len(source_sentences),
            'covered_sentences': covered_count,
            'threshold': threshold,
            'interpretation': _interpret_semantic_coverage(coverage_score),
            'error': None
        }

    except ImportError as e:
        return {
            'score': None,
            'error': f'sentence-transformers not installed. Run: pip install sentence-transformers. Error: {str(e)}'
        }
    except Exception as e:
        return {
            'score': None,
            'error': str(e)
        }


def _interpret_semantic_coverage(score: float) -> str:
    """Interpret Semantic Coverage score."""
    if score >= 0.9:
        return "Excellent Coverage"
    elif score >= 0.7:
        return "Good Coverage"
    elif score >= 0.5:
        return "Partial Coverage"
    elif score >= 0.3:
        return "Low Coverage"
    else:
        return "Poor Coverage"


def compute_bertscore_recall_source(
    summary: str,
    source: Optional[str] = None,
    reference_summary: Optional[str] = None,
) -> Dict:
    """
    Measure what fraction of the source document's meaning appears in the summary using BERT embeddings.

    This metric answers: "How much of the source's semantic content did the summary capture?"
    Uses contextualized BERT embeddings for token-level semantic matching. Recall score close to 1.0
    means the summary represents most of the source's meaning. Score < 0.3 suggests missing content.

    Use this when: You want to measure semantic completeness (meaning overlap, not word overlap).

    Args:
        summary (str): Generated summary text to evaluate
        source (str, optional): Original source document text to summarize
        reference_summary (str, optional): Reference summary that represents a score of 5 (ideal quality)

    Returns:
        Dict: Result dictionary with keys:
            - recall (float): Semantic recall score from 0.0 to 1.0 (higher = more source content captured)
            - precision (float): How much of summary is relevant to source (0.0 to 1.0)
            - f1 (float): Harmonic mean of precision and recall (0.0 to 1.0)
            - interpretation (str): Human-readable label like "High Coverage" or "Low Coverage"
            - error (str, optional): Error message if bert-score package not installed

    Example:
        >>> result = compute_bertscore_recall_source(
        ...     summary="Short summary...",
        ...     source="Long source text..."
        ... )
        >>> result['recall']  # e.g., 0.45 (45% of source meaning captured)
        >>> result['interpretation']  # "Moderate Coverage"
    """
    try:
        from bert_score import score as bert_score

        # Validate required parameters
        if source is None and reference_summary is None:
            return {
                'recall': None,
                'error': 'Either source or reference_summary must be provided'
            }

        # Use source if provided, otherwise use reference_summary
        comparison_text = source if source is not None else reference_summary

        # BERTScore expects references and candidates
        # For completeness: how much of source is captured in summary
        # We use summary as candidate, source/reference as reference
        P, R, F1 = bert_score(
            [summary],  # candidates
            [comparison_text],   # references
            lang='en',
            rescale_with_baseline=True,
            device='cpu'
        )

        recall = float(R[0])

        return {
            'recall': round(recall, 4),
            'precision': round(float(P[0]), 4),
            'f1': round(float(F1[0]), 4),
            'interpretation': _interpret_bertscore_recall(recall),
            'error': None
        }

    except ImportError:
        return {
            'recall': None,
            'error': 'bert-score not installed. Run: pip install bert-score'
        }
    except Exception as e:
        return {
            'recall': None,
            'error': str(e)
        }


def _interpret_bertscore_recall(score: float) -> str:
    """Interpret BERTScore Recall for completeness."""
    if score >= 0.5:
        return "High Coverage"
    elif score >= 0.3:
        return "Moderate Coverage"
    elif score >= 0.1:
        return "Low Coverage"
    else:
        return "Very Low Coverage"


def compute_bartscore(
    summary: str,
    source: Optional[str] = None,
    reference_summary: Optional[str] = None,
    model_name: str = "facebook/bart-large-cnn"
) -> Dict:
    """
    Calculate how likely a pre-trained summarization model would generate this summary from the source.

    This metric answers: "Would a good summarization model produce this summary from this source?"
    Computes the log-probability that BART would generate the summary given the source.
    Higher scores (closer to 0) indicate better alignment. Very negative scores indicate poor coverage.

    Use this when: You want a probability-based assessment of summary quality from source.
    Note: Requires downloading a large model (~1.6GB). Disabled by default.

    Args:
        summary (str): Generated summary text to evaluate
        source (str, optional): Original source document text to summarize
        reference_summary (str, optional): Reference summary that represents a score of 5 (ideal quality)
        model_name (str, optional): HuggingFace BART model name. Default "facebook/bart-large-cnn" (~1.6GB)

    Returns:
        Dict: Result dictionary with keys:
            - score (float): BARTScore value (negative log-likelihood). Higher is better (e.g., -1.5 is better than -3.0)
            - interpretation (str): Human-readable label like "Excellent", "Good", "Moderate", or "Poor"
            - error (str, optional): Error message if transformers package not installed

    Example:
        >>> result = compute_bartscore(
        ...     summary="Generated summary...",
        ...     source="Source document..."
        ... )
        >>> result['score']  # e.g., -2.1 (moderate quality)
        >>> result['interpretation']  # "Good"
    """
    try:
        import torch
        from transformers import BartTokenizer, BartForConditionalGeneration

        # Validate required parameters
        if source is None and reference_summary is None:
            return {
                'score': None,
                'error': 'Either source or reference_summary must be provided'
            }

        # Use source if provided, otherwise use reference_summary
        comparison_text = source if source is not None else reference_summary

        # Load model and tokenizer
        tokenizer = BartTokenizer.from_pretrained(model_name)
        model = BartForConditionalGeneration.from_pretrained(model_name)
        model.eval()

        # Tokenize
        inputs = tokenizer(
            comparison_text,
            return_tensors='pt',
            truncation=True,
            max_length=1024
        )

        outputs = tokenizer(
            summary,
            return_tensors='pt',
            truncation=True,
            max_length=1024
        )

        # Compute log probability of summary given source
        with torch.no_grad():
            # Get model output
            result = model(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                labels=outputs['input_ids']
            )
            # Negative log likelihood loss
            nll = result.loss.item()

        # Convert to score (higher is better)
        # BARTScore is typically the negative of the loss
        bartscore = -nll

        return {
            'score': round(bartscore, 4),
            'interpretation': _interpret_bartscore(bartscore),
            'error': None
        }

    except ImportError as e:
        return {
            'score': None,
            'error': f'transformers not installed. Error: {str(e)}'
        }
    except Exception as e:
        return {
            'score': None,
            'error': str(e)
        }


def _interpret_bartscore(score: float) -> str:
    """Interpret BARTScore."""
    # BARTScore is negative log likelihood, so higher (less negative) is better
    if score >= -1.0:
        return "Excellent"
    elif score >= -2.0:
        return "Good"
    elif score >= -3.0:
        return "Moderate"
    else:
        return "Poor"


def compute_all_completeness_metrics(
    summary: str,
    source: Optional[str] = None,
    reference_summary: Optional[str] = None,
    use_semantic_coverage: bool = True,
    use_bertscore_recall: bool = True,
    use_bartscore: bool = False  # Disabled by default (large model)
) -> Dict[str, Dict]:
    """
    Run all available completeness metrics to check if summary captures key source information.

    This function computes multiple metrics that answer: "Did the summary include all the important
    information from the source?" Each metric uses a different approach:
    - Semantic Coverage: Sentence-level embedding similarity (fast, ~80MB model)
    - BERTScore Recall: Token-level BERT embeddings (medium, ~500MB)
    - BARTScore: Generative probability (slow, ~1.6GB, disabled by default)

    Use this when: You want a comprehensive completeness assessment with multiple metrics.

    Args:
        summary (str): Generated summary text to evaluate
        source (str, optional): Original source document text to summarize
        reference_summary (str, optional): Reference summary that represents a score of 5 (ideal quality)
        use_semantic_coverage (bool, optional): Enable sentence-level coverage check. Default True
        use_bertscore_recall (bool, optional): Enable BERTScore recall metric. Default True
        use_bartscore (bool, optional): Enable BARTScore (requires large model download). Default False

    Returns:
        Dict[str, Dict]: Dictionary mapping metric names to their results:
            - 'SemanticCoverage': Results from compute_semantic_coverage() if enabled
            - 'BERTScoreRecall': Results from compute_bertscore_recall_source() if enabled
            - 'BARTScore': Results from compute_bartscore() if enabled
            Each value is a dict with 'score', 'interpretation', and possibly 'error' keys.

    Example:
        >>> results = compute_all_completeness_metrics(
        ...     summary="Summary...",
        ...     source="Source..."
        ... )
        >>> results['SemanticCoverage']['score']  # e.g., 0.65
        >>> results['BERTScoreRecall']['recall']  # e.g., 0.42
        >>> list(results.keys())  # ['SemanticCoverage', 'BERTScoreRecall']
    """
    results = {}

    if use_semantic_coverage:
        results['SemanticCoverage'] = compute_semantic_coverage(
            summary=summary,
            source=source,
            reference_summary=reference_summary
        )

    if use_bertscore_recall:
        results['BERTScoreRecall'] = compute_bertscore_recall_source(
            summary=summary,
            source=source,
            reference_summary=reference_summary
        )

    if use_bartscore:
        results['BARTScore'] = compute_bartscore(
            summary=summary,
            source=source,
            reference_summary=reference_summary
        )

    return results
