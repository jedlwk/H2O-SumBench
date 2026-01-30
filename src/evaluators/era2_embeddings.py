"""
Era 2: Embedding-Based Metrics (Vibe Checkers).

These metrics measure semantic similarity using contextual embeddings,
capturing paraphrasing and meaning beyond exact word matches.
"""

import os
from typing import Dict
import warnings

# CRITICAL: Force CPU before ANY imports that might trigger PyTorch
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['CUDA_LAUNCH_BLOCKING'] = '0'
os.environ['USE_CUDA'] = '0'

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Global flag to track if MoverScore import has been attempted and failed
_MOVERSCORE_UNAVAILABLE = False
_MOVERSCORE_ERROR_MSG = None


def compute_bertscore(
    summary: str,
    source: str = None,
    reference_summary: str = None,
    model_type: str = "distilbert-base-uncased"
) -> Dict[str, float]:
    """
    Calculate semantic similarity between texts using contextualized BERT word embeddings.

    This metric answers: "Do the texts have similar meanings even with different wording?"
    Uses BERT to understand context: "bank" near "river" vs "bank" near "money" get different
    embeddings. Captures paraphrasing and semantic equivalence. Scores typically 0.7-0.9 for
    good paraphrases, 0.5-0.7 for partial matches.

    Use this when: You want to check if a summary conveys the same meaning as the reference,
    even if using completely different words (semantic conformance).

    Args:
        summary (str): Generated summary text to evaluate
        source (str, optional): Source document text to compare against
        reference_summary (str, optional): Reference summary that represents ideal quality
        model_type (str, optional): HuggingFace BERT model name. Default "distilbert-base-uncased" (~250MB)

    Returns:
        Dict[str, float]: Dictionary with BERTScore metrics:
            - precision (float): What fraction of summary tokens match source semantically (0.0 to 1.0)
            - recall (float): What fraction of source tokens are captured in summary (0.0 to 1.0)
            - f1 (float): Harmonic mean of precision and recall (0.0 to 1.0, main score to use)
            - error (str, optional): Error message if bert-score package not installed

    Example:
        >>> result = compute_bertscore(
        ...     summary="A feline rested on a rug.",
        ...     source="The cat sat on the mat."
        ... )
        >>> result['f1']  # e.g., 0.82 (high semantic similarity despite different words)
        >>> result['precision']  # e.g., 0.85
    """
    try:
        from bert_score import score

        # Validate required parameters
        if source is None and reference_summary is None:
            return {
                'precision': None,
                'recall': None,
                'f1': None,
                'error': 'Either source or reference_summary must be provided'
            }

        # Use source if provided, otherwise use reference_summary
        comparison_text = source if source is not None else reference_summary

        # Compute BERTScore
        P, R, F1 = score(
            [summary],
            [comparison_text],
            model_type=model_type,
            verbose=False,
            device='cpu'  # Use 'cuda' if GPU available
        )

        return {
            'precision': round(P.item(), 4),
            'recall': round(R.item(), 4),
            'f1': round(F1.item(), 4),
            'error': None
        }

    except Exception as e:
        return {
            'precision': None,
            'recall': None,
            'f1': None,
            'error': str(e)
        }


def compute_moverscore(
    summary: str,
    source: str = None,
    reference_summary: str = None,
    model_name: str = "distilbert-base-uncased"
) -> Dict[str, float]:
    """
    Calculate semantic alignment using optimal transport distance between word embeddings.

    This metric answers: "What's the minimum 'cost' to transform the summary's word meanings
    into the source's word meanings?" Uses Earth Mover's Distance (optimal transport) to find
    the best alignment between word embeddings. Lower distance = better semantic match.
    Typically scores 0.4-0.6 for good matches.

    Use this when: You want a more sophisticated semantic similarity metric that considers
    the optimal alignment of word meanings, not just token-by-token matching.

    Note: This package has known issues with CPU-only PyTorch. May return error if not properly configured.

    Args:
        summary (str): Generated summary text to evaluate
        source (str, optional): Source document text to compare against
        reference_summary (str, optional): Reference summary that represents ideal quality
        model_name (str, optional): HuggingFace model for embeddings. Default "distilbert-base-uncased"

    Returns:
        Dict[str, float]: Dictionary with MoverScore:
            - moverscore (float): Semantic alignment score, typically 0.0 to 1.0 (higher = better match)
            - error (str, optional): Error message if moverscore-v2 not installed or CUDA issues

    Example:
        >>> result = compute_moverscore(
        ...     summary="A feline was sitting.",
        ...     source="The cat sat."
        ... )
        >>> result['moverscore']  # e.g., 0.52 (moderate semantic alignment)
    """
    global _MOVERSCORE_UNAVAILABLE, _MOVERSCORE_ERROR_MSG

    # If MoverScore import previously failed, return cached error immediately
    if _MOVERSCORE_UNAVAILABLE:
        return {
            'moverscore': None,
            'error': _MOVERSCORE_ERROR_MSG or "MoverScore unavailable (previous import failed)"
        }

    # Validate required parameters
    if source is None and reference_summary is None:
        return {
            'moverscore': None,
            'error': 'Either source or reference_summary must be provided'
        }

    # Use source if provided, otherwise use reference_summary
    comparison_text = source if source is not None else reference_summary

    # Use the safe wrapper to import MoverScore
    try:
        from collections import defaultdict

        # Import from wrapper instead of moverscore_v2 directly
        try:
            from .moverscore_wrapper import get_idf_dict, word_mover_score
        except (ImportError, RuntimeError, AssertionError) as import_error:
            # CUDA error during import
            error_str = str(import_error)
            _MOVERSCORE_UNAVAILABLE = True
            _MOVERSCORE_ERROR_MSG = (
                f"MoverScore unavailable: {error_str}\n"
                "Try running setup.sh again to reinstall dependencies."
            )
            return {
                'moverscore': None,
                'error': _MOVERSCORE_ERROR_MSG
            }

        # Prepare texts (MoverScore expects list of strings, not list of word lists)
        references = [comparison_text]
        hypotheses = [summary]

        # Create IDF dictionary (uniform weights for simplicity)
        idf_dict_ref = defaultdict(lambda: 1.0)
        idf_dict_hyp = defaultdict(lambda: 1.0)

        # Compute MoverScore with explicit CPU device
        scores = word_mover_score(
            references,
            hypotheses,
            idf_dict_ref,
            idf_dict_hyp,
            stop_words=[],
            n_gram=1,
            remove_subwords=True,
            batch_size=1,
            device='cpu'  # Force CPU usage
        )

        return {
            'moverscore': round(scores[0], 4),
            'error': None
        }

    except ImportError as e:
        return {
            'moverscore': None,
            'error': (
                "MoverScore not installed. "
                "Run: pip3 install git+https://github.com/AIPHES/emnlp19-moverscore.git"
            )
        }
    except (AssertionError, RuntimeError) as e:
        # Handle CUDA-related errors during execution
        error_str = str(e)
        if "CUDA" in error_str or "cuda" in error_str.lower():
            return {
                'moverscore': None,
                'error': (
                    "MoverScore CUDA error. The package has a known issue with CPU-only PyTorch. "
                    "Reinstall PyTorch: pip3 uninstall torch && pip3 install torch --index-url https://download.pytorch.org/whl/cpu"
                )
            }
        return {
            'moverscore': None,
            'error': f"MoverScore error: {error_str}"
        }
    except Exception as e:
        error_msg = str(e)
        if "No module named" in error_msg or "cannot import" in error_msg:
            error_msg = "MoverScore not properly installed."
        elif "CUDA" in error_msg or "cuda" in error_msg.lower():
            error_msg = "MoverScore CUDA initialization error (known issue with CPU-only mode)."

        return {
            'moverscore': None,
            'error': error_msg
        }


def compute_all_era2_metrics(
    summary: str,
    source: str = None,
    reference_summary: str = None
) -> Dict[str, Dict[str, float]]:
    """
    Run all embedding-based semantic similarity metrics to compare summary against reference.

    This function computes 2 embedding metrics that answer: "Do the texts have similar meanings
    regardless of exact wording?" Both use BERT-based contextual embeddings to understand semantic
    content, making them excellent for evaluating paraphrases and semantic conformance.

    Use this when: You want to check if a generated summary captures the same meaning as a reference
    summary, even if using completely different vocabulary or sentence structures.

    Args:
        summary (str): Generated summary text to evaluate
        source (str, optional): Source document text to compare against
        reference_summary (str, optional): Reference summary that represents ideal quality

    Returns:
        Dict[str, Dict[str, float]]: Dictionary mapping metric names to their results:
            - 'BERTScore': Token-level semantic similarity (precision, recall, f1)
            - 'MoverScore': Optimal transport semantic alignment (moverscore)
            Each value is a dict with score keys and possibly 'error'.

    Example:
        >>> results = compute_all_era2_metrics(
        ...     summary="A fast auburn fox leaps.",
        ...     source="The quick brown fox jumps."
        ... )
        >>> results['BERTScore']['f1']  # e.g., 0.88 (high semantic match)
        >>> results['MoverScore']['moverscore']  # e.g., 0.65
        >>> list(results.keys())  # ['BERTScore', 'MoverScore']
    """
    return {
        'BERTScore': compute_bertscore(
            summary=summary,
            source=source,
            reference_summary=reference_summary
        ),
        'MoverScore': compute_moverscore(
            summary=summary,
            source=source,
            reference_summary=reference_summary
        )
    }
