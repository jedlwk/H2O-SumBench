"""
UniEval: Unified Multi-Dimensional Evaluator.

UniEval is a unified evaluator that converts evaluation tasks into Boolean QA problems.
It provides multi-dimensional evaluation: coherence, consistency, fluency.

This serves as the backup for BLEURT (which has TensorFlow conflicts).

Paper: "Towards a Unified Multi-Dimensional Evaluator for Text Generation" (EMNLP 2022)
Model: MingZhong/unieval-sum (~1.5GB)
"""

import os
from typing import Dict, List, Optional
import warnings

# Force CPU mode
os.environ['CUDA_VISIBLE_DEVICES'] = ''
warnings.filterwarnings('ignore')

# Global cache for UniEval evaluator (lazy loaded)
_unieval_evaluator = None


def compute_unieval(
    source: str,
    summary: str,
    dimensions: Optional[List[str]] = None
) -> Dict:
    """
    Evaluate summary quality across multiple dimensions using UniEval's Boolean QA framework.

    This metric answers: "Is the summary coherent, factually consistent, and fluent?" UniEval
    converts evaluation into Boolean question-answering: asking yes/no questions for each dimension
    and converting to scores. Uses the MingZhong/unieval-sum model (~1.5GB) trained on human judgments.
    Scores range 0.0 to 1.0 for each dimension.

    Use this when: You want multi-dimensional quality assessment (coherence, consistency, fluency)
    from a unified model. Good alternative to BLEURT which has TensorFlow conflicts. Requires model download.

    Args:
        source (str): Original source document text to check consistency against
        summary (str): Generated summary text to evaluate
        dimensions (Optional[List[str]], optional): List of dimensions to evaluate.
            Options: ["coherence", "consistency", "fluency"]. Default None evaluates all three.

    Returns:
        Dict: Result dictionary with keys:
            - coherence (float): Logical flow score from 0.0 to 1.0 (higher = better structure)
            - consistency (float): Factual alignment score from 0.0 to 1.0 (higher = more faithful)
            - fluency (float): Writing quality score from 0.0 to 1.0 (higher = more natural)
            - interpretations (Dict): Human-readable labels for each dimension:
                - "Excellent" (≥0.85), "Good" (≥0.70), "Fair" (≥0.50), "Poor" (≥0.30), "Very Poor" (<0.30)
            - error (str, optional): Error message if UniEval package not installed
            - note (str, optional): Indicates if fallback implementation was used

    Example:
        >>> result = compute_unieval("Paris is capital.", "Paris is French capital.")
        >>> result['consistency']  # e.g., 0.92 (high factual consistency)
        >>> result['interpretations']['consistency']  # "Excellent"
        >>> result['coherence']  # e.g., 0.88
    """
    global _unieval_evaluator

    if dimensions is None:
        dimensions = ["coherence", "consistency", "fluency"]

    try:
        # Try to import UniEval
        from UniEval.utils import convert_to_json
        from UniEval.metric.evaluator import get_evaluator

        # Load evaluator if not cached
        if _unieval_evaluator is None:
            _unieval_evaluator = get_evaluator("summarization", device="cpu")

        # Format data for UniEval
        data = convert_to_json(
            output_list=[summary],
            src_list=[source]
        )

        # Evaluate
        eval_scores = _unieval_evaluator.evaluate(
            data,
            dims=dimensions,
            overall=False,
            print_result=False
        )

        # Extract scores from first result
        result = eval_scores[0] if eval_scores else {}

        return {
            'coherence': round(result.get('coherence', 0), 4),
            'consistency': round(result.get('consistency', 0), 4),
            'fluency': round(result.get('fluency', 0), 4),
            'interpretations': {
                'coherence': _interpret_unieval_score(result.get('coherence')),
                'consistency': _interpret_unieval_score(result.get('consistency')),
                'fluency': _interpret_unieval_score(result.get('fluency'))
            }
        }

    except ImportError:
        # UniEval not installed - try alternative approach using transformers directly
        return _compute_unieval_fallback(source, summary, dimensions)

    except Exception as e:
        return {
            'coherence': None,
            'consistency': None,
            'fluency': None,
            'error': str(e)
        }


def _compute_unieval_fallback(
    source: str,
    summary: str,
    dimensions: List[str]
) -> Dict:
    """
    Fallback implementation using HuggingFace transformers when UniEval library is not installed.

    This function provides an alternative way to use the UniEval model by directly loading it from
    HuggingFace and performing inference. Converts evaluation questions into Boolean QA format
    (e.g., "Is this summary coherent?") and interprets yes/no responses as scores (1.0 or 0.0).

    Use this when: UniEval library is not installed but transformers package is available.
    Internal fallback function called automatically by compute_unieval().

    Args:
        source (str): Original source document text
        summary (str): Generated summary to evaluate
        dimensions (List[str]): List of dimensions to evaluate (coherence, consistency, fluency)

    Returns:
        Dict: Same format as compute_unieval() with coherence, consistency, fluency scores,
              interpretations, and a 'note' field indicating fallback was used
    """
    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

        model_name = "MingZhong/unieval-sum"

        # Load model and tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        model.eval()

        results = {}

        # UniEval uses Boolean QA format for each dimension
        dimension_questions = {
            'coherence': f"question: Is this a coherent summary? </s> summary: {summary}",
            'consistency': f"question: Is this summary consistent with the document? </s> document: {source} </s> summary: {summary}",
            'fluency': f"question: Is this summary fluent? </s> summary: {summary}"
        }

        for dim in dimensions:
            if dim not in dimension_questions:
                continue

            input_text = dimension_questions[dim]

            # Truncate if too long
            inputs = tokenizer(
                input_text,
                return_tensors="pt",
                truncation=True,
                max_length=1024
            )

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=10,
                    num_beams=1
                )

            response = tokenizer.decode(outputs[0], skip_special_tokens=True).lower()

            # UniEval outputs "yes" or "no" - convert to score
            if "yes" in response:
                score = 1.0
            elif "no" in response:
                score = 0.0
            else:
                # Try to extract probability-based score
                score = 0.5

            results[dim] = round(score, 4)

        return {
            'coherence': results.get('coherence', None),
            'consistency': results.get('consistency', None),
            'fluency': results.get('fluency', None),
            'interpretations': {
                'coherence': _interpret_unieval_score(results.get('coherence')),
                'consistency': _interpret_unieval_score(results.get('consistency')),
                'fluency': _interpret_unieval_score(results.get('fluency'))
            },
            'note': 'Using fallback implementation (UniEval library not installed)'
        }

    except ImportError as e:
        return {
            'coherence': None,
            'consistency': None,
            'fluency': None,
            'error': f'transformers not installed. Error: {str(e)}'
        }
    except Exception as e:
        return {
            'coherence': None,
            'consistency': None,
            'fluency': None,
            'error': str(e)
        }


def _interpret_unieval_score(score: Optional[float]) -> str:
    """Interpret UniEval dimension scores (0-1 scale)."""
    if score is None:
        return "N/A"
    if score >= 0.85:
        return "Excellent"
    elif score >= 0.70:
        return "Good"
    elif score >= 0.50:
        return "Fair"
    elif score >= 0.30:
        return "Poor"
    else:
        return "Very Poor"


def compute_all_unieval_metrics(source: str, summary: str) -> Dict[str, Dict]:
    """
    Run all UniEval dimensions (coherence, consistency, fluency) in a single evaluation call.

    This function provides a convenient wrapper to evaluate all three UniEval dimensions at once,
    returning results in a structured format consistent with other evaluator modules. Evaluates
    logical flow, factual consistency, and writing quality using the unified UniEval framework.

    Use this when: You want to compute all UniEval metrics in one call with consistent output format.
    Wrapper function that calls compute_unieval() with all dimensions enabled.

    Args:
        source (str): Original source document text to check consistency against
        summary (str): Generated summary text to evaluate

    Returns:
        Dict[str, Dict]: Dictionary with single key 'UniEval' mapping to results:
            - 'UniEval': Dictionary with coherence, consistency, fluency scores and interpretations
              (same format as compute_unieval() return value)

    Example:
        >>> results = compute_all_unieval_metrics("Source text...", "Summary text...")
        >>> results['UniEval']['coherence']  # e.g., 0.85
        >>> results['UniEval']['consistency']  # e.g., 0.90
        >>> results['UniEval']['fluency']  # e.g., 0.88
        >>> list(results.keys())  # ['UniEval']
    """
    return {
        'UniEval': compute_unieval(source, summary)
    }
