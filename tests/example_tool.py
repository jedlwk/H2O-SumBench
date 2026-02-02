import json
import sys
import argparse
from typing import Dict

def compute_rouge_scores(
    summary: str,
    source: str = None,
    reference_summary: str = None
) -> Dict[str, float]:
    """
    Calculate word and phrase overlap between summary and reference text using ROUGE metrics.

    This metric answers: "How many words/phrases match between the summary and reference?"
    ROUGE-1 counts single word matches, ROUGE-2 counts two-word phrase matches, ROUGE-L finds
    the longest common sequence. Scores range 0.0 to 1.0 (higher = more overlap).

    Use this when: You want to compare a generated summary against a reference summary to check
    if it uses similar wording and captures the same information.

    Args:
        summary (str): Generated summary text to evaluate
        source (str, optional): Source document text to compare against
        reference_summary (str, optional): Reference summary that represents ideal quality

    Returns:
        Dict[str, float]: Dictionary with ROUGE scores:
            - rouge1 (float): Single word overlap F1 score (0.0 to 1.0)
            - rouge2 (float): Two-word phrase overlap F1 score (0.0 to 1.0)
            - rougeL (float): Longest common subsequence F1 score (0.0 to 1.0)
            - error (str, optional): Error message if computation failed

    Example:
        >>> result = compute_rouge_scores(
        ...     summary="A cat sat on a mat.",
        ...     source="The cat sat on the mat."
        ... )
        >>> result['rouge1']  # e.g., 0.85 (high single-word overlap)
        >>> result['rouge2']  # e.g., 0.67 (good phrase overlap)
    """
    try:
        from rouge_score import rouge_scorer

        # Validate required parameters
        if source is None and reference_summary is None:
            return {
                'rouge1': None,
                'rouge2': None,
                'rougeL': None,
                'error': 'Either source or reference_summary must be provided'
            }

        # Use source if provided, otherwise use reference_summary
        comparison_text = source if source is not None else reference_summary

        scorer = rouge_scorer.RougeScorer(
            ['rouge1', 'rouge2', 'rougeL'],
            use_stemmer=True
        )

        scores = scorer.score(comparison_text, summary)

        return {
            'rouge1': round(scores['rouge1'].fmeasure, 4),
            'rouge2': round(scores['rouge2'].fmeasure, 4),
            'rougeL': round(scores['rougeL'].fmeasure, 4),
            'error': None
        }

    except Exception as e:
        return {
            'rouge1': None,
            'rouge2': None,
            'rougeL': None,
            'error': str(e)
        }
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", required=True)
    parser.add_argument("--reference", required=False)
    parser.add_argument("--source", required=False)
    
    args = parser.parse_args()
    
    result = compute_rouge_scores(
        summary=args.summary,
        reference_summary=args.reference,
        source=args.source
    )
    
    # The agent reads the STDOUT, so print the result as JSON
    print(json.dumps(result))