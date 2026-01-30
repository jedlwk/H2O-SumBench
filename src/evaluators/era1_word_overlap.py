"""
Era 1: Word Overlap Metrics (Bean Counters).

These metrics measure text similarity through exact word matching,
with some variants supporting synonyms and stemming.
"""

from typing import Dict
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Global cache for GPT-2 perplexity model (~600MB)
_perplexity_model = None
_perplexity_tokenizer = None


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


def compute_bleu_score(
    summary: str,
    source: str = None,
    reference_summary: str = None
) -> Dict[str, float]:
    """
    Calculate n-gram precision between summary and reference using BLEU score.

    This metric answers: "What percentage of words/phrases in the summary appear in the reference?"
    Originally designed for machine translation. Measures precision (not recall), so shorter
    summaries that match well score higher. Typical scores for summaries: 0.2-0.4 is good.

    Use this when: You want to check if the generated text uses the same vocabulary as the reference,
    especially for tasks where exact wording matters.

    Args:
        summary (str): Generated summary text to evaluate
        source (str, optional): Source document text to compare against
        reference_summary (str, optional): Reference summary that represents ideal quality

    Returns:
        Dict[str, float]: Dictionary with BLEU score:
            - bleu (float): Precision score from 0.0 to 1.0 (0.3+ is good for summaries)
            - error (str, optional): Error message if computation failed

    Example:
        >>> result = compute_bleu_score(
        ...     summary="The cat sat down.",
        ...     source="The cat sat."
        ... )
        >>> result['bleu']  # e.g., 0.65 (good precision match)
    """
    try:
        from sacrebleu import sentence_bleu

        # Validate required parameters
        if source is None and reference_summary is None:
            return {
                'bleu': None,
                'error': 'Either source or reference_summary must be provided'
            }

        # Use source if provided, otherwise use reference_summary
        comparison_text = source if source is not None else reference_summary

        # BLEU expects references as a list
        score = sentence_bleu(summary, [comparison_text])

        return {
            'bleu': round(score.score / 100, 4),  # Normalize to 0-1
            'error': None
        }

    except Exception as e:
        return {
            'bleu': None,
            'error': str(e)
        }


def compute_meteor_score(
    summary: str,
    source: str = None,
    reference_summary: str = None
) -> Dict[str, float]:
    """
    Calculate semantic word overlap considering synonyms and word stems using METEOR.

    This metric answers: "Do the texts match even with different but equivalent wording?"
    More flexible than BLEU/ROUGE: "car" matches "automobile", "running" matches "runs".
    Considers word order and alignment. Scores typically higher than BLEU.

    Use this when: You want forgiving word matching that recognizes synonyms and word forms,
    good for evaluating paraphrasing quality.

    Args:
        summary (str): Generated summary text to evaluate
        source (str, optional): Source document text to compare against
        reference_summary (str, optional): Reference summary that represents ideal quality

    Returns:
        Dict[str, float]: Dictionary with METEOR score:
            - meteor (float): Alignment score from 0.0 to 1.0 (higher = better semantic match)
            - error (str, optional): Error message if NLTK data not available

    Example:
        >>> result = compute_meteor_score(
        ...     summary="A feline rushed rapidly.",
        ...     source="The cat ran quickly."
        ... )
        >>> result['meteor']  # e.g., 0.45 (recognizes synonym pairs)
    """
    try:
        from nltk.translate.meteor_score import meteor_score
        from nltk.tokenize import word_tokenize
        import nltk
        import ssl

        # Validate required parameters
        if source is None and reference_summary is None:
            return {
                'meteor': None,
                'error': 'Either source or reference_summary must be provided'
            }

        # Use source if provided, otherwise use reference_summary
        comparison_text = source if source is not None else reference_summary

        # Handle SSL certificate issues
        try:
            _create_unverified_https_context = ssl._create_unverified_context
        except AttributeError:
            pass
        else:
            ssl._create_default_https_context = _create_unverified_https_context

        # Ensure required NLTK data is available
        required_data = [
            ('corpora/wordnet', 'wordnet'),
            ('corpora/omw-1.4', 'omw-1.4'),
            ('tokenizers/punkt', 'punkt'),
            ('tokenizers/punkt_tab', 'punkt_tab')
        ]

        for path, name in required_data:
            try:
                nltk.data.find(path)
            except LookupError:
                try:
                    nltk.download(name, quiet=True)
                except:
                    pass  # Continue even if download fails

        # Tokenize texts
        source_tokens = word_tokenize(comparison_text.lower())
        summary_tokens = word_tokenize(summary.lower())

        # Compute METEOR
        score = meteor_score([source_tokens], summary_tokens)

        return {
            'meteor': round(score, 4),
            'error': None
        }

    except Exception as e:
        error_msg = str(e)
        if 'punkt_tab' in error_msg:
            error_msg = "NLTK punkt_tab missing. Run: python3 -c \"import nltk; nltk.download('punkt_tab')\""
        return {
            'meteor': None,
            'error': error_msg
        }


def compute_levenshtein_score(
    summary: str,
    source: str = None,
    reference_summary: str = None
) -> Dict[str, float]:
    """
    Calculate character-level edit distance similarity between two texts.

    This metric answers: "How many character insertions/deletions/substitutions to make texts match?"
    Measures string similarity at character level. 1.0 = identical, 0.0 = completely different.
    Sensitive to typos, spacing, and punctuation differences.

    Use this when: You want to measure exact string similarity including formatting, or detect
    near-duplicate texts with minor differences.

    Args:
        summary (str): Generated summary text to evaluate
        source (str, optional): Source document text to compare against
        reference_summary (str, optional): Reference summary that represents ideal quality

    Returns:
        Dict[str, float]: Dictionary with Levenshtein similarity:
            - levenshtein (float): Normalized similarity from 0.0 to 1.0 (1.0 = identical strings)
            - error (str, optional): Error message if python-Levenshtein not installed

    Example:
        >>> result = compute_levenshtein_score(
        ...     summary="hello word",
        ...     source="hello world"
        ... )
        >>> result['levenshtein']  # e.g., 0.91 (1 character different out of 11)
    """
    try:
        import Levenshtein

        # Validate required parameters
        if source is None and reference_summary is None:
            return {
                'levenshtein': None,
                'error': 'Either source or reference_summary must be provided'
            }

        # Use source if provided, otherwise use reference_summary
        comparison_text = source if source is not None else reference_summary

        # Compute distance
        distance = Levenshtein.distance(comparison_text, summary)

        # Normalize to similarity score (0-1)
        max_length = max(len(comparison_text), len(summary))
        if max_length == 0:
            similarity = 1.0
        else:
            similarity = 1 - (distance / max_length)

        return {
            'levenshtein': round(similarity, 4),
            'error': None
        }

    except Exception as e:
        return {
            'levenshtein': None,
            'error': str(e)
        }


def compute_perplexity(
    summary: str,
    source: str = None,
    reference_summary: str = None
) -> Dict[str, float]:
    """
    Measure text fluency and naturalness using GPT-2 language model perplexity.

    This metric answers: "Does the summary sound natural and grammatically correct?"
    Lower perplexity (< 50) = fluent natural text. Higher (> 200) = awkward/unnatural.
    Note: This only checks fluency, NOT factual accuracy or meaning.

    Use this when: You want to check if text is grammatical and naturally written,
    independent of whether it matches the source content.

    Args:
        summary (str): Generated summary text to evaluate for fluency
        source (str, optional): Unused, kept for API consistency
        reference_summary (str, optional): Unused, kept for API consistency

    Returns:
        Dict[str, float]: Dictionary with perplexity scores:
            - perplexity (float): Raw perplexity value (lower is better, typically 10-200)
            - normalized_score (float): Normalized fluency score 0.0 to 1.0 (higher = more fluent)
            - error (str, optional): Error message if transformers or GPT-2 not available

    Example:
        >>> result = compute_perplexity(summary="The cat sat on the mat.")
        >>> result['perplexity']  # e.g., 28.5 (fluent text)
        >>> result['normalized_score']  # e.g., 0.72 (normalized fluency)
    """
    global _perplexity_model, _perplexity_tokenizer

    try:
        from transformers import GPT2LMHeadModel, GPT2TokenizerFast
        import torch
        import math

        # Load GPT-2 model (cached globally to avoid reload overhead)
        if _perplexity_model is None:
            model_id = "gpt2"
            _perplexity_model = GPT2LMHeadModel.from_pretrained(model_id)
            _perplexity_tokenizer = GPT2TokenizerFast.from_pretrained(model_id)
            _perplexity_model.eval()

        model = _perplexity_model
        tokenizer = _perplexity_tokenizer

        # Encode the summary
        encodings = tokenizer(summary, return_tensors="pt", truncation=True, max_length=512)

        # Calculate perplexity
        with torch.no_grad():
            outputs = model(**encodings, labels=encodings["input_ids"])
            loss = outputs.loss
            perplexity = math.exp(loss.item())

        # Normalize perplexity to 0-1 scale (lower is better)
        # Use inverse and normalize: good perplexity < 50, bad > 200
        normalized = 1 / (1 + math.log(perplexity))

        return {
            'perplexity': round(perplexity, 4),
            'normalized_score': round(normalized, 4),
            'error': None
        }

    except Exception as e:
        return {
            'perplexity': None,
            'normalized_score': None,
            'error': str(e)
        }


def compute_chrf_score(
    summary: str,
    source: str = None,
    reference_summary: str = None
) -> Dict[str, float]:
    """
    Calculate character n-gram overlap with word order consideration using chrF++.

    This metric answers: "Do texts match at character level including partial word matches?"
    More forgiving than word-level metrics: handles typos, morphology ("running"/"runs"),
    and compound words better. Useful for morphologically rich languages.

    Use this when: You want robust matching that tolerates spelling variations, word endings,
    and morphological differences while still considering word order.

    Args:
        summary (str): Generated summary text to evaluate
        source (str, optional): Source document text to compare against
        reference_summary (str, optional): Reference summary that represents ideal quality

    Returns:
        Dict[str, float]: Dictionary with chrF++ scores:
            - chrf (float): Normalized character F-score from 0.0 to 1.0 (higher = better match)
            - raw_score (float): Raw score from 0 to 100 (not normalized)
            - error (str, optional): Error message if sacrebleu not installed

    Example:
        >>> result = compute_chrf_score(
        ...     summary="The runners run quickly.",
        ...     source="The runner quickly ran."
        ... )
        >>> result['chrf']  # e.g., 0.78 (high character-level similarity despite word differences)
    """
    try:
        from sacrebleu.metrics import CHRF

        # Validate required parameters
        if source is None and reference_summary is None:
            return {
                'chrf': None,
                'error': 'Either source or reference_summary must be provided'
            }

        # Use source if provided, otherwise use reference_summary
        comparison_text = source if source is not None else reference_summary

        chrf = CHRF(word_order=2)  # chrF++ includes word bigrams
        score = chrf.sentence_score(summary, [comparison_text])

        return {
            'chrf': round(score.score / 100, 4),  # Normalize to 0-1
            'raw_score': round(score.score, 2),
            'error': None
        }

    except ImportError:
        return {
            'chrf': None,
            'error': 'sacrebleu not installed. Run: pip install sacrebleu'
        }
    except Exception as e:
        return {
            'chrf': None,
            'error': str(e)
        }


def compute_all_era1_metrics(
    summary: str,
    source: str = None,
    reference_summary: str = None
) -> Dict[str, Dict[str, float]]:
    """
    Run all word-overlap and lexical similarity metrics to compare summary against reference.

    This function computes 6 lexical metrics that answer: "How well does the summary match
    the reference's exact words, phrases, and character sequences?" Useful for checking if
    a generated summary uses similar vocabulary and structure as a human-written reference.

    Use this when: You have a reference summary and want comprehensive lexical comparison
    using multiple complementary approaches (word n-grams, characters, synonyms, fluency).

    Args:
        summary (str): Generated summary text to evaluate
        source (str, optional): Source document text to compare against
        reference_summary (str, optional): Reference summary that represents ideal quality

    Returns:
        Dict[str, Dict[str, float]]: Dictionary mapping metric names to their results:
            - 'ROUGE': N-gram overlap scores (rouge1, rouge2, rougeL)
            - 'BLEU': Precision-based n-gram score
            - 'METEOR': Synonym and stem-aware alignment score
            - 'chrF++': Character-level n-gram F-score
            - 'Levenshtein': Character edit distance similarity
            - 'Perplexity': Fluency score using GPT-2
            Each value is a dict with score keys and possibly 'error'.

    Example:
        >>> results = compute_all_era1_metrics(
        ...     summary="Generated summary...",
        ...     source="Reference summary..."
        ... )
        >>> results['ROUGE']['rouge1']  # e.g., 0.65
        >>> results['BLEU']['bleu']  # e.g., 0.28
        >>> list(results.keys())  # ['ROUGE', 'BLEU', 'METEOR', 'chrF++', 'Levenshtein', 'Perplexity']
    """
    return {
        'ROUGE': compute_rouge_scores(
            summary=summary,
            source=source,
            reference_summary=reference_summary
        ),
        'BLEU': compute_bleu_score(
            summary=summary,
            source=source,
            reference_summary=reference_summary
        ),
        'METEOR': compute_meteor_score(
            summary=summary,
            source=source,
            reference_summary=reference_summary
        ),
        'chrF++': compute_chrf_score(
            summary=summary,
            source=source,
            reference_summary=reference_summary
        ),
        'Levenshtein': compute_levenshtein_score(
            summary=summary,
            source=source,
            reference_summary=reference_summary
        ),
        'Perplexity': compute_perplexity(
            summary=summary,
            source=source,
            reference_summary=reference_summary
        )
    }
