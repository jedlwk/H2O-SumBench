"""
Era 3 Group A: Logic Checkers (Truth Squad).

This module implements logic-based consistency checking:
- NLI (DeBERTa-v3): Local Natural Language Inference
- AlignScore: Unified alignment-based factual consistency metric
- FactChecker (API): API-based fact-checking using LLMs
"""

import os
from typing import Dict, Optional
import warnings

# Force CPU mode for PyTorch/Transformers
os.environ['CUDA_VISIBLE_DEVICES'] = ''

warnings.filterwarnings('ignore')

# Global model caches to avoid reloading on each call
_nli_pipeline = None
_factcc_pipeline = None


def compute_nli_score(
    summary: str,
    source: str = None,
    reference_summary: str = None,
    model_name: str = "microsoft/deberta-v3-base"
) -> Dict[str, float]:
    """
    Check if summary claims are logically supported by source using Natural Language Inference.

    This metric answers: "Can the summary's statements be logically inferred from the source?"
    Uses DeBERTa-v3 NLI model to classify the relationship as entailment/neutral/contradiction.
    Score > 0.7 = summary is well-supported. Score < 0.4 = potential contradictions or hallucinations.

    Use this when: You want to detect factual inconsistencies, hallucinations, or unsupported claims
    in summaries. This is the recommended first-line defense against factual errors.

    Note: Truncates texts to ~400 words due to model token limit.

    Args:
        summary (str): Generated summary to check for logical consistency
        source (str, optional): Original source document that should support the summary
        reference_summary (str, optional): Not used for NLI (kept for API consistency)
        model_name (str, optional): HuggingFace NLI model. Default "microsoft/deberta-v3-base" (~600MB)

    Returns:
        Dict[str, float]: Result dictionary with keys:
            - nli_score (float): Entailment probability from 0.0 to 1.0 (higher = more consistent)
            - label (str): Classification label like "LABEL_0" (entailment) or "LABEL_2" (contradiction)
            - interpretation (str): Human-readable label like "Highly Consistent" or "Inconsistent"
            - error (str, optional): Error message if transformers not available

    Example:
        >>> result = compute_nli_score(
        ...     summary="Paris is French capital.",
        ...     source="Paris is capital of France."
        ... )
        >>> result['nli_score']  # e.g., 0.92 (high entailment)
        >>> result['interpretation']  # "Highly Consistent"
    """
    global _nli_pipeline

    # Validate required parameters
    if source is None:
        return {
            'nli_score': None,
            'label': None,
            'error': 'Source document is required for NLI evaluation'
        }

    try:
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
            pipeline
        )

        # Load NLI model (cached globally)
        if _nli_pipeline is None:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForSequenceClassification.from_pretrained(model_name)
            _nli_pipeline = pipeline(
                "text-classification",
                model=model,
                tokenizer=tokenizer,
                device=-1  # Force CPU
            )

        nli_pipeline = _nli_pipeline

        # Truncate long texts (DeBERTa max length: 512 tokens)
        max_words = 400  # Conservative limit
        source_truncated = ' '.join(source.split()[:max_words])
        summary_truncated = ' '.join(summary.split()[:max_words])

        # Format: premise [SEP] hypothesis
        nli_input = f"{source_truncated} [SEP] {summary_truncated}"

        # Get NLI prediction
        result = nli_pipeline(nli_input, truncation=True, max_length=512)

        label = result[0]['label']
        score = result[0]['score']

        # Extract entailment score
        # DeBERTa outputs: LABEL_0 (entailment), LABEL_1 (neutral), LABEL_2 (contradiction)
        if 'ENTAIL' in label.upper() or label == 'LABEL_0':
            entailment_score = score
        else:
            # If neutral or contradiction, invert score
            entailment_score = 1 - score

        return {
            'nli_score': round(entailment_score, 4),
            'label': label,
            'interpretation': _interpret_nli_score(entailment_score),
            'error': None
        }

    except Exception as e:
        return {
            'nli_score': None,
            'label': 'ERROR',
            'error': str(e)
        }


def _interpret_nli_score(score: float) -> str:
    """
    Interpret NLI score for human readability.

    Args:
        score: NLI entailment score (0-1).

    Returns:
        Human-readable interpretation.
    """
    if score >= 0.8:
        return "Highly Consistent"
    elif score >= 0.6:
        return "Mostly Consistent"
    elif score >= 0.4:
        return "Partially Consistent"
    else:
        return "Inconsistent"


def compute_factchecker_score(
    summary: str,
    source: str = None,
    reference_summary: str = None,
    model_name: Optional[str] = None,
    use_api: bool = True
) -> Dict:
    """
    Perform detailed claim-by-claim fact-checking using a large language model API.

    This metric answers: "Which specific claims in the summary are unsupported or contradicted?"
    Uses an LLM (default: Llama-3.3-70B) to extract each factual claim from the summary and verify
    it against the source. Provides detailed breakdown of claims checked, issues found, and explanations.

    Use this when: You want granular, explainable fact-checking with specific claim-level analysis.
    Requires H2OGPTE API access. Slower but more detailed than local models.

    Note: Requires H2OGPTE_API_KEY and H2OGPTE_ADDRESS environment variables.

    Args:
        summary (str): Generated summary with claims to fact-check
        source (str, optional): Original source document to verify claims against
        reference_summary (str, optional): Not used for fact-checking (kept for API consistency)
        model_name (str, optional): LLM model name. Default "meta-llama/Llama-3.3-70B-Instruct"
        use_api (bool, optional): Whether to actually call API. Default True

    Returns:
        Dict: Result dictionary with keys:
            - score (float): Factuality score from 0.0 to 1.0 (normalized from 1-10 scale)
            - raw_score (float): Original score from 1 to 10
            - claims_checked (int): Total number of factual claims extracted from summary
            - issues_found (int): Number of unsupported or contradicted claims
            - explanation (str): Detailed explanation of findings
            - full_response (str): Complete LLM response for debugging
            - interpretation (str): Human-readable label like "Fully Factual" or "Partially Factual"
            - error (str, optional): Error message if API not configured

    Example:
        >>> result = compute_factchecker_score(
        ...     summary="The dog ate.",
        ...     source="The cat ate."
        ... )
        >>> result['claims_checked']  # 1
        >>> result['issues_found']  # 1 (wrong animal)
        >>> result['score']  # e.g., 0.3 (30% factual)
    """
    # Validate required parameters
    if source is None:
        return {
            'score': None,
            'error': 'Source document is required for fact-checking'
        }

    if not use_api:
        return {
            'score': None,
            'error': 'API-based fact-checking disabled'
        }

    try:
        # Check if API is configured
        from dotenv import load_dotenv
        load_dotenv()

        api_key = os.getenv('H2OGPTE_API_KEY')
        address = os.getenv('H2OGPTE_ADDRESS')

        if not api_key or not address:
            return {
                'score': None,
                'error': 'API not configured (H2OGPTE_API_KEY or H2OGPTE_ADDRESS missing)'
            }

        # Import LLM client
        from h2ogpte import H2OGPTE

        # Use provided model or default
        if model_name is None:
            model_name = 'meta-llama/Llama-3.3-70B-Instruct'

        client = H2OGPTE(address=address, api_key=api_key)

        # Create fact-checking prompt
        prompt = f"""You are an expert fact-checker. Your task is to verify EVERY claim in the summary against the source document.

**Source Document**:
{source}

**Summary to Fact-Check**:
{summary}

**Instructions**:
1. Extract each factual claim from the summary
2. For each claim, check if it is:
   - SUPPORTED: Directly stated or clearly implied in the source
   - UNSUPPORTED: Not mentioned or cannot be verified from source
   - CONTRADICTED: Conflicts with information in the source
3. Count the claims and issues
4. Provide a factuality score from 1-10

**Response Format**:
Claims Checked: [number]
Issues Found: [number of unsupported or contradicted claims]
Score: [1-10]
Explanation: [Brief summary of any issues found, or "All claims verified" if perfect]"""

        # Query the LLM
        chat_session_id = client.create_chat_session()

        with client.connect(chat_session_id) as session:
            reply = session.query(
                prompt,
                llm=model_name,
                timeout=90,
            )

        response = reply.content

        # Parse response
        claims_checked = 0
        issues_found = 0
        score = None
        explanation = ""

        for line in response.split('\n'):
            line = line.strip()
            if line.startswith('Claims Checked:'):
                try:
                    claims_checked = int(line.split(':')[1].strip())
                except:
                    pass
            elif line.startswith('Issues Found:'):
                try:
                    issues_found = int(line.split(':')[1].strip())
                except:
                    pass
            elif line.startswith('Score:'):
                try:
                    score_text = line.split(':')[1].strip()
                    score = float(score_text)
                except:
                    pass
            elif line.startswith('Explanation:'):
                explanation = line.split(':', 1)[1].strip()

        # Normalize score to 0-1
        if score is not None:
            normalized_score = score / 10.0
        else:
            normalized_score = None

        return {
            'score': normalized_score,
            'raw_score': score,
            'claims_checked': claims_checked,
            'issues_found': issues_found,
            'explanation': explanation if explanation else 'No explanation provided',
            'full_response': response,
            'interpretation': _interpret_factchecker_score(normalized_score) if normalized_score else 'N/A',
            'error': None
        }

    except Exception as e:
        return {
            'score': None,
            'error': str(e)
        }


def _interpret_factchecker_score(score: float) -> str:
    """Interpret fact-checker score."""
    if score is None:
        return 'N/A'
    if score >= 0.9:
        return "Fully Factual"
    elif score >= 0.7:
        return "Mostly Factual"
    elif score >= 0.5:
        return "Partially Factual"
    else:
        return "Factually Questionable"


def compute_factcc_score(
    summary: str,
    source: str = None,
    reference_summary: str = None
) -> Dict:
    """
    Check factual consistency using a BERT model specifically trained for summarization fact-checking.

    This metric answers: "Is this summary factually consistent with the source?"
    Uses DeBERTa-base-MNLI model (similar to original FactCC) trained specifically for detecting
    factual errors in summaries. Scores > 0.7 indicate consistency, < 0.4 suggests inconsistencies.

    Use this when: You want a specialized fact-checking model trained specifically for summarization
    tasks, complementary to general NLI models.

    Note: Uses DeBERTa-base-MNLI as alternative to original FactCC checkpoint. Truncates to ~400 words.

    Args:
        summary (str): Generated summary to fact-check
        source (str, optional): Original source document to verify claims against
        reference_summary (str, optional): Not used for FactCC (kept for API consistency)

    Returns:
        Dict: Result dictionary with keys:
            - score (float): Consistency score from 0.0 to 1.0 (higher = more consistent)
            - label (str): "Consistent" or "Inconsistent"
            - raw_label (str): Original model label before interpretation
            - interpretation (str): Human-readable label like "Highly Consistent"
            - error (str, optional): Error message if model not available

    Example:
        >>> result = compute_factcc_score(
        ...     summary="The dog ate food.",
        ...     source="The cat ate food."
        ... )
        >>> result['score']  # e.g., 0.35 (low consistency - different animal)
        >>> result['label']  # "Inconsistent"
    """
    global _factcc_pipeline

    # Validate required parameters
    if source is None:
        return {
            'score': None,
            'error': 'Source document is required for FactCC evaluation'
        }

    try:
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
            pipeline
        )

        # Load FactCC model (cached globally)
        # Uses deberta-base-mnli as alternative to original FactCC checkpoint
        if _factcc_pipeline is None:
            model_name = "microsoft/deberta-base-mnli"
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForSequenceClassification.from_pretrained(model_name)
            _factcc_pipeline = pipeline(
                "text-classification",
                model=model,
                tokenizer=tokenizer,
                device=-1  # Force CPU
            )

        classifier = _factcc_pipeline

        # Truncate long texts
        max_words = 400
        source_truncated = ' '.join(source.split()[:max_words])
        summary_truncated = ' '.join(summary.split()[:max_words])

        # Format for consistency checking
        text_input = f"{source_truncated} [SEP] {summary_truncated}"

        # Get prediction
        result = classifier(text_input, truncation=True, max_length=512)

        label = result[0]['label']
        score = result[0]['score']

        # Map to consistency score (ENTAILMENT = consistent)
        if 'ENTAIL' in label.upper() or 'CONSISTENT' in label.upper():
            consistency_score = score
            is_consistent = True
        else:
            consistency_score = 1 - score
            is_consistent = False

        return {
            'score': round(consistency_score, 4),
            'label': 'Consistent' if is_consistent else 'Inconsistent',
            'raw_label': label,
            'interpretation': _interpret_factcc_score(consistency_score),
            'error': None
        }

    except Exception as e:
        return {
            'score': None,
            'error': str(e)
        }


def _interpret_factcc_score(score: float) -> str:
    """Interpret FactCC score."""
    if score >= 0.8:
        return "Highly Consistent"
    elif score >= 0.6:
        return "Mostly Consistent"
    elif score >= 0.4:
        return "Partially Consistent"
    else:
        return "Inconsistent"


# Global AlignScore model cache to avoid reloading
_alignscore_model = None
_alignscore_tokenizer = None


def compute_alignscore(
    summary: str,
    source: str = None,
    reference_summary: str = None,
    model_name: str = "liuyanyi/AlignScore-large-hf"
) -> Dict:
    """
    Evaluate factual consistency using state-of-the-art unified alignment model (RECOMMENDED).

    This metric answers: "How well does the summary align factually with the source across multiple
    semantic dimensions?" Uses RoBERTa-large fine-tuned on 7 diverse alignment tasks (entailment,
    paraphrase, fact verification, etc.). Currently the most reliable single metric for factual accuracy.

    Use this when: You want the most robust and comprehensive factual consistency check with a single
    metric. This is the recommended metric for production fact-checking.

    Paper: "AlignScore: Evaluating Factual Consistency with a Unified Alignment Function"
    Model: https://huggingface.co/liuyanyi/AlignScore-large-hf

    Args:
        summary (str): Generated summary (claim) to fact-check
        source (str, optional): Original source document (context/premise) to verify against
        reference_summary (str, optional): Not used for AlignScore (kept for API consistency)
        model_name (str, optional): HuggingFace model. Default "liuyanyi/AlignScore-large-hf" (~1.3GB)

    Returns:
        Dict: Result dictionary with keys:
            - score (float): Alignment score from 0.0 to 1.0 (higher = more factually consistent)
            - interpretation (str): Human-readable label like "Fully Consistent" or "Partially Consistent"
            - error (str, optional): Error message if model not available

    Example:
        >>> result = compute_alignscore(
        ...     summary="Paris is France's capital city.",
        ...     source="Paris is the capital of France."
        ... )
        >>> result['score']  # e.g., 0.95 (very high factual alignment)
        >>> result['interpretation']  # "Fully Consistent"
    """
    global _alignscore_model, _alignscore_tokenizer

    # Validate required parameters
    if source is None:
        return {
            'score': None,
            'error': 'Source document is required for AlignScore evaluation'
        }

    try:
        import torch
        from transformers import AutoModelForSequenceClassification, RobertaTokenizer

        # Load model and tokenizer if not cached
        if _alignscore_model is None:
            _alignscore_tokenizer = RobertaTokenizer.from_pretrained('roberta-large')
            _alignscore_model = AutoModelForSequenceClassification.from_pretrained(
                model_name,
                trust_remote_code=True
            )
            _alignscore_model.eval()

        # Tokenize inputs
        inputs = _alignscore_tokenizer(
            source,
            summary,
            return_tensors='pt',
            truncation=True,
            max_length=512
        )

        # Get score
        with torch.no_grad():
            outputs = _alignscore_model(**inputs)
            score = outputs.reg_label_logits.item()

        return {
            'score': round(float(score), 4),
            'interpretation': _interpret_alignscore(float(score)),
            'error': None
        }

    except ImportError as e:
        return {
            'score': None,
            'error': f'Required packages not installed. Error: {str(e)}'
        }
    except Exception as e:
        return {
            'score': None,
            'error': str(e)
        }


def _interpret_alignscore(score: float) -> str:
    """Interpret AlignScore for human readability."""
    if score >= 0.9:
        return "Fully Consistent"
    elif score >= 0.7:
        return "Highly Consistent"
    elif score >= 0.5:
        return "Mostly Consistent"
    elif score >= 0.3:
        return "Partially Consistent"
    else:
        return "Inconsistent"


def compute_coverage_score(
    summary: str,
    source: str = None,
    reference_summary: str = None
) -> Dict:
    """
    Calculate what percentage of named entities from source appear in summary using NER.

    This metric answers: "Did the summary mention the key people, places, organizations, and dates?"
    Extracts named entities (PERSON, ORG, GPE, DATE, etc.) using spaCy and counts how many from
    the source are present in the summary. Score of 0.7+ means most key entities captured.

    Use this when: You want to ensure summaries preserve important factual details like names,
    locations, organizations, and dates. Good for news, reports, and fact-heavy documents.

    Args:
        summary (str): Generated summary to check for entity coverage
        source (str, optional): Original source document containing entities to capture
        reference_summary (str, optional): Not used for Coverage (kept for API consistency)

    Returns:
        Dict: Result dictionary with keys:
            - score (float): Entity coverage ratio from 0.0 to 1.0 (e.g., 0.75 = 75% of entities present)
            - source_entities (int): Total number of unique entities in source
            - covered_entities (int): Number of source entities found in summary
            - missing_entities (list): Up to 5 example entities not captured in summary
            - interpretation (str): Human-readable label like "Good Coverage" or "Low Coverage"
            - error (str, optional): Error message if spaCy not installed

    Example:
        >>> result = compute_coverage_score(
        ...     summary="John visited Paris.",
        ...     source="John Smith visited Paris in 2023."
        ... )
        >>> result['score']  # e.g., 0.67 (2 of 3 entities: John, Paris present; 2023 missing)
        >>> result['missing_entities']  # ['2023']
    """
    # Validate required parameters
    if source is None:
        return {
            'score': None,
            'error': 'Source document is required for Coverage evaluation'
        }

    try:
        import spacy

        # Load spaCy English model
        try:
            nlp = spacy.load("en_core_web_sm")
        except OSError:
            # Model not installed
            return {
                'score': None,
                'error': 'spaCy model not installed. Run: python -m spacy download en_core_web_sm'
            }

        # Extract entities from source
        source_doc = nlp(source)
        source_entities = set()
        for ent in source_doc.ents:
            source_entities.add(ent.text.lower())

        if not source_entities:
            return {
                'score': 1.0,  # If no entities in source, summary trivially covers all
                'source_entities': 0,
                'covered_entities': 0,
                'interpretation': 'No named entities in source',
                'error': None
            }

        # Extract entities from summary
        summary_doc = nlp(summary)
        summary_entities = set()
        for ent in summary_doc.ents:
            summary_entities.add(ent.text.lower())

        # Calculate coverage
        covered = source_entities.intersection(summary_entities)
        coverage_score = len(covered) / len(source_entities)

        return {
            'score': round(coverage_score, 4),
            'source_entities': len(source_entities),
            'covered_entities': len(covered),
            'missing_entities': list(source_entities - summary_entities)[:5],  # Show up to 5 missing
            'interpretation': _interpret_coverage_score(coverage_score),
            'error': None
        }

    except ImportError:
        return {
            'score': None,
            'error': 'spaCy not installed. Run: pip install spacy'
        }
    except Exception as e:
        return {
            'score': None,
            'error': str(e)
        }


def _interpret_coverage_score(score: float) -> str:
    """Interpret Coverage Score for human readability."""
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


def compute_all_era3_metrics(
    summary: str,
    source: str = None,
    reference_summary: str = None,
    use_factchecker: bool = False,
    use_factcc: bool = False,
    use_alignscore: bool = False,
    use_coverage: bool = False,
    use_unieval: bool = False,
    factchecker_model: Optional[str] = None
) -> Dict[str, Dict[str, float]]:
    """
    Run all available faithfulness and factual consistency metrics to detect hallucinations and errors.

    This function computes multiple metrics that answer: "Is the summary factually accurate and supported
    by the source?" Uses various approaches: NLI models, specialized fact-checking models, entity coverage,
    and optional API-based claim verification. NLI is always included; others are opt-in.

    Use this when: You want comprehensive faithfulness checking with multiple complementary approaches
    to detect different types of factual errors (hallucinations, contradictions, unsupported claims).

    Args:
        summary (str): Generated summary to fact-check
        source (str, optional): Original source document to verify summary against
        reference_summary (str, optional): Not used for Era3 metrics (kept for API consistency)
        use_factchecker (bool, optional): Enable API-based LLM fact-checker (slow, requires API). Default False
        use_factcc (bool, optional): Enable FactCC BERT model (~600MB). Default False
        use_alignscore (bool, optional): Enable AlignScore unified model (~1.3GB, RECOMMENDED). Default False
        use_coverage (bool, optional): Enable NER entity coverage check (requires spaCy). Default False
        use_unieval (bool, optional): Enable UniEval multi-dimensional scorer. Default False
        factchecker_model (str, optional): LLM model name for API fact-checker. Default None (uses Llama-3.3-70B)

    Returns:
        Dict[str, Dict]: Dictionary mapping metric names to their results:
            - 'NLI': Always included - DeBERTa-v3 entailment check
            - 'FactCC': If use_factcc=True - BERT-based consistency
            - 'AlignScore': If use_alignscore=True - Unified alignment (RECOMMENDED)
            - 'Coverage': If use_coverage=True - Named entity coverage
            - 'UniEval': If use_unieval=True - Multi-dimensional evaluation
            - 'FactChecker': If use_factchecker=True - API-based claim verification
            Each value is a dict with score, interpretation, and possibly error keys.

    Example:
        >>> results = compute_all_era3_metrics(
        ...     summary="Paris is French capital.",
        ...     source="Paris is capital of France.",
        ...     use_alignscore=True,
        ...     use_coverage=True
        ... )
        >>> results['NLI']['nli_score']  # e.g., 0.91
        >>> results['AlignScore']['score']  # e.g., 0.94
        >>> list(results.keys())  # ['NLI', 'AlignScore', 'Coverage']
    """
    # Validate required parameters
    if source is None:
        return {
            'error': 'Source document is required for Era3 metrics'
        }

    results = {
        'NLI': compute_nli_score(summary=summary, source=source)
    }

    # Add FactCC if enabled
    if use_factcc:
        results['FactCC'] = compute_factcc_score(summary=summary, source=source)

    # Add AlignScore if enabled
    if use_alignscore:
        results['AlignScore'] = compute_alignscore(summary=summary, source=source)

    # Add Coverage Score if enabled (for completeness check)
    if use_coverage:
        results['Coverage'] = compute_coverage_score(summary=summary, source=source)

    # Add UniEval if enabled (BLEURT backup - multi-dimensional evaluation)
    if use_unieval:
        from src.evaluators.era3_unieval import compute_unieval
        results['UniEval'] = compute_unieval(summary=summary, source=source)

    # Add API-based fact-checker if enabled
    if use_factchecker:
        results['FactChecker'] = compute_factchecker_score(
            summary=summary,
            source=source,
            model_name=factchecker_model,
            use_api=True
        )

    return results
