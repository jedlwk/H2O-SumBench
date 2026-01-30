#!/usr/bin/env python3
"""
Era 3 Group B: LLM-as-a-Judge Evaluators (API-based)
Uses H2OGPTE API to evaluate summaries using powerful LLMs.

Implements G-Eval, DAG and Prometheus.
"""

import os
from typing import Dict, Any, Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Module-level configuration
_API_KEY = os.getenv('H2OGPTE_API_KEY')
_ADDRESS = os.getenv('H2OGPTE_ADDRESS')
_client = None


def get_client():
    """
    Get or create the H2OGPTE client.
    Lazy loads the client on first use.

    Returns:
        H2OGPTE client instance

    Raises:
        ValueError: If API credentials are not set
    """
    global _client

    if not _API_KEY or not _ADDRESS:
        raise ValueError(
            "H2OGPTE_API_KEY and H2OGPTE_ADDRESS must be set in .env file"
        )

    if _client is None:
        from h2ogpte import H2OGPTE
        _client = H2OGPTE(
            address=_ADDRESS,
            api_key=_API_KEY,
        )
    return _client


def create_faithfulness_prompt(source: str, summary: str) -> str:
    """
    Generate a structured G-Eval prompt to evaluate if summary claims are supported by the source.

    This function creates a prompt that asks an LLM to verify every claim in the summary
    against the source document, identifying hallucinations, contradictions, or unsupported
    statements. The prompt follows the G-Eval methodology with a 1-10 scoring rubric.

    Use this when: You need to construct a faithfulness evaluation prompt for an LLM judge.
    Internal helper function for evaluate_faithfulness().

    Args:
        source (str): Original source document text that serves as ground truth
        summary (str): Generated summary to fact-check against the source

    Returns:
        str: Formatted prompt string ready for LLM consumption with:
            - Task definition and scoring criteria (1-10 scale)
            - Source document and summary to evaluate
            - Instructions for systematic claim verification
            - Required response format (Score + Explanation)

    Example:
        >>> prompt = create_faithfulness_prompt("Paris is in France.", "Paris is French capital.")
        >>> "FAITHFULNESS" in prompt  # True
        >>> "Score:" in prompt  # True
    """
    prompt = f"""
    You are an expert evaluator for text summarization. Your task is to evaluate the FAITHFULNESS of a summary.

    **Faithfulness Definition**: A summary is faithful if all claims and facts in it are directly supported by the source document. Faithful summaries contain no hallucinations, contradictions, or unsupported claims.

    **Evaluation Criteria**:
    - Score 1-2: Multiple unsupported claims or contradictions
    - Score 3-4: Some claims not fully supported by source
    - Score 5-6: Mostly faithful with minor issues
    - Score 7-8: Highly faithful, all major claims supported
    - Score 9-10: Perfect faithfulness, every claim verifiable

    **Source Document**:
    {source}

    **Summary to Evaluate**:
    {summary}

    **Instructions**:
    1. Check each claim in the summary against the source
    2. Identify any unsupported claims or contradictions
    3. Rate faithfulness from 1-10
    4. Provide a brief explanation (1-2 sentences)

    **Response Format**:
    Score: [1-10]
    Explanation: [Your reasoning]
    """
    return prompt


def create_coherence_prompt(summary: str) -> str:
    """
    Generate a structured G-Eval prompt to evaluate logical flow and structure of the summary.

    This function creates a prompt that asks an LLM to assess how well the summary flows,
    checking for logical connections between sentences, clear topic progression, and overall
    structural quality. Uses a 1-10 scoring rubric from the G-Eval framework.

    Use this when: You need to construct a coherence evaluation prompt for an LLM judge.
    Internal helper function for evaluate_coherence().

    Args:
        summary (str): Generated summary text to evaluate for coherence

    Returns:
        str: Formatted prompt string ready for LLM consumption with:
            - Coherence definition and scoring criteria (1-10 scale)
            - Summary text to evaluate
            - Instructions to assess logical flow and structure
            - Required response format (Score + Explanation)

    Example:
        >>> prompt = create_coherence_prompt("First sentence. Second sentence.")
        >>> "COHERENCE" in prompt  # True
        >>> "logical flow" in prompt.lower()  # True
    """
    prompt = f"""
    You are an expert evaluator for text summarization. Your task is to evaluate the COHERENCE of a summary.

    **Coherence Definition**: A coherent summary flows logically, with clear connections between sentences and ideas. It is well-structured and easy to follow.

    **Evaluation Criteria**:
    - Score 1-2: Disjointed, confusing structure
    - Score 3-4: Some logical flow issues
    - Score 5-6: Adequate flow, minor gaps
    - Score 7-8: Good logical structure
    - Score 9-10: Excellent flow, perfectly structured

    **Summary to Evaluate**:
    {summary}

    **Instructions**:
    1. Assess the logical flow between sentences
    2. Check for clear topic progression
    3. Rate coherence from 1-10
    4. Provide a brief explanation (1-2 sentences)

    **Response Format**:
    Score: [1-10]
    Explanation: [Your reasoning]
    """
    return prompt


def create_relevance_prompt(source: str, summary: str) -> str:
    """
    Generate a structured G-Eval prompt to evaluate if summary captures key information from source.

    This function creates a prompt that asks an LLM to check whether the summary includes the
    most important points from the source while avoiding trivial or off-topic details. Evaluates
    completeness and information selection quality on a 1-10 scale.

    Use this when: You need to construct a relevance evaluation prompt for an LLM judge.
    Internal helper function for evaluate_relevance().

    Args:
        source (str): Original source document containing the key information
        summary (str): Generated summary to evaluate for relevance

    Returns:
        str: Formatted prompt string ready for LLM consumption with:
            - Relevance definition and scoring criteria (1-10 scale)
            - Source document and summary to evaluate
            - Instructions to identify key points and check coverage
            - Required response format (Score + Explanation)

    Example:
        >>> prompt = create_relevance_prompt("Main point here.", "Summary text.")
        >>> "RELEVANCE" in prompt  # True
        >>> "important information" in prompt.lower()  # True
    """
    prompt = f"""
    You are an expert evaluator for text summarization. Your task is to evaluate the RELEVANCE of a summary.

    **Relevance Definition**: A relevant summary captures the most important information from the source document without including trivial or off-topic details.

    **Evaluation Criteria**:
    - Score 1-2: Misses main points, includes irrelevant info
    - Score 3-4: Captures some key points, has extraneous content
    - Score 5-6: Covers main points adequately
    - Score 7-8: Captures all important information
    - Score 9-10: Perfect selection of key information

    **Source Document**:
    {source}

    **Summary to Evaluate**:
    {summary}

    **Instructions**:
    1. Identify the main points in the source
    2. Check if the summary captures them
    3. Rate relevance from 1-10
    4. Provide a brief explanation (1-2 sentences)

    **Response Format**:
    Score: [1-10]
    Explanation: [Your reasoning]
    """
    return prompt


def create_fluency_prompt(summary: str) -> str:
    """
    Generate a structured G-Eval prompt to evaluate grammatical correctness and writing quality.

    This function creates a prompt that asks an LLM to assess the summary's grammar, naturalness
    of language, and overall readability. Checks for errors, awkward phrasing, and writing quality
    on a 1-10 scale from the G-Eval framework.

    Use this when: You need to construct a fluency evaluation prompt for an LLM judge.
    Internal helper function for evaluate_fluency().

    Args:
        summary (str): Generated summary text to evaluate for fluency

    Returns:
        str: Formatted prompt string ready for LLM consumption with:
            - Fluency definition and scoring criteria (1-10 scale)
            - Summary text to evaluate
            - Instructions to check grammar, naturalness, and readability
            - Required response format (Score + Explanation)

    Example:
        >>> prompt = create_fluency_prompt("Well written summary here.")
        >>> "FLUENCY" in prompt  # True
        >>> "grammatically correct" in prompt.lower()  # True
    """
    prompt = f"""
    You are an expert evaluator for text summarization. Your task is to evaluate the FLUENCY of a summary.

    **Fluency Definition**: A fluent summary is grammatically correct, uses natural language, and is easy to read without awkward phrasing.

    **Evaluation Criteria**:
    - Score 1-2: Multiple grammar errors, unnatural phrasing
    - Score 3-4: Several fluency issues
    - Score 5-6: Generally fluent with minor issues
    - Score 7-8: Highly fluent, natural language
    - Score 9-10: Perfect fluency, publication-quality

    **Summary to Evaluate**:
    {summary}

    **Instructions**:
    1. Check for grammatical correctness
    2. Assess naturalness of language
    3. Rate fluency from 1-10
    4. Provide a brief explanation (1-2 sentences)

    **Response Format**:
    Score: [1-10]
    Explanation: [Your reasoning]
    """
    return prompt


def parse_llm_response(response: str) -> tuple[Optional[float], Optional[str]]:
    """
    Extract numerical score and explanation text from LLM's structured response.

    This function parses LLM responses that follow the format "Score: X\nExplanation: Y"
    or similar patterns ([RESULT], Feedback:). Handles various formatting variations and
    extracts the numeric score and reasoning text.

    Use this when: You need to parse structured output from LLM evaluation responses.
    Internal helper function used by all evaluate_* functions.

    Args:
        response (str): Raw text response from the LLM containing score and explanation

    Returns:
        tuple[Optional[float], Optional[str]]: A tuple containing:
            - score (float or None): Extracted numeric score, or None if parsing failed
            - explanation (str or None): Extracted explanation text, or None if not found

    Example:
        >>> response = "Score: 8\nExplanation: Good quality summary"
        >>> score, explanation = parse_llm_response(response)
        >>> score  # 8.0
        >>> explanation  # "Good quality summary"
    """
    try:
        lines = response.strip().split('\n')
        score = None
        explanation = None

        for line in lines:
            line = line.strip()
            if line.startswith('Score:') or line.startswith('[RESULT]'):
                # Clean the line: Remove tags and brackets, then convert
                score_text = line.replace('Score:', '').replace('[RESULT]', '')
                score_text = score_text.replace('[', '').replace(']', '').strip()
                if score_text:
                    score = float(score_text)
            elif line.startswith('Explanation:') or line.startswith('Feedback:'):
                explanation = line.replace('Explanation:', '').replace('Feedback:', '').strip()

        return score, explanation
    except Exception as e:
        print(f"Error parsing LLM response: {e}")
        return None, None


def query_llm(
    prompt: str,
    model_name: str = 'meta-llama/Llama-3.3-70B-Instruct',
    timeout: int = 60
) -> str:
    """
    Send a prompt to the H2OGPTE API and return the LLM's response text.

    This function handles the API communication by creating a chat session, sending the prompt,
    and retrieving the model's response. Used as the core communication layer for all LLM-based
    evaluation functions. Requires H2OGPTE_API_KEY and H2OGPTE_ADDRESS environment variables.

    Use this when: You need to query the H2OGPTE LLM for evaluation tasks.
    Internal helper function used by all evaluate_* functions.

    Args:
        prompt (str): The evaluation prompt to send to the LLM (typically from create_*_prompt functions)
        model_name (str, optional): H2OGPTE model identifier. Default "meta-llama/Llama-3.3-70B-Instruct".
                                    Other options include GPT-4, Claude, or other supported models.
        timeout (int, optional): Maximum time to wait for response in seconds. Default 60.

    Returns:
        str: The LLM's complete response text (unparsed)

    Raises:
        ValueError: If H2OGPTE API credentials are not configured
        Exception: If API call fails or times out

    Example:
        >>> prompt = "Evaluate this summary..."
        >>> response = query_llm(prompt, timeout=90)
        >>> "Score:" in response  # True (if prompt requested a score)
    """
    client = get_client()
    chat_session_id = client.create_chat_session()

    with client.connect(chat_session_id) as session:
        reply = session.query(
            prompt,
            llm=model_name,
            timeout=timeout,
        )
        return reply.content


def evaluate_faithfulness(
    summary: str,
    source: str = None,
    reference_summary: str = None,
    model_name: str = 'meta-llama/Llama-3.3-70B-Instruct',
    timeout: int = 60
) -> Dict[str, Any]:
    """
    Use a large language model to evaluate if summary claims are supported by source (G-Eval framework).

    This metric answers: "Are all facts in the summary directly supported by the source?"
    An LLM reads both texts and judges faithfulness on a 1-10 scale with explanation.
    Detects hallucinations, contradictions, and unsupported claims. No token limits unlike local models.

    Use this when: You want human-like, nuanced fact-checking without token length restrictions.
    Slower than local models but more thorough. Requires H2OGPTE API.

    Args:
        summary (str): Generated summary text to evaluate
        source (str, optional): Source document text to compare against
        reference_summary (str, optional): Reference summary that represents ideal quality
        model_name (str, optional): H2OGPTE LLM model. Default "meta-llama/Llama-3.3-70B-Instruct"
        timeout (int, optional): API timeout in seconds. Default 60

    Returns:
        Dict[str, Any]: Result dictionary with keys:
            - score (float): Normalized faithfulness score from 0.0 to 1.0
            - raw_score (float): Original 1-10 scale score
            - explanation (str): LLM's reasoning for the score
            - full_response (str): Complete LLM response for debugging
            - error (str, optional): Error message if API call failed

    Example:
        >>> result = evaluate_faithfulness(
        ...     summary="Paris is French capital.",
        ...     source="Paris is capital."
        ... )
        >>> result['raw_score']  # e.g., 9 (highly faithful)
        >>> result['explanation']  # "All claims directly supported by source"
    """
    try:
        # Validate required parameters
        if source is None:
            return {
                'score': None,
                'error': 'Source document is required for faithfulness evaluation'
            }

        prompt = create_faithfulness_prompt(source, summary)
        response = query_llm(prompt, model_name, timeout)
        score, explanation = parse_llm_response(response)

        if score is not None:
            # Normalize to 0-1 scale
            normalized_score = score / 10.0
            return {
                'score': normalized_score,
                'raw_score': score,
                'explanation': explanation or 'No explanation provided',
                'full_response': response,
                'error': None
            }
        else:
            return {
                'score': None,
                'error': 'Failed to parse score from LLM response',
                'full_response': response,
            }
    except Exception as e:
        return {
            'score': None,
            'error': str(e),
        }


def evaluate_coherence(
    summary: str,
    source: str = None,
    reference_summary: str = None,
    model_name: str = 'meta-llama/Llama-3.3-70B-Instruct',
    timeout: int = 60
) -> Dict[str, Any]:
    """
    Use a large language model to evaluate if the summary flows logically with clear structure (G-Eval framework).

    This metric answers: "Does the summary have good logical flow and clear connections between ideas?"
    An LLM reads the summary and judges coherence on a 1-10 scale, checking for logical progression,
    clear topic transitions, and overall structural quality. Helps identify disjointed or confusing text.

    Use this when: You want human-like assessment of text structure and readability flow.
    Requires H2OGPTE API access. No token limits unlike local coherence models.

    Args:
        summary (str): Generated summary text to evaluate
        source (str, optional): Source document text to compare against
        reference_summary (str, optional): Reference summary that represents ideal quality
        model_name (str, optional): H2OGPTE LLM model. Default "meta-llama/Llama-3.3-70B-Instruct"
        timeout (int, optional): API timeout in seconds. Default 60

    Returns:
        Dict[str, Any]: Result dictionary with keys:
            - score (float): Normalized coherence score from 0.0 to 1.0 (higher = better flow)
            - raw_score (float): Original 1-10 scale score from LLM
            - explanation (str): LLM's reasoning about the logical structure
            - full_response (str): Complete LLM response for debugging
            - error (str, optional): Error message if API call failed

    Example:
        >>> result = evaluate_coherence("First point. Second point connects logically.")
        >>> result['raw_score']  # e.g., 8 (good coherence)
        >>> result['explanation']  # "Clear logical progression between ideas"
    """
    try:
        prompt = create_coherence_prompt(summary)
        response = query_llm(prompt, model_name, timeout)
        score, explanation = parse_llm_response(response)

        if score is not None:
            normalized_score = score / 10.0
            return {
                'score': normalized_score,
                'raw_score': score,
                'explanation': explanation or 'No explanation provided',
                'full_response': response,
                'error': None
            }
        else:
            return {
                'score': None,
                'error': 'Failed to parse score from LLM response',
                'full_response': response,
            }
    except Exception as e:
        return {
            'score': None,
            'error': str(e),
        }


def evaluate_relevance(
    summary: str,
    source: str = None,
    reference_summary: str = None,
    model_name: str = 'meta-llama/Llama-3.3-70B-Instruct',
    timeout: int = 60
) -> Dict[str, Any]:
    """
    Use a large language model to evaluate if summary captures key source information (G-Eval framework).

    This metric answers: "Does the summary include the most important points from the source?"
    An LLM reads both texts and judges relevance on a 1-10 scale, checking if key information is
    captured while avoiding trivial details. Measures completeness and information selection quality.

    Use this when: You want human-like assessment of whether summary captures salient points.
    Requires H2OGPTE API access. Can handle full-length documents unlike token-limited local models.

    Args:
        summary (str): Generated summary text to evaluate
        source (str, optional): Source document text to compare against
        reference_summary (str, optional): Reference summary that represents ideal quality
        model_name (str, optional): H2OGPTE LLM model. Default "meta-llama/Llama-3.3-70B-Instruct"
        timeout (int, optional): API timeout in seconds. Default 60

    Returns:
        Dict[str, Any]: Result dictionary with keys:
            - score (float): Normalized relevance score from 0.0 to 1.0 (higher = better coverage)
            - raw_score (float): Original 1-10 scale score from LLM
            - explanation (str): LLM's reasoning about information selection
            - full_response (str): Complete LLM response for debugging
            - error (str, optional): Error message if API call failed

    Example:
        >>> result = evaluate_relevance(
        ...     summary="Summary of point A.",
        ...     source="Main point A. Detail B."
        ... )
        >>> result['raw_score']  # e.g., 7 (good but missing detail B)
        >>> result['explanation']  # "Captures main point but omits some details"
    """
    try:
        # Validate required parameters
        if source is None:
            return {
                'score': None,
                'error': 'Source document is required for relevance evaluation'
            }

        prompt = create_relevance_prompt(source, summary)
        response = query_llm(prompt, model_name, timeout)
        score, explanation = parse_llm_response(response)

        if score is not None:
            normalized_score = score / 10.0
            return {
                'score': normalized_score,
                'raw_score': score,
                'explanation': explanation or 'No explanation provided',
                'full_response': response,
                'error': None
            }
        else:
            return {
                'score': None,
                'error': 'Failed to parse score from LLM response',
                'full_response': response,
            }
    except Exception as e:
        return {
            'score': None,
            'error': str(e),
        }


def evaluate_fluency(
    summary: str,
    source: str = None,
    reference_summary: str = None,
    model_name: str = 'meta-llama/Llama-3.3-70B-Instruct',
    timeout: int = 60
) -> Dict[str, Any]:
    """
    Use a large language model to evaluate grammatical correctness and writing quality (G-Eval framework).

    This metric answers: "Is the summary well-written with natural, grammatically correct language?"
    An LLM reads the summary and judges fluency on a 1-10 scale, checking for grammar errors,
    awkward phrasing, and overall readability. Helps identify poorly written or unnatural text.

    Use this when: You want human-like assessment of writing quality and grammatical correctness.
    Requires H2OGPTE API access. More nuanced than simple grammar checkers.

    Args:
        summary (str): Generated summary text to evaluate
        source (str, optional): Source document text to compare against
        reference_summary (str, optional): Reference summary that represents ideal quality
        model_name (str, optional): H2OGPTE LLM model. Default "meta-llama/Llama-3.3-70B-Instruct"
        timeout (int, optional): API timeout in seconds. Default 60

    Returns:
        Dict[str, Any]: Result dictionary with keys:
            - score (float): Normalized fluency score from 0.0 to 1.0 (higher = better writing)
            - raw_score (float): Original 1-10 scale score from LLM
            - explanation (str): LLM's reasoning about grammar and naturalness
            - full_response (str): Complete LLM response for debugging
            - error (str, optional): Error message if API call failed

    Example:
        >>> result = evaluate_fluency("This summary are well written and clear.")
        >>> result['raw_score']  # e.g., 4 (grammar error: "are" should be "is")
        >>> result['explanation']  # "Subject-verb agreement error detected"
    """
    try:
        prompt = create_fluency_prompt(summary)
        response = query_llm(prompt, model_name, timeout)
        score, explanation = parse_llm_response(response)

        if score is not None:
            normalized_score = score / 10.0
            return {
                'score': normalized_score,
                'raw_score': score,
                'explanation': explanation or 'No explanation provided',
                'full_response': response,
                'error': None
            }
        else:
            return {
                'score': None,
                'error': 'Failed to parse score from LLM response',
                'full_response': response,
            }
    except Exception as e:
        return {
            'score': None,
            'error': str(e),
        }


def evaluate_dag(
    summary: str,
    source: str = None,
    reference_summary: str = None,
    model_name: str = 'meta-llama/Llama-3.3-70B-Instruct',
    timeout: int = 60
) -> Dict[str, Any]:
    """
    Use a large language model with a decision tree approach to evaluate summary quality holistically (DAG framework).

    This metric answers: "Is the summary factually accurate, complete, and clear?" using a structured
    3-step decision tree instead of a single score. DAG (Directed Acyclic Graph) from DeepEval breaks
    evaluation into: Step 1 (Factual Accuracy 0-2), Step 2 (Completeness 0-2), Step 3 (Clarity 0-2).
    Total score 0-6 captures multiple quality dimensions systematically.

    Use this when: You want a structured, multi-dimensional evaluation instead of a single holistic score.
    Provides more granular feedback than single-score metrics. Requires H2OGPTE API access.

    Args:
        summary (str): Generated summary text to evaluate
        source (str, optional): Source document text to compare against
        reference_summary (str, optional): Reference summary that represents ideal quality
        model_name (str, optional): H2OGPTE LLM model. Default "meta-llama/Llama-3.3-70B-Instruct"
        timeout (int, optional): API timeout in seconds. Default 60

    Returns:
        Dict[str, Any]: Result dictionary with keys:
            - score (float): Normalized total score from 0.0 to 1.0 (sum of 3 steps / 6)
            - raw_score (int): Original total score from 0 to 6 (sum of all steps)
            - step1_factual (int): Factual accuracy subscore (0=no, 1=mostly, 2=yes)
            - step2_completeness (int): Completeness subscore (0=no, 1=mostly, 2=yes)
            - step3_clarity (int): Clarity subscore (0=no, 1=mostly, 2=yes)
            - explanation (str): LLM's reasoning for the decision path taken
            - full_response (str): Complete LLM response for debugging
            - error (str, optional): Error message if API call failed

    Example:
        >>> result = evaluate_dag(
        ...     summary="Paris is French capital.",
        ...     source="Paris is in France."
        ... )
        >>> result['raw_score']  # e.g., 5 (out of 6)
        >>> result['step1_factual']  # e.g., 2 (yes, factually correct)
        >>> result['step2_completeness']  # e.g., 2 (yes, covers main point)
        >>> result['step3_clarity']  # e.g., 1 (mostly clear)
    """
    try:
        # Validate required parameters
        if source is None:
            return {
                'score': None,
                'error': 'Source document is required for DAG evaluation'
            }

        prompt = f"""
        You are an expert evaluator using a structured decision tree approach (DAG - Directed Acyclic Graph).

        **Evaluation Task**: Assess this summary using a step-by-step decision tree.

        **Source Document**:
        {source}

        **Summary to Evaluate**:
        {summary}

        **Decision Tree Evaluation**:

        Step 1: FACTUAL ACCURACY
        Question: Are all facts in the summary correct?
        - If YES → Score 2 → Go to Step 2
        - If MOSTLY YES → Score 1 → Go to Step 2
        - If NO → Score 0 → STOP (Final: 0-2)

        Step 2: COMPLETENESS
        Question: Does the summary cover the main points?
        - If YES → Score 2 → Go to Step 3
        - If MOSTLY YES → Score 1 → Go to Step 3
        - If NO → Score 0 → Add to Step 1 score

        Step 3: CLARITY
        Question: Is the summary clear and well-written?
        - If YES → Score 2 → Calculate final
        - If MOSTLY YES → Score 1 → Calculate final
        - If NO → Score 0 → Calculate final

        **Response Format**:
        Step 1 Score: [0/1/2]
        Step 2 Score: [0/1/2]
        Step 3 Score: [0/1/2]
        Total Score: [0-6]
        Explanation: [Brief explanation of the decision path taken]
        """

        response = query_llm(prompt, model_name, timeout)

        # Parse response
        step1 = None
        step2 = None
        step3 = None
        total = None
        explanation = ""

        for line in response.split('\n'):
            line = line.strip()
            if line.startswith('Step 1 Score:'):
                try:
                    step1 = int(line.split(':')[1].strip().split('/')[0])
                except:
                    pass
            elif line.startswith('Step 2 Score:'):
                try:
                    step2 = int(line.split(':')[1].strip().split('/')[0])
                except:
                    pass
            elif line.startswith('Step 3 Score:'):
                try:
                    step3 = int(line.split(':')[1].strip().split('/')[0])
                except:
                    pass
            elif line.startswith('Total Score:'):
                try:
                    score_text = line.split(':')[1].strip().split('-')[0].strip()
                    total = int(score_text)
                except:
                    pass
            elif line.startswith('Explanation:'):
                explanation = line.split(':', 1)[1].strip()

        if total is not None:
            normalized_score = total / 6.0  # Normalize to 0-1
        else:
            normalized_score = None

        return {
            'score': normalized_score,
            'raw_score': total,
            'step1_factual': step1,
            'step2_completeness': step2,
            'step3_clarity': step3,
            'explanation': explanation if explanation else 'No explanation provided',
            'full_response': response,
            'error': None
        }

    except Exception as e:
        return {
            'score': None,
            'error': str(e)
        }


def create_prometheus_absolute_prompt() -> str:
    """
    Generate the Prometheus framework prompt template for absolute grading against a reference.

    This function creates the structured prompt format used by the Prometheus evaluation framework,
    which compares a generated response against a reference answer using a detailed rubric. The
    template includes placeholders for instruction, response, reference answer, and scoring rubric.

    Use this when: You need to construct a Prometheus evaluation prompt template.
    Internal helper function for evaluate_prometheus().

    Returns:
        str: Prometheus prompt template string with placeholders:
            - {instruction}: The task description
            - {response}: The generated summary to evaluate
            - {reference_answer}: The reference summary (score 5 benchmark)
            - {rubric}: The evaluation criteria/rubric

    Example:
        >>> template = create_prometheus_absolute_prompt()
        >>> "{instruction}" in template  # True
        >>> "Score Rubrics:" in template  # True
    """
    prompt = """
    ###Task Description:
    An instruction (might include an Input inside it), a response to evaluate, a reference answer that gets a score of 5, and a score rubric representing a evaluation criteria are given.
    1. Write a detailed explanation that assess the quality of the response strictly based on the given score rubric, not evaluating in general.
    2. After writing a explanation, write a score that is an integer between 1 and 5. You should refer to the score rubric.
    3. The output format should look as follows:
        "Explanation: (write the evaluation here)
        Score: (an integer number between 1 and 5)"
    4. Please do not generate any other opening, closing, and explanations.

    ###The instruction to evaluate:
    {instruction}

    ###Response to evaluate:
    {response}

    ###Reference Answer (Score 5):
    {reference_answer}

    ###Score Rubrics:
    {rubric}

    ###Explanation:
    """
    return prompt


def evaluate_prometheus(
    summary: str,
    source: str = None,
    reference_summary: str = None,
    model_name: str = 'meta-llama/Llama-3.3-70B-Instruct',
    timeout: int = 60
) -> Dict[str, Any]:
    """
    Use a large language model to evaluate summary quality against a reference using Prometheus framework.

    This metric answers: "How well does this summary compare to a reference summary in information density?"
    An LLM compares the generated summary against a reference summary on a 1-5 scale using the Prometheus
    evaluation framework. Focuses on information density: capturing essential facts concisely with logical flow.

    Use this when: You have a reference summary and want to assess relative quality with rubric-based evaluation.
    Requires H2OGPTE API access. More structured than simple comparison metrics.

    Args:
        summary (str): Generated summary text to evaluate
        source (str, optional): Source document text to compare against
        reference_summary (str, optional): Reference summary that represents ideal quality
        model_name (str, optional): H2OGPTE LLM model. Default "meta-llama/Llama-3.3-70B-Instruct"
        timeout (int, optional): API timeout in seconds. Default 60

    Returns:
        Dict[str, Any]: Result dictionary with keys:
            - score (float): Normalized score from 0.0 to 1.0 (1-5 scale normalized)
            - raw_score (int): Original 1-5 scale score from Prometheus framework
            - explanation (str): LLM's detailed reasoning using the rubric
            - full_response (str): Complete LLM response for debugging
            - error (str, optional): Error message if API call failed

    Example:
        >>> result = evaluate_prometheus(
        ...     summary="Paris is French capital.",
        ...     reference_summary="Paris is France's capital."
        ... )
        >>> result['raw_score']  # e.g., 4 (out of 5, good information density)
        >>> result['explanation']  # "Captures essential facts concisely..."
    """
    try:
        # Validate required parameters
        if reference_summary is None:
            return {
                'score': None,
                'error': 'Reference summary is required for Prometheus evaluation'
            }

        system_prompt = "You are a fair judge assistant tasked with providing clear, objective feedback based on specific criteria, ensuring each assessment reflects the absolute standards set for performance."
        instruction = "Summarize the provided source text concisely without losing the core message or factual accuracy."
        rubric = "Which summary exhibits better 'Information Density'? A superior summary should include all essential facts from the source while using fewer words and maintaining a more logical flow than its competitor."

        prompt = create_prometheus_absolute_prompt()
        user_content = system_prompt + "\n\n" + prompt.format(
            instruction=instruction,
            response=summary,
            reference_answer=reference_summary,
            rubric=rubric
        )

        response = query_llm(user_content, model_name, timeout)
        score, explanation = parse_llm_response(response)

        if score is not None:
            normalized_score = score / 5.0
            return {
                'score': normalized_score,
                'raw_score': score,
                'explanation': explanation or 'No explanation provided',
                'full_response': response,
                'error': None
            }
        else:
            return {
                'score': None,
                'error': 'Failed to parse score from LLM response',
                'full_response': response,
            }

    except Exception as e:
        return {
            'score': None,
            'error': str(e)
        }


def evaluate_all(
    summary: str,
    source: str = None,
    reference_summary: str = None,
    model_name: str = 'meta-llama/Llama-3.3-70B-Instruct',
    timeout: int = 60,
    include_dag: bool = False,
    include_prometheus: bool = False
) -> Dict[str, Dict[str, Any]]:
    """
    Run comprehensive LLM-based evaluation across multiple quality dimensions (G-Eval + optional DAG/Prometheus).

    This function uses a large language model to evaluate summaries like a human expert would, checking:
    - Faithfulness: Are claims supported by source? (factual accuracy)
    - Coherence: Does it flow logically? (structure and readability)
    - Relevance: Does it capture key points? (completeness)
    - Fluency: Is it well-written? (grammar and style)
    Plus optional holistic evaluations (DAG, Prometheus).

    Use this when: You want human-like comprehensive quality assessment without token limits.
    Requires H2OGPTE API access. Allows full-length documents unlike local models.

    Args:
        summary (str): Generated summary to evaluate
        source (str, optional): Original source document text (required for faithfulness, relevance, DAG)
        reference_summary (str, optional): Reference summary for comparison (required for Prometheus if enabled)
        model_name (str, optional): H2OGPTE model. Default "meta-llama/Llama-3.3-70B-Instruct"
        timeout (int, optional): Timeout per API call in seconds. Default 60
        include_dag (bool, optional): Enable DAG decision tree evaluation (factual+complete+clear). Default False
        include_prometheus (bool, optional): Enable Prometheus holistic scoring vs reference. Default False

    Returns:
        Dict[str, Dict[str, Any]]: Dictionary mapping dimension names to results:
            - 'faithfulness': Factual accuracy check (always included)
            - 'coherence': Logical flow check (always included)
            - 'relevance': Key points coverage (always included)
            - 'fluency': Grammar and writing quality (always included)
            - 'dag': Decision tree evaluation (if include_dag=True)
            - 'prometheus': Reference-based holistic score (if include_prometheus=True)
            Each value has keys: score (0-1), raw_score (1-10), explanation, full_response

    Example:
        >>> results = evaluate_all(
        ...     summary="Paris is French capital.",
        ...     source="Paris is capital of France.",
        ...     reference_summary="Paris is France's capital.",
        ...     include_dag=True
        ... )
        >>> results['faithfulness']['raw_score']  # e.g., 9
        >>> results['relevance']['raw_score']  # e.g., 10
        >>> results['dag']['raw_score']  # e.g., 6 (out of 6)
        >>> list(results.keys())  # ['faithfulness', 'coherence', 'relevance', 'fluency', 'dag']
    """
    results = {
        'faithfulness': evaluate_faithfulness(
            summary=summary,
            source=source,
            model_name=model_name,
            timeout=timeout
        ),
        'coherence': evaluate_coherence(
            summary=summary,
            model_name=model_name,
            timeout=timeout
        ),
        'relevance': evaluate_relevance(
            summary=summary,
            source=source,
            model_name=model_name,
            timeout=timeout
        ),
        'fluency': evaluate_fluency(
            summary=summary,
            model_name=model_name,
            timeout=timeout
        ),
    }

    # Add DAG if requested
    if include_dag:
        results['dag'] = evaluate_dag(
            summary=summary,
            source=source,
            model_name=model_name,
            timeout=timeout
        )

    # Add Prometheus if requested
    if include_prometheus:
        results['prometheus'] = evaluate_prometheus(
            summary=summary,
            reference_summary=reference_summary,
            model_name=model_name,
            timeout=timeout
        )

    return results
