"""
SumOmniEval: Text Summarization Evaluation Tool
A comprehensive evaluation framework for assessing text summarization quality.
"""

# CRITICAL: Force CPU mode FIRST before ANY other imports
import force_cpu  # noqa: F401

import os
import sys

import streamlit as st
from typing import Dict, Any

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.evaluators.era1_word_overlap import compute_all_era1_metrics
from src.evaluators.era2_embeddings import compute_all_era2_metrics
from src.evaluators.era3_logic_checkers import compute_all_era3_metrics
from src.utils.data_loader import load_sample_data, get_sample_by_index
import pandas as pd
import json
from io import BytesIO

# Check if H2OGPTE API is available
try:
    from src.evaluators.era3_llm_judge import LLMJudgeEvaluator
    from dotenv import load_dotenv
    load_dotenv()
    import os
    H2OGPTE_AVAILABLE = bool(os.getenv('H2OGPTE_API_KEY') and os.getenv('H2OGPTE_ADDRESS'))
except ImportError:
    H2OGPTE_AVAILABLE = False


# Page configuration
st.set_page_config(
    page_title="SumOmniEval",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)


def initialize_session_state():
    """Initialize session state variables."""
    # Start with empty text areas - user must select a row
    if 'source_text' not in st.session_state:
        st.session_state.source_text = ""
    if 'reference_text' not in st.session_state:
        st.session_state.reference_text = ""
    if 'summary_text' not in st.session_state:
        st.session_state.summary_text = ""

    if 'results' not in st.session_state:
        st.session_state.results = None
    if 'last_sample' not in st.session_state:
        st.session_state.last_sample = None
    if 'selected_model' not in st.session_state:
        st.session_state.selected_model = 'meta-llama/Llama-3.3-70B-Instruct'
    if 'uploaded_documents' not in st.session_state:
        st.session_state.uploaded_documents = []
    if 'uploaded_dataset' not in st.session_state:
        st.session_state.uploaded_dataset = None
    if 'dataset_columns' not in st.session_state:
        st.session_state.dataset_columns = []
    if 'source_column' not in st.session_state:
        st.session_state.source_column = None
    if 'reference_column' not in st.session_state:
        st.session_state.reference_column = None
    if 'summary_column' not in st.session_state:
        st.session_state.summary_column = None
    if 'dataset_cleared' not in st.session_state:
        st.session_state.dataset_cleared = False
    if 'columns_selected' not in st.session_state:
        st.session_state.columns_selected = False
    if 'uploader_key' not in st.session_state:
        st.session_state.uploader_key = 0
    if 'last_uploader_key' not in st.session_state:
        st.session_state.last_uploader_key = -1
    if 'batch_evaluation_running' not in st.session_state:
        st.session_state.batch_evaluation_running = False
    if 'batch_results' not in st.session_state:
        st.session_state.batch_results = None
    if 'batch_file_format' not in st.session_state:
        st.session_state.batch_file_format = None
    if 'batch_filename' not in st.session_state:
        st.session_state.batch_filename = None
    if 'start_batch_eval' not in st.session_state:
        st.session_state.start_batch_eval = False


def parse_dataset_file(uploaded_file):
    """
    Parse uploaded dataset file into a DataFrame.

    Supports:
    - CSV: Standard comma-separated values
    - JSON: Array of objects [{"col1": "val1", "col2": "val2"}, ...]
    - Excel: .xlsx or .xls files
    - TSV: Tab-separated values

    Returns:
        tuple: (DataFrame, filename, error_message)
        - If successful: (df, filename, None)
        - If failed: (None, filename, error_message)
    """
    import json
    import pandas as pd

    filename = uploaded_file.name
    file_extension = filename.split('.')[-1].lower()

    try:
        # Parse based on file type
        if file_extension == 'csv':
            df = pd.read_csv(uploaded_file)

        elif file_extension == 'tsv':
            df = pd.read_csv(uploaded_file, sep='\t')

        elif file_extension in ['xlsx', 'xls']:
            df = pd.read_excel(uploaded_file)

        elif file_extension == 'json':
            content = json.loads(uploaded_file.read().decode('utf-8'))

            # Check if it's an array of objects
            if isinstance(content, list):
                df = pd.DataFrame(content)
            else:
                return None, filename, "JSON must be an array of objects: [{...}, {...}]"

        else:
            return None, filename, f"Unsupported file format: {file_extension}. Please use CSV, JSON, Excel, or TSV."

        # Validate: Must have at least 2 columns
        if len(df.columns) < 2:
            return None, filename, f"File must have at least 2 columns. Found only {len(df.columns)} column(s)."

        # Validate: Must have at least 1 row
        if len(df) == 0:
            return None, filename, "File is empty (no data rows)."

        # Clean up: Remove completely empty columns
        df = df.dropna(axis=1, how='all')

        # Clean up: Remove completely empty rows
        df = df.dropna(axis=0, how='all')

        # Reset index
        df = df.reset_index(drop=True)

        return df, filename, None

    except Exception as e:
        return None, filename, f"Error parsing file: {str(e)}"


def check_metric_availability():
    """Check which metrics are available without importing them (to avoid CUDA issues)."""
    import importlib.util

    available = {
        'era1': True,  # Always available (lightweight)
        'era2_bertscore': False,
        'era2_moverscore': False,
        'era3': False
    }

    # Check BERTScore (safe to import)
    try:
        import bert_score
        available['era2_bertscore'] = True
    except ImportError:
        pass

    # Check MoverScore WITHOUT importing (importing triggers CUDA)
    if importlib.util.find_spec('moverscore_v2') is not None:
        available['era2_moverscore'] = True

    # Check Transformers for Era 3
    try:
        from transformers import AutoModelForSequenceClassification
        available['era3'] = True
    except ImportError:
        pass

    return available


def display_metric_info():
    """Display information about available metrics."""
    with st.expander("📚 About the Metrics"):
        st.markdown("""
        **SumOmniEval** evaluates summary quality using **15 different metrics** organized into **3 evaluation eras**.

        Each era represents a different approach to measuring how good a summary is:

        - **Era 1: Word Overlap** - Do the words match?
        - **Era 2: Semantic Embeddings** - Does the meaning match?
        - **Era 3: Logic & AI Judges** - Is it factually correct and well-written?

        Together, these metrics provide a complete picture of summary quality from multiple perspectives.

        Click **"Evaluate Summary"** below to see detailed explanations for each era and their metrics.
        """)


def format_score_display(score: float, metric_type: str = "general", max_score: float = 1.0) -> str:
    """
    Format score with color coding.

    Args:
        score: The score value (0-1 or 0-10).
        metric_type: Type of metric for threshold selection.
        max_score: Maximum score value (1.0 or 10.0).

    Returns:
        Formatted HTML string with color.
    """
    # Define thresholds
    thresholds = {
        "general": (0.7, 0.4),
        "bertscore": (0.75, 0.65),  # Changed from 0.85, 0.75
        "bleu": (0.3, 0.15),
        "geval": (8.0, 5.0),        # For 1-10 scale
        "prometheus": (4.0, 2.5)    # For 1-5 scale

    }

    good_threshold, poor_threshold = thresholds.get(
        metric_type,
        thresholds["general"]
    )

    # For G-Eval (1-10 scale), adjust thresholds
    if max_score == 10.0:
        raw_score = score
        if raw_score >= good_threshold:
            color = "#28a745"  # Green
        elif raw_score >= poor_threshold:
            color = "#ffc107"  # Yellow
        else:
            color = "#dc3545"  # Red
        # No decimals for AI Simulator scores
        return f'<span style="color: {color}; font-weight: bold;">{int(round(raw_score))}/10</span>'
    elif max_score == 5.0:
        raw_score = score
        if raw_score >= good_threshold:
            color = "#28a745"  # Green
        elif raw_score >= poor_threshold:
            color = "#ffc107"  # Yellow
        else:
            color = "#dc3545"  # Red
        return f'<span style="color: {color}; font-weight: bold;">{int(round(raw_score))}/5</span>'
    else:
        # Standard 0-1 scale
        if score >= good_threshold:
            color = "#28a745"  # Green
        elif score >= poor_threshold:
            color = "#ffc107"  # Yellow
        else:
            color = "#dc3545"  # Red
        return f'<span style="color: {color}; font-weight: bold;">{score:.2f}/1.00</span>'


def display_results(results: Dict[str, Dict[str, Any]]):
    """
    Display evaluation results in an organized format.

    Args:
        results: Dictionary containing metric results by era.

    Note: Uses st.session_state.source_text to check for token limit warnings.
    """
    # Check if source text exceeds 400 words (for token limit warnings)
    source_text = st.session_state.get('source_text', '')
    word_count = len(source_text.split())
    show_token_warning = word_count > 400
    st.markdown("---")
    st.header("📊 Evaluation Results")

    # Era 1 Results
    if "era1" in results and results["era1"]:
        st.subheader("🔤 Era 1: Word Overlap & Fluency")

        # Add expandable explanation for Era 1
        with st.expander("ℹ️ What are Word Overlap Metrics?"):
            st.markdown("""
            **Theme**: The Age of "Exact Matches"

            In the early days (2000s), we treated text like Scrabble tiles. We assumed that if the computer used
            the exact same words as the human reference, it must be right. We didn't care about meaning; we only
            cared about matching symbols.

            **Our Metrics**:
            - **ROUGE & BLEU**: The industry standards. ROUGE focuses on recall (did you include all the reference
              words?), while BLEU focuses on precision.
            - **METEOR**: The clever cousin. ROUGE fails if you write "fast" instead of "quick." METEOR fixes this
              by counting synonyms and stem forms (running = run).
            - **Levenshtein Distance**: The spellchecker. It measures "edit distance" - how many deletions or swaps
              it takes to turn the summary into the reference.
            - **Perplexity**: This measures fluency, not truth. It checks how "surprised" a model is by the text.
              (Warning: A model can hallucinate a lie with perfect fluency/low perplexity).

            **The Failure Mode**: The "Death of ROUGE"
            - Source: "The movie was bad."
            - AI Summary: "The film was terrible."
            - ROUGE Score: 0.0 (Because "film" ≠ "movie" and "terrible" ≠ "bad")
            - **The Lesson**: These metrics punish creativity and paraphrase.

            **Pros & Cons**:
            - ✅ Fast, cheap, and standard (everyone knows them)
            - ❌ Misses synonyms, ignores structure, creates "Frankenstein" sentences
            """)

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**ROUGE Scores**")
            rouge_scores = results["era1"].get("ROUGE", {})
            if "error" not in rouge_scores:
                st.markdown(f"- ROUGE-1: {format_score_display(rouge_scores.get('rouge1', 0))}", unsafe_allow_html=True)
                st.markdown(f"- ROUGE-2: {format_score_display(rouge_scores.get('rouge2', 0))}", unsafe_allow_html=True)
                st.markdown(f"- ROUGE-L: {format_score_display(rouge_scores.get('rougeL', 0))}", unsafe_allow_html=True)
                st.caption("ℹ️ Normalized: higher = better word overlap")
            else:
                st.error(f"Error: {rouge_scores['error']}")

            st.markdown("**BLEU Score**")
            bleu_score = results["era1"].get("BLEU", {})
            if "error" not in bleu_score:
                st.markdown(f"- BLEU: {format_score_display(bleu_score.get('bleu', 0), 'bleu')}", unsafe_allow_html=True)
                st.caption("ℹ️ Normalized: higher = better precision")
            else:
                st.error(f"Error: {bleu_score['error']}")

        with col2:
            st.markdown("**METEOR Score**")
            meteor_score = results["era1"].get("METEOR", {})
            if "error" not in meteor_score:
                st.markdown(f"- METEOR: {format_score_display(meteor_score.get('meteor', 0))}", unsafe_allow_html=True)
                st.caption("ℹ️ Normalized: higher = better semantic match")
            else:
                st.error(f"Error: {meteor_score['error']}")

            st.markdown("**Levenshtein Similarity**")
            lev_score = results["era1"].get("Levenshtein", {})
            if "error" not in lev_score:
                st.markdown(f"- Similarity: {format_score_display(lev_score.get('levenshtein', 0))}", unsafe_allow_html=True)
                st.caption("ℹ️ Normalized: higher = more similar text")
            else:
                st.error(f"Error: {lev_score['error']}")

        with col3:
            st.markdown("**Perplexity (Fluency)**")
            perp_score = results["era1"].get("Perplexity", {})
            if "error" not in perp_score:
                st.markdown(f"- Perplexity: {perp_score.get('perplexity', 0):.2f}", unsafe_allow_html=True)
                st.caption("ℹ️ Lower perplexity = more fluent/natural text")
                st.markdown(f"- Fluency Score: {format_score_display(perp_score.get('normalized_score', 0))}", unsafe_allow_html=True)
                st.caption("ℹ️ Normalized: higher = better fluency")
            else:
                st.warning(f"⚠️ {perp_score['error']}")

    # Era 2 Results
    if "era2" in results and results["era2"]:
        st.markdown("---")
        st.subheader("🧠 Era 2: Embeddings")

        # Add expandable explanation for Era 2
        with st.expander("ℹ️ What are Embedding Metrics?"):
            st.markdown("""
            **Theme**: The Age of "Semantic Similarity"

            Around 2019, we realized that exact words don't matter - meaning matters. We started using 'Embeddings'
            (dense vector representations) to map words into space. If 'Lawyer' and 'Attorney' are close in space,
            they should count as a match.

            **Our Metrics**:
            - **BERTScore**: Calculates the cosine similarity between the summary's "vibe" and the source's "vibe"
              using contextual embeddings.
            - **MoverScore**: Uses "Earth Mover's Distance" (a transportation math problem) to calculate the "cost"
              of moving the meaning from the summary to the source. It is often softer and more robust than BERTScore.

            **The Failure Mode**: The "Negation Trap"
            - Sentence A: "The patient has cancer."
            - Sentence B: "The patient has no cancer."
            - BERTScore: 0.96 (96% similarity)
            - **The Lesson**: Because these sentences share almost all the same context, embeddings think they are
              identical. They are blind to small logic words like "not," "never," or "unless."

            **Pros & Cons**:
            - ✅ Captures synonyms and paraphrasing perfectly
            - ❌ Terrible at "Factuality" - Can't distinguish between opposite claims if context is similar
            """)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**BERTScore**")
            bert_scores = results["era2"].get("BERTScore", {})
            if "error" not in bert_scores:
                st.markdown(f"- Precision: {format_score_display(bert_scores.get('precision', 0), 'bertscore')}", unsafe_allow_html=True)
                st.caption("ℹ️ Normalized: higher = better relevance")
                st.markdown(f"- Recall: {format_score_display(bert_scores.get('recall', 0), 'bertscore')}", unsafe_allow_html=True)
                st.caption("ℹ️ Normalized: higher = better coverage")
                st.markdown(f"- F1: {format_score_display(bert_scores.get('f1', 0), 'bertscore')}", unsafe_allow_html=True)
                st.caption("ℹ️ Normalized: higher = better semantic match")
                if show_token_warning:
                    st.warning(f"⚠️ **Token Limit**: Your source text is {word_count} words, exceeding the 400-word limit. It was truncated to ~400 words for this metric.")
            else:
                st.error(f"Error: {bert_scores['error']}")

        with col2:
            st.markdown("**MoverScore**")
            mover_score = results["era2"].get("MoverScore", {})
            if "error" not in mover_score:
                st.markdown(f"- Score: {format_score_display(mover_score.get('moverscore', 0))}", unsafe_allow_html=True)
                st.caption("ℹ️ Normalized: higher = better semantic alignment")
                if show_token_warning:
                    st.warning(f"⚠️ **Token Limit**: Your source text is {word_count} words, exceeding the 400-word limit. It was truncated to ~400 words for this metric.")
            else:
                st.error(f"Error: {mover_score['error']}")

    # Era 3 Results - Logic & AI Judges (combined 3A + 3B)
    if ("era3" in results and results["era3"]) or ("era3b" in results and results["era3b"]):
        st.markdown("---")
        st.subheader("🎯 Era 3: Logic & AI Judges")

        with st.expander("ℹ️ What are Logic & AI Judge Metrics?"):
            st.markdown("""
            **Theme**: The Age of "Reasoning" & "Fact-Checking"

            We stopped trying to use math formulas to grade language. We realized that to judge a summary, you need
            to understand logic. We split into two camps: The **Logic Checkers** (who use NLI to find truth) and
            the **AI Simulators** (who mimic human grading).

            **Logic Checkers** (The "Truth" Squad):
            - **NLI (Natural Language Inference)**: Uses DeBERTa-v3 to determine if the source logically supports
              the summary. Asks: "Does Sentence A logically prove Sentence B?"
            - **FactCC**: A BERT model trained specifically to flag text as "Consistent" or "Inconsistent."
            - **FactChecker**: Uses an LLM to check every claim and identify specific unsupported statements.

            **Pros & Cons**:
            - ✅ The Gold Standard for Hallucination Detection (Faithfulness)
            - ❌ Doesn't measure "Vibe" or "Flow", and it requires significant compute!

            ---

            **AI Simulators** (LLM-as-a-Judge):
            - **G-Eval**: We give a rubric to a powerful LLM: "Rate this summary from 1-10 on faithfulness. Think
              step-by-step." Evaluates across 4 dimensions: Faithfulness, Coherence, Relevance, and Fluency.
            - **DAG (DeepEval)**: The structured judge. It breaks the evaluation into a decision tree (Directed
              Acyclic Graph) of smaller questions rather than one big score.

            **Pros & Cons**:
            - ✅ Highest correlation with human judgment. Can measure nuance, tone, and style.
            - ❌ Expensive, slow, and the judge can potentially be biased (LLMs prefer their own writing style).
            """)

        # Display Era 3 metrics only if they exist
        if "era3" in results and results["era3"]:
            st.markdown("### Logic Checkers")

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**NLI - Natural Language Inference** (DeBERTa-v3)")
                st.caption("**What it does:** Checks if the summary is logically supported by the source")
                st.caption("**Testing for:** Does the source prove the summary's claims?")
                st.caption("**Example:** Source says 'hired engineers' → Good: 'expanded team' | Bad: 'fired engineers'")

                nli_score = results["era3"].get("NLI", {})
                if "error" not in nli_score:
                    score_val = nli_score.get('nli_score', 0)
                    label = nli_score.get('label', 'UNKNOWN')

                    # Convert LABEL_0/1/2 to human-readable text
                    label_map = {
                        'LABEL_0': 'Entailment',
                        'LABEL_1': 'Neutral',
                        'LABEL_2': 'Contradiction',
                        'ENTAILMENT': 'Entailment',
                        'NEUTRAL': 'Neutral',
                        'CONTRADICTION': 'Contradiction'
                    }
                    readable_label = label_map.get(label, label)

                    # Add detailed explanations for each verdict
                    verdict_explanations = {
                        'Entailment': 'The source **logically supports** the summary. The claims in the summary can be proven true from the source.',
                        'Neutral': 'The source **neither confirms nor contradicts** the summary. Some information may be missing or unclear.',
                        'Contradiction': 'The source **contradicts** the summary. The summary makes claims that go against what the source says.'
                    }

                    st.markdown(f"**Score:** {format_score_display(score_val, 'general', 1.0)}", unsafe_allow_html=True)
                    st.caption("ℹ️ Normalized: higher = stronger logical support")
                    st.markdown(f"**Verdict:** {readable_label}")

                    # Show explanation for the verdict
                    if readable_label in verdict_explanations:
                        st.caption(f"ℹ️ {verdict_explanations[readable_label]}")

                    if show_token_warning:
                        st.warning(f"⚠️ **Token Limit**: Your source text is {word_count} words, exceeding the 400-word limit. It was truncated to ~400 words for this metric.")
                else:
                    st.error(f"Error: {nli_score['error']}")

            with col2:
                # FactCC results (if enabled)
                if "FactCC" in results["era3"]:
                    st.markdown("**FactCC - BERT Consistency Checker**")
                    st.caption("**What it does:** Binary fact-checking using fine-tuned BERT")
                    st.caption("**Testing for:** Is the summary factually consistent?")
                    st.caption("**Example:** Source: 'costs $50' → Good: '$50' | Bad: '$500'")

                    factcc_score = results["era3"].get("FactCC", {})
                    if "error" not in factcc_score and factcc_score.get('score') is not None:
                        score = factcc_score['score']
                        label = factcc_score.get('label', 'N/A')

                        st.markdown(f"**Score:** {format_score_display(score, 'general', 1.0)}", unsafe_allow_html=True)
                        st.caption("ℹ️ Normalized: higher = more factually consistent")
                        st.markdown(f"**Label:** {label}")
                        if show_token_warning:
                            st.warning(f"⚠️ **Token Limit**: Your source text is {word_count} words, exceeding the 400-word limit. It was truncated to ~400 words for this metric.")
                    else:
                        st.error(f"Error: {factcc_score.get('error')}")
                elif "FactChecker" in results["era3"]:
                    # FactChecker results (if enabled)
                    st.markdown("**FactChecker - LLM Fact Verification**")
                    st.caption("**What it does:** Uses an LLM to verify each claim in the summary against the source document")
                    st.caption("**Testing for:** Identifies factual errors, unsupported claims, or hallucinations with detailed reasoning")
                    st.caption("**Example:** Source: 'Einstein published relativity in 1905' → Good: '1905' (1.00) | Bad: '1915' (0.50)")

                    fc_score = results["era3"].get("FactChecker", {})
                    if "error" not in fc_score and fc_score.get('score') is not None:
                        score = fc_score['score']

                        st.markdown(f"**Score:** {format_score_display(score, 'general', 1.0)}", unsafe_allow_html=True)
                        st.caption("ℹ️ Normalized: 1.00 = perfect accuracy, 0.00 = completely inaccurate")
                        if fc_score.get('explanation'):
                            with st.expander("📝 Details"):
                                st.write(fc_score['explanation'])
                    else:
                        st.warning(f"⚠️ {fc_score.get('error', 'No result')}")
                else:
                    st.info("ℹ️ Optional metrics not enabled")

            # If both FactCC and FactChecker are enabled, show both
            if "FactCC" in results["era3"] and "FactChecker" in results["era3"]:
                st.markdown("---")
                st.markdown("**FactChecker - LLM Fact Verification**")
                st.caption("**What it does:** Uses an LLM to verify each claim in the summary against the source document")
                st.caption("**Testing for:** Identifies factual errors, unsupported claims, or hallucinations with detailed reasoning")
                st.caption("**Example:** Source: 'Einstein published relativity in 1905' → Good: '1905' (1.00) | Bad: '1915' (0.50)")

                fc_score = results["era3"].get("FactChecker", {})
                if "error" not in fc_score and fc_score.get('score') is not None:
                    score = fc_score['score']

                    st.markdown(f"**Score:** {format_score_display(score, 'general', 1.0)}", unsafe_allow_html=True)
                    st.caption("ℹ️ Normalized: 1.00 = perfect accuracy, 0.00 = completely inaccurate")
                    if fc_score.get('explanation'):
                        with st.expander("📝 Details"):
                            st.write(fc_score['explanation'])
                else:
                    st.warning(f"⚠️ {fc_score.get('error', 'No result')}")

    # Era 3B Results - AI Simulators (G-Eval)
    if "era3b" in results and results["era3b"]:
        st.markdown("---")
        st.markdown("### AI Simulators (LLM-as-a-Judge)")

        if "error" in results["era3b"]:
            st.error(f"Error: {results['era3b']['error']}")
        else:
            # Display each dimension
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**G-Eval: Faithfulness**")
                st.caption("**What it does:** Evaluates factual accuracy using LLM reasoning")
                st.caption("**Testing for:** Are all claims supported by the source?")
                st.caption("**Example:** Source: 'Sales increased 20% to $2M' → Good: 'Sales rose 20% to $2M' (9/10) | Bad: 'Sales doubled to $4M' (3/10)")

                faith_result = results["era3b"].get("faithfulness", {})
                if "error" not in faith_result and faith_result.get('score') is not None:
                    raw_score = faith_result.get('raw_score', faith_result['score'] * 10)
                    st.markdown(f"**Score:** {format_score_display(raw_score, 'geval', 10.0)}", unsafe_allow_html=True)
                    st.caption("ℹ️ Normalized: 10/10 = perfect accuracy, 1/10 = mostly inaccurate")
                    if faith_result.get('explanation'):
                        explanation_text = faith_result.get('explanation', 'N/A')
                        if len(explanation_text) > 100:
                            with st.expander("💬 Explanation"):
                                st.write(explanation_text)
                        else:
                            st.caption(f"💬 {explanation_text}")
                else:
                    st.warning(f"⚠️ {faith_result.get('error', 'No result')}")

                st.markdown("---")
                st.markdown("**G-Eval: Coherence**")
                st.caption("**What it does:** Evaluates logical flow and organization")
                st.caption("**Testing for:** Does it flow well? Clear transitions?")
                st.caption("**Example:** Good: 'First A, this caused B, then C' (9/10) | Bad: 'C, also A, B too' (4/10)")

                coh_result = results["era3b"].get("coherence", {})
                if "error" not in coh_result and coh_result.get('score') is not None:
                    raw_score = coh_result.get('raw_score', coh_result['score'] * 10)
                    st.markdown(f"**Score:** {format_score_display(raw_score, 'geval', 10.0)}", unsafe_allow_html=True)
                    st.caption("ℹ️ Normalized: 10/10 = excellent flow, 1/10 = disjointed")
                    if coh_result.get('explanation'):
                        explanation_text = coh_result.get('explanation', 'N/A')
                        if len(explanation_text) > 100:
                            with st.expander("💬 Explanation"):
                                st.write(explanation_text)
                        else:
                            st.caption(f"💬 {explanation_text}")
                else:
                    st.warning(f"⚠️ {coh_result.get('error', 'No result')}")

            with col2:
                st.markdown("**G-Eval: Relevance**")
                st.caption("**What it does:** Evaluates information selection")
                st.caption("**Testing for:** Main points covered? Irrelevant info excluded?")
                st.caption("**Example:** Source: Climate change (CO2, temp, policy) → Good: covers all 3 (9/10) | Bad: only CO2 (4/10)")

                rel_result = results["era3b"].get("relevance", {})
                if "error" not in rel_result and rel_result.get('score') is not None:
                    raw_score = rel_result.get('raw_score', rel_result['score'] * 10)
                    st.markdown(f"**Score:** {format_score_display(raw_score, 'geval', 10.0)}", unsafe_allow_html=True)
                    st.caption("ℹ️ Normalized: 10/10 = perfect coverage, 1/10 = misses key points")
                    if rel_result.get('explanation'):
                        explanation_text = rel_result.get('explanation', 'N/A')
                        if len(explanation_text) > 100:
                            with st.expander("💬 Explanation"):
                                st.write(explanation_text)
                        else:
                            st.caption(f"💬 {explanation_text}")
                else:
                    st.warning(f"⚠️ {rel_result.get('error', 'No result')}")

                st.markdown("---")
                st.markdown("**G-Eval: Fluency**")
                st.caption("**What it does:** Evaluates writing quality and grammar")
                st.caption("**Testing for:** Correct grammar? Natural language?")
                st.caption("**Example:** Good: 'The company expanded rapidly' (9/10) | Bad: 'Company did expanding rapid' (2/10)")

                flu_result = results["era3b"].get("fluency", {})
                if "error" not in flu_result and flu_result.get('score') is not None:
                    raw_score = flu_result.get('raw_score', flu_result['score'] * 10)
                    st.markdown(f"**Score:** {format_score_display(raw_score, 'geval', 10.0)}", unsafe_allow_html=True)
                    st.caption("ℹ️ Normalized: 10/10 = publication quality, 1/10 = multiple errors")
                    if flu_result.get('explanation'):
                        explanation_text = flu_result.get('explanation', 'N/A')
                        if len(explanation_text) > 100:
                            with st.expander("💬 Explanation"):
                                st.write(explanation_text)
                        else:
                            st.caption(f"💬 {explanation_text}")
                else:
                    st.warning(f"⚠️ {flu_result.get('error', 'No result')}")

            # DAG (DeepEval) results if enabled
            if "dag" in results["era3b"]:
                st.markdown("---")
                st.markdown("**DAG - Decision Tree Evaluation** (DeepEval)")
                st.caption("**What it does:** Evaluates summary through 3 sequential steps, each scoring 0-2 points")
                st.caption("**Testing for:** Step 1: Are facts correct? → Step 2: Are main points covered? → Step 3: Is writing clear?")
                st.caption("**Example:** Perfect summary gets 2+2+2=6/6 | Good summary gets 2+1+1=4/6 | Poor summary gets 1+0+1=2/6")

                dag_result = results["era3b"].get("dag", {})
                if "error" not in dag_result and dag_result.get('score') is not None:
                    raw_score = dag_result.get('raw_score', 0)

                    # Color code based on 0-6 scale
                    if raw_score >= 5:
                        color = "#28a745"  # Green
                    elif raw_score >= 3:
                        color = "#ffc107"  # Yellow
                    else:
                        color = "#dc3545"  # Red

                    st.markdown(f"**Score:** <span style='color: {color}; font-weight: bold;'>{raw_score}/6</span>", unsafe_allow_html=True)
                    st.caption("ℹ️ Normalized: 6/6 = perfect, 0/6 = fails all criteria")

                    col_a, col_b, col_c = st.columns(3)
                    with col_a:
                        step1 = dag_result.get('step1_factual', 'N/A')
                        st.caption(f"**Step 1 (Factual):** {step1}/2")
                    with col_b:
                        step2 = dag_result.get('step2_completeness', 'N/A')
                        st.caption(f"**Step 2 (Complete):** {step2}/2")
                    with col_c:
                        step3 = dag_result.get('step3_clarity', 'N/A')
                        st.caption(f"**Step 3 (Clarity):** {step3}/2")

                    if dag_result.get('explanation'):
                        with st.expander("📝 Decision Path"):
                            st.write(dag_result['explanation'])
                else:
                    st.warning(f"⚠️ {dag_result.get('error', 'No result')}")
            
            # Prometheus results if enabled
            if "prometheus" in results["era3b"]:
                st.markdown("---")
                st.markdown("**Prometheus - Reference-Based Evaluation**")
                st.caption("**What it does:** Evaluates summary quality against reference using LLM reasoning")
                st.caption("**Testing for:** How well does the summary match the reference?")
                st.caption("**Example:** Reference-based holistic quality assessment (1-5 scale)")

                prom_result = results["era3b"].get("prometheus", {})
                if "error" not in prom_result and prom_result.get('score') is not None:
                    raw_score = prom_result.get('raw_score', prom_result['score'])
                    st.markdown(f"**Score:** {format_score_display(raw_score, 'prometheus', 5.0)}", unsafe_allow_html=True)
                    st.caption("ℹ️ Normalized: 5/5 = excellent quality, 1/5 = poor quality")
                    if prom_result.get('explanation'):
                        explanation_text = prom_result.get('explanation', 'N/A')
                        if len(explanation_text) > 100:
                            with st.expander("💬 Explanation"):
                                st.write(explanation_text)
                        else:
                            st.caption(f"💬 {explanation_text}")
                else:
                    st.warning(f"⚠️ {prom_result.get('error', 'No result')}")

        # Batch evaluation button (show after DAG results if dataset uploaded)
        # Batch evaluation button (show after DAG results if dataset uploaded)
        if st.session_state.uploaded_dataset is not None and \
           st.session_state.source_column and st.session_state.summary_column and st.session_state.reference_column and \
           st.session_state.columns_selected and H2OGPTE_AVAILABLE:
            st.markdown("---")
            st.subheader("📊 Batch Evaluation")
            st.caption("Evaluate the entire dataset with API metrics (FactChecker, G-Eval, DAG, Prometheus)")

            # Use callback to set state immediately when button is clicked
            def start_batch_evaluation_main():
                st.session_state.start_batch_eval = True

            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                st.button(
                    "🚀 Evaluate Entire Dataset",
                    type="primary",
                    use_container_width=True,
                    key="batch_eval_main",
                    on_click=start_batch_evaluation_main
                )


def batch_evaluate_dataset(df: pd.DataFrame, source_col: str, reference_col: str, summary_col: str, model_name: str, progress_bar, status_text, preview_placeholder=None):
    """
    Evaluate entire dataset with API metrics only (no token limits).

    Args:
        df: DataFrame with source, reference, and summary columns
        source_col: Name of source text column
        reference_col: Name of reference summary column
        summary_col: Name of generated summary column
        model_name: LLM model to use for evaluation
        progress_bar: Streamlit progress bar widget
        status_text: Streamlit empty text widget for status updates
        preview_placeholder: Streamlit placeholder for live preview (first 10 rows)

    Returns:
        DataFrame with added metric columns
    """
    results_df = df.copy()

    # Initialize result columns (6 metrics: 4 G-Eval dimensions + DAG + Prometheus)
    results_df['geval_faithfulness'] = None
    results_df['geval_coherence'] = None
    results_df['geval_relevance'] = None
    results_df['geval_fluency'] = None
    results_df['dag_score'] = None
    results_df['prometheus_score'] = None

    # Create LLM Judge evaluator
    evaluator = LLMJudgeEvaluator(model_name=model_name)

    total_rows = len(df)

    for row_num, (idx, row) in enumerate(df.iterrows(), start=1):
        source_text = str(row[source_col])
        summary_text = str(row[summary_col])
        reference_text = str(row[reference_col])

        # Update progress
        progress_bar.progress(row_num / total_rows)
        status_text.text(f"Processing row {row_num}/{total_rows}")

        try:
            # G-Eval Faithfulness (this serves as our fact-checking metric)
            faithfulness_result = evaluator.evaluate_faithfulness(source_text, summary_text)
            results_df.at[idx, 'geval_faithfulness'] = faithfulness_result.get('score', None)

            # G-Eval Coherence
            coherence_result = evaluator.evaluate_coherence(summary_text)
            results_df.at[idx, 'geval_coherence'] = coherence_result.get('score', None)

            # G-Eval Relevance
            relevance_result = evaluator.evaluate_relevance(source_text, summary_text)
            results_df.at[idx, 'geval_relevance'] = relevance_result.get('score', None)

            # G-Eval Fluency
            fluency_result = evaluator.evaluate_fluency(summary_text)
            results_df.at[idx, 'geval_fluency'] = fluency_result.get('score', None)

            # DAG
            dag_result = evaluator.evaluate_dag(source_text, summary_text)
            results_df.at[idx, 'dag_score'] = dag_result.get('raw_score', None)

            # Prometheus
            prometheus_result = evaluator.evaluate_prometheus(source_text, summary_text, reference_text)
            results_df.at[idx, 'prometheus_score'] = prometheus_result.get('score', None)


        except Exception as e:
            st.error(f"Error processing row {row_num}: {str(e)}")
            # Fill with None on error
            results_df.at[idx, 'geval_faithfulness'] = None
            results_df.at[idx, 'geval_coherence'] = None
            results_df.at[idx, 'geval_relevance'] = None
            results_df.at[idx, 'geval_fluency'] = None
            results_df.at[idx, 'dag_score'] = None
            results_df.at[idx, 'prometheus_score'] = None

        # Update live preview for first 10 rows
        if preview_placeholder is not None and row_num <= 10:
            # Show preview of first 10 rows with all columns
            preview_df = results_df.head(10).copy()
            preview_placeholder.dataframe(preview_df, use_container_width=True)

    progress_bar.progress(1.0)
    status_text.text(f"✅ Completed! Processed {total_rows}/{total_rows} rows")

    return results_df


def export_results(df: pd.DataFrame, original_format: str, original_filename: str) -> BytesIO:
    """
    Export results DataFrame to same format as original file.

    Args:
        df: DataFrame with results
        original_format: File extension (csv, json, xlsx, tsv)
        original_filename: Original filename for naming

    Returns:
        BytesIO object with exported data
    """
    output = BytesIO()

    if original_format == 'csv':
        df.to_csv(output, index=False)
        output.seek(0)
    elif original_format == 'tsv':
        df.to_csv(output, index=False, sep='\t')
        output.seek(0)
    elif original_format in ['xlsx', 'xls']:
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df.to_excel(writer, index=False, sheet_name='Results')
        output.seek(0)
    elif original_format == 'json':
        json_data = df.to_json(orient='records', indent=2)
        output.write(json_data.encode('utf-8'))
        output.seek(0)

    return output


def main():
    """Main application function."""
    initialize_session_state()

    # Check if batch evaluation was triggered by button click
    if st.session_state.start_batch_eval:
        st.session_state.start_batch_eval = False  # Reset flag
        st.session_state.batch_results = None  # Clear old results
        st.session_state.batch_evaluation_running = True  # Start evaluation
        st.rerun()

    # Handle batch evaluation - MUST be at top before any other UI renders
    if st.session_state.batch_evaluation_running:
        try:
            st.title("📊 SumOmniEval - Batch Evaluation")
            st.markdown("---")
            st.header("📊 Batch Evaluation in Progress")

            # Get dataset info
            df = st.session_state.uploaded_dataset
            source_col = st.session_state.source_column
            reference_col = st.session_state.reference_column
            summary_col = st.session_state.summary_column
            model_name = st.session_state.selected_model

            if df is None:
                st.error("❌ No dataset found! Please upload a dataset first.")
                st.session_state.batch_evaluation_running = False
                st.stop()

            # Get original file format
            original_filename = st.session_state.get('last_uploaded_file', 'results')
            file_extension = original_filename.split('.')[-1].lower()

            # Show evaluation info
            num_rows = len(df)
            st.info(f"🔄 Evaluating {num_rows} rows with API metrics (G-Eval, DAG, Prometheus)...")
            st.caption(f"Using model: **{model_name.split('/')[-1]}**")
            st.caption("This may take several minutes depending on dataset size...")
            st.markdown("")  # spacing

            # Create progress bar and status text widgets
            st.markdown("**Progress:**")
            progress_bar = st.progress(0)
            status_text = st.empty()
            status_text.text("Initializing...")

            # Create preview section for first 10 rows
            st.markdown("")
            st.markdown("**Live Preview (first 10 rows):**")
            preview_placeholder = st.empty()

            # Run batch evaluation with live preview
            results_df = batch_evaluate_dataset(df, source_col, reference_col, summary_col, model_name, progress_bar, status_text, preview_placeholder)

            # Store results
            st.session_state.batch_results = results_df
            st.session_state.batch_file_format = file_extension
            st.session_state.batch_filename = original_filename
            st.session_state.batch_evaluation_running = False

            st.success("✅ Batch evaluation complete!")
            st.rerun()

        except Exception as e:
            st.error(f"❌ Error during batch evaluation: {str(e)}")
            st.session_state.batch_evaluation_running = False

        # Don't render anything else while evaluating
        st.stop()

    # Check available metrics
    available = check_metric_availability()

    # Header
    st.title("📊 SumOmniEval")
    st.markdown("### Text Summarization Evaluation Tool")
    if H2OGPTE_AVAILABLE:
        st.markdown("Evaluate summary quality across **3 eras + AI evaluators** (up to 15 metrics)")
    else:
        st.markdown("Evaluate summary quality across **9 state-of-the-art NLP metrics**")

    # Show installation warning if needed
    if not available['era2_bertscore'] or not available['era3']:
        with st.warning("⚠️ **Installation Notice**"):
            missing = []
            if not available['era2_bertscore']:
                missing.append("Era 2 BERTScore")
            if not available['era2_moverscore']:
                missing.append("Era 2 MoverScore (optional)")
            if not available['era3']:
                missing.append("Era 3 metrics (optional)")

            st.markdown(f"""
            Some optional metrics are not installed: **{', '.join(missing)}**

            **The app will work with Era 1 metrics (ROUGE, BLEU, METEOR, Levenshtein).**

            To install all metrics, see `INSTALLATION_FIXES.md` or run:
            ```bash
            pip3 install -r requirements-minimal.txt
            ```
            """)

    display_metric_info()

    # Sidebar
    st.sidebar.header("⚙️ Configuration")

    # File uploader for dataset
    st.sidebar.subheader("📤 Upload Your Dataset")
    uploaded_file = st.sidebar.file_uploader(
        "Upload CSV, JSON, Excel, or TSV file",
        type=['csv', 'json', 'xlsx', 'xls', 'tsv'],
        key=f"dataset_uploader_{st.session_state.uploader_key}",
        help="Upload a dataset with multiple rows. File must have at least 2 columns.\n"
             "Supported formats:\n"
             "• CSV/TSV: Standard tabular format\n"
             "• JSON: Array of objects [{...}, {...}]\n"
             "• Excel: .xlsx or .xls files"
    )

    # Process uploaded file
    if uploaded_file is not None:
        filename = uploaded_file.name
        current_uploader_key = st.session_state.uploader_key

        # Check if this file is from a new uploader widget (after clear) or a different file
        is_new_uploader = current_uploader_key != st.session_state.last_uploader_key
        is_different_file = st.session_state.get('last_uploaded_file') != filename

        # Should process if: new uploader widget OR different file OR no dataset loaded
        should_process = (
            is_new_uploader or
            is_different_file or
            st.session_state.uploaded_dataset is None
        )

        if should_process:
            df, filename, error = parse_dataset_file(uploaded_file)

            if error:
                st.sidebar.error(f"❌ {error}")
            else:
                # Store dataset in session state
                st.session_state.uploaded_dataset = df
                st.session_state.dataset_columns = list(df.columns)
                st.session_state.last_uploaded_file = filename
                st.session_state.last_uploader_key = current_uploader_key
                st.session_state.dataset_cleared = False
                st.sidebar.success(f"✅ Loaded: {filename}")
                st.sidebar.info(f"📊 {len(df)} rows × {len(df.columns)} columns")

                # Clear text areas when new file is uploaded
                st.session_state.source_text = ""
                st.session_state.reference_text = ""
                st.session_state.summary_text = ""

                # Reset column selections when new file is uploaded
                st.session_state.source_column = None
                st.session_state.reference_column = None
                st.session_state.summary_column = None
                st.session_state.columns_selected = False
                st.session_state.data_selector = 0  # Reset to placeholder
        else:
            # Already processed this file
            if st.session_state.uploaded_dataset is not None:
                st.sidebar.success(f"✅ Loaded: {filename}")
                st.sidebar.info(f"📊 {len(st.session_state.uploaded_dataset)} rows × {len(st.session_state.dataset_columns)} columns")

    # Column selection (only show if dataset is uploaded)
    if st.session_state.uploaded_dataset is not None and not st.session_state.dataset_cleared:
        st.sidebar.markdown("---")
        st.sidebar.subheader("🔧 Map Columns")

        col1, col2 = st.sidebar.columns(2)

        def on_column_change():
            """Mark that user has selected columns."""
            if st.session_state.source_column and st.session_state.summary_column and st.session_state.reference_column:
                st.session_state.columns_selected = True

        with col1:
            source_col = st.selectbox(
                "Source Text Column:",
                options=st.session_state.dataset_columns,
                key="source_col_selector",
                help="Select the column containing the full source documents",
                on_change=on_column_change
            )
            st.session_state.source_column = source_col

        with col2:
            # Filter out the source column from summary options
            summary_options = [col for col in st.session_state.dataset_columns if col != source_col]
            if summary_options:
                summary_col = st.selectbox(
                    "Summary Column:",
                    options=summary_options,
                    key="summary_col_selector",
                    help="Select the column containing the summaries",
                    on_change=on_column_change
                )
                st.session_state.summary_column = summary_col
            else:
                st.sidebar.warning("⚠️ Need at least 2 columns")
        
        # Filter out already selected columns
        reference_options = [col for col in st.session_state.dataset_columns 
                            if col != source_col and col != st.session_state.summary_column]
        if reference_options:
            reference_col = st.selectbox(
                "Reference Column:",
                options=reference_options,
                key="reference_col_selector",
                help="Select the column containing the reference summaries",
                on_change=on_column_change
            )
            st.session_state.reference_column = reference_col


        # Mark columns as selected when both are set
        if st.session_state.source_column and st.session_state.summary_column and st.session_state.reference_column:
            st.session_state.columns_selected = True

        # Show preview of column mapping
        if st.session_state.source_column and st.session_state.summary_column and st.session_state.reference_column:
            st.sidebar.caption(f"✅ Source: `{st.session_state.source_column}`")
            st.sidebar.caption(f"✅ Summary: `{st.session_state.summary_column}`")
            st.sidebar.caption(f"✅ Reference: `{st.session_state.reference_column}`")

    # Show clear button if dataset is uploaded
    if st.session_state.uploaded_dataset is not None and not st.session_state.dataset_cleared:
        if st.sidebar.button("🗑️ Clear Uploaded Dataset"):
            # Increment uploader key to force new uploader widget
            st.session_state.uploader_key += 1
            # Set flag to prevent re-processing the same file
            st.session_state.dataset_cleared = True
            # Clear all dataset-related state
            st.session_state.uploaded_dataset = None
            st.session_state.dataset_columns = []
            st.session_state.source_column = None
            st.session_state.reference_column = None
            st.session_state.summary_column = None
            st.session_state.columns_selected = False
            st.session_state.data_selector = 0  # Reset to first sample
            # Note: Keep last_uploaded_file so we know not to re-process it
            # Load first sample data
            try:
                sample = get_sample_by_index(0)
                st.session_state.source_text = sample['source']
                st.session_state.reference_text = sample.get('reference', '')
                st.session_state.summary_text = sample['summary']
            except:
                pass
            st.rerun()

    # Data selector
    st.sidebar.markdown("---")
    st.sidebar.subheader("📁 Select Data")

    try:
        # Determine data source
        if st.session_state.uploaded_dataset is not None and \
            st.session_state.source_column and st.session_state.summary_column and st.session_state.reference_column and \
            st.session_state.columns_selected:
            # Use uploaded dataset
            df = st.session_state.uploaded_dataset
            source_col = st.session_state.source_column
            reference_col = st.session_state.reference_column
            summary_col = st.session_state.summary_column

            # Build options list with placeholder
            all_options = ["-- Select a row --"] + [f"Row {i+1}" for i in range(len(df))]

            # Use on_change callback to load data
            def on_data_change():
                """Callback when data selection changes."""
                selected_idx = st.session_state.data_selector
                if selected_idx > 0:  # Not the placeholder
                    row = df.iloc[selected_idx - 1]  # Adjust for placeholder
                    st.session_state.source_text = str(row[source_col])
                    st.session_state.reference_text = str(row[reference_col])
                    st.session_state.summary_text = str(row[summary_col])

            # Determine default index (0 = placeholder)
            default_idx = st.session_state.get('data_selector', 0)
            if default_idx >= len(all_options):
                default_idx = 0

            selected_data = st.sidebar.selectbox(
                "Choose row to evaluate:",
                options=range(len(all_options)),
                format_func=lambda x: all_options[x],
                index=default_idx,
                key="data_selector",
                on_change=on_data_change
            )

            # Show current selection if a row is selected (not placeholder)
            if selected_data > 0:
                st.sidebar.info(f"📄 Currently: {all_options[selected_data]}")

        else:
            # Use sample data (default)
            df = load_sample_data()
            num_samples = len(df)

            # Build options list with placeholder
            all_options = ["-- Select a row --"] + [f"Sample {i+1}" for i in range(num_samples)]

            # Determine default index (0 = placeholder)
            default_idx = st.session_state.get('data_selector', 0)
            if default_idx >= len(all_options):
                default_idx = 0

            # Use on_change callback to auto-load
            def on_data_change():
                """Callback when data selection changes."""
                selected_idx = st.session_state.data_selector
                if selected_idx > 0:  # Not the placeholder
                    sample = get_sample_by_index(selected_idx - 1)  # Adjust for placeholder
                    st.session_state.source_text = sample['source']
                    st.session_state.reference_text = sample.get('reference', '')
                    st.session_state.summary_text = sample['summary']

            selected_data = st.sidebar.selectbox(
                "Choose sample to evaluate:",
                options=range(len(all_options)),
                format_func=lambda x: all_options[x],
                index=default_idx,
                key="data_selector",
                on_change=on_data_change
            )

            # Show current selection if a sample is selected (not placeholder)
            if selected_data > 0:
                st.sidebar.info(f"📄 Currently: {all_options[selected_data]}")

    except Exception as e:
        st.sidebar.error(f"Error loading data: {e}")

    # Batch evaluation button (only show if dataset uploaded and API available)
    if st.session_state.uploaded_dataset is not None and \
        st.session_state.source_column and st.session_state.summary_column and st.session_state.reference_column and \
        st.session_state.columns_selected and H2OGPTE_AVAILABLE:
        st.sidebar.markdown("---")
        st.sidebar.subheader("📊 Batch Evaluation")
        st.sidebar.caption("Evaluate entire dataset with API metrics (FactChecker, G-Eval, DAG, Prometheus)")

        # Use callback to set state immediately when button is clicked
        def start_batch_evaluation():
            st.session_state.start_batch_eval = True

        st.sidebar.button(
            "🚀 Evaluate Entire Dataset",
            type="primary",
            use_container_width=True,
            on_click=start_batch_evaluation,
            key="batch_eval_sidebar"
        )

    # Model selection for LLM-as-a-Judge (if API available)
    if H2OGPTE_AVAILABLE:
        st.sidebar.markdown("---")
        st.sidebar.subheader("🤖 LLM Model Selection")

        # Available models list (only the 3 tested working models)
        available_models = [
            'meta-llama/Llama-3.3-70B-Instruct',  # Default
            'meta-llama/Meta-Llama-3.1-70B-Instruct',
            'deepseek-ai/DeepSeek-R1',
        ]

        selected_model = st.sidebar.selectbox(
            "Select LLM Model:",
            options=available_models,
            index=0,  # Default to Llama-3.3-70B
            help="Choose the LLM model for API metrics (FactChecker, G-Eval, DAG, Prometheus)"
        )
        st.session_state.selected_model = selected_model
        st.sidebar.caption(f"✅ Using: {selected_model.split('/')[-1]}")

    # Automatically set which metrics to run based on availability
    run_era1 = True  # Always available
    run_era2 = available['era2_bertscore']  # Run if BERTScore available
    run_era3 = available['era3']  # Run if Era 3 available
    use_factcc = available['era3']  # Use FactCC if Era 3 available
    use_factchecker = available['era3'] and H2OGPTE_AVAILABLE  # Use if both available
    run_era3b = H2OGPTE_AVAILABLE  # Run if API configured
    use_dag = H2OGPTE_AVAILABLE  # Use DAG if API configured
    use_prometheus = H2OGPTE_AVAILABLE  # Use Prometheus if API configured

    # Main content
    st.header("📝 Input Texts")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Source Text")
        source_text = st.text_area(
            "Enter the original source document:",
            value=st.session_state.source_text,
            height=300,
            help="The source text to summarize"
        )

    with col2:
        st.subheader("Reference Summary")
        reference_text = st.text_area(
            "Enter the reference summary:",
            value=st.session_state.reference_text,
            height=300,
            help="The reference summary"
        )
    
    with col3: 
        st.subheader("Summary")
        summary_text = st.text_area(
            "Enter the generated summary:",
            value=st.session_state.summary_text,
            height=300,
            help="The summary to evaluate"
        )

    # Update session state with current text area values
    st.session_state.source_text = source_text
    st.session_state.reference_text = reference_text
    st.session_state.summary_text = summary_text

    # Evaluation button
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 1, 1])

    with col2:
        evaluate_button = st.button(
            "🚀 Evaluate Summary",
            type="primary",
            use_container_width=True
        )

    # Run evaluation
    if evaluate_button:
        if not source_text.strip() or not summary_text.strip():
            st.error("⚠️ Please provide both source text and summary.")
        else:
            results = {}

            with st.spinner("Computing evaluation metrics..."):
                # Era 1
                if run_era1:
                    with st.spinner("Running Era 1 metrics..."):
                        results["era1"] = compute_all_era1_metrics(
                            source_text,
                            summary_text
                        )

                # Era 2
                if run_era2:
                    with st.spinner("Running Era 2 metrics (may take 5-10 seconds)..."):
                        results["era2"] = compute_all_era2_metrics(
                            source_text,
                            summary_text
                        )

                # Era 3 - Logic Checkers
                if run_era3:
                    spinner_text = "Running Era 3 Logic Checkers (NLI"
                    if use_factcc:
                        spinner_text += " + FactCC"
                    if use_factchecker:
                        spinner_text += " + FactChecker API"
                    spinner_text += ")..."

                    with st.spinner(spinner_text):
                        results["era3"] = compute_all_era3_metrics(
                            source_text,
                            summary_text,
                            use_factcc=use_factcc,
                            use_factchecker=use_factchecker,
                            factchecker_model=st.session_state.selected_model if use_factchecker else None
                        )

                # Era 3 - AI Simulators
                if run_era3b:
                    spinner_text = f"Running Era 3 AI Simulators (G-Eval"
                    if use_dag:
                        spinner_text += " + DAG"
                    if use_prometheus:
                        spinner_text += " + Prometheus"
                    spinner_text += f") using {st.session_state.selected_model.split('/')[-1]}..."

                    with st.spinner(spinner_text):
                        try:
                            evaluator = LLMJudgeEvaluator(model_name=st.session_state.selected_model)
                            results["era3b"] = evaluator.evaluate_all(
                                source_text,
                                reference_text,
                                summary_text,
                                timeout=90,
                                include_dag=use_dag,
                                include_prometheus=use_prometheus
                            )
                        except Exception as e:
                            results["era3b"] = {"error": str(e)}

            st.session_state.results = results
            st.success("✅ Evaluation complete!")

    # Display batch results and download button (only if not currently evaluating)
    if st.session_state.batch_results is not None and not st.session_state.batch_evaluation_running:
        st.markdown("---")
        st.header("📥 Download Results")

        results_df = st.session_state.batch_results
        file_format = st.session_state.batch_file_format
        original_filename = st.session_state.batch_filename

        # Show preview
        st.subheader("Preview Results")
        st.dataframe(results_df.head(10), use_container_width=True)
        st.caption(f"Showing first 10 rows of {len(results_df)} total rows")

        # Export results
        output_filename = original_filename.replace(f'.{file_format}', f'_evaluated.{file_format}')
        export_data = export_results(results_df, file_format, output_filename)

        # Download button
        mime_types = {
            'csv': 'text/csv',
            'tsv': 'text/tab-separated-values',
            'xlsx': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            'xls': 'application/vnd.ms-excel',
            'json': 'application/json'
        }

        st.download_button(
            label=f"⬇️ Download {output_filename}",
            data=export_data,
            file_name=output_filename,
            mime=mime_types.get(file_format, 'application/octet-stream'),
            type="primary",
            use_container_width=True
        )

        # Clear results button
        if st.button("🗑️ Clear Results", use_container_width=True):
            st.session_state.batch_results = None
            st.session_state.batch_file_format = None
            st.session_state.batch_filename = None
            st.rerun()

    # Display results
    if st.session_state.results:
        display_results(st.session_state.results)

    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #666;'>
        <p>SumOmniEval v1.0.0 | Built with Streamlit |
        <a href='https://github.com/yourusername/SumOmniEval'>GitHub</a></p>
        </div>
        """,
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
