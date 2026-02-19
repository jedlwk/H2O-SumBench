"""
H2O SumBench: Text Summarization Evaluation Tool
A comprehensive evaluation framework for assessing text summarization quality.
"""

# CRITICAL: Force CPU mode FIRST before ANY other imports
import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.utils import force_cpu  # noqa: F401

import streamlit as st
from typing import Dict, Any

from src.evaluators.era1_word_overlap import compute_all_era1_metrics
from src.evaluators.era2_embeddings import compute_all_era2_metrics
from src.evaluators.era3_logic_checkers import compute_all_era3_metrics
from src.evaluators.completeness_metrics import compute_all_completeness_metrics
from src.utils.data_loader import load_sample_data, get_sample_by_index, get_sample_labels
import pandas as pd
import json
from io import BytesIO

# Check if H2OGPTE API is available
try:
    from src.evaluators.era3_llm_judge import (
        evaluate_faithfulness,
        evaluate_coherence,
        evaluate_relevance,
        evaluate_fluency,
        evaluate_dag,
        evaluate_prometheus,
        evaluate_all,
        evaluate_custom
    )
    from dotenv import load_dotenv
    load_dotenv()
    import os
    H2OGPTE_AVAILABLE = bool(os.getenv('H2OGPTE_API_KEY') and os.getenv('H2OGPTE_ADDRESS'))
except ImportError:
    H2OGPTE_AVAILABLE = False


# Logo path
LOGO_PATH = os.path.join(os.path.dirname(__file__), '..', 'logo.png')

# Page configuration
st.set_page_config(
    page_title="H2O SumBench | H2O.ai",
    page_icon=LOGO_PATH if os.path.exists(LOGO_PATH) else "📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for branding, toast, educational content
CUSTOM_CSS = """
<style>
@keyframes fadeInOut {
    0% { opacity: 0; transform: translateX(100px); }
    10% { opacity: 1; transform: translateX(0); }
    90% { opacity: 1; transform: translateX(0); }
    100% { opacity: 0; transform: translateX(100px); }
}

.toast-notification {
    position: fixed;
    top: 60px;
    right: 20px;
    padding: 12px 20px;
    background: #1E1E2E;
    color: #E0E0E0;
    border-left: 3px solid #FEC925;
    border-radius: 8px;
    font-weight: 500;
    box-shadow: 0 6px 20px rgba(0,0,0,0.5);
    z-index: 9999;
    animation: fadeInOut 4s ease-in-out forwards;
    max-width: 350px;
}

/* Gold top border on main content */
section.main > div.block-container {
    border-top: 3px solid #FEC925;
    padding-top: 2rem;
}

/* Sidebar styling */
section[data-testid="stSidebar"] > div {
    background: linear-gradient(180deg, #1A1A2E 0%, #1E1E2E 100%);
}

/* Dashboard metric boxes */
[data-testid="stMetric"] {
    background-color: #1A1A2E;
    border-left: 3px solid #FEC925;
    border-radius: 8px;
    padding: 12px 16px;
}

/* Gold dividers */
.gold-divider {
    border: none;
    border-top: 2px solid #FEC925;
    margin: 1.5rem 0;
}

/* Section divider -- heavy gold bar for major sections (Faithfulness / Completeness / Holistic) */
.section-divider {
    border: none;
    border-top: 3px solid #FEC925;
    margin: 2rem 0 1.5rem 0;
}

/* Metric divider -- subtle separator between individual metrics within a section */
.metric-divider {
    border: none;
    border-top: 1px solid #2A2A3E;
    margin: 1rem 0;
}

/* Assessment summary box */
.assessment-box {
    border: 4px solid #4A90D9;
    border-radius: 10px;
    padding: 16px 20px;
    margin: 1.5rem 0 0.5rem 0;
}
.assessment-box.good { border-color: #28a745; background: rgba(40,167,69,0.08); }
.assessment-box.mixed { border-color: #FEC925; background: rgba(254,201,37,0.08); }
.assessment-box.bad { border-color: #dc3545; background: rgba(220,53,69,0.08); }
.assessment-box .assess-title { font-weight: 700; font-size: 1.05em; }
.assessment-box .assess-note { margin-top: 8px; font-size: 0.9em; opacity: 0.85; }

/* Educational callout box */
.edu-callout {
    border-left: 4px solid #FEC925;
    background-color: #1A1A2E;
    color: #E0E0E0;
    padding: 12px 16px;
    border-radius: 0 8px 8px 0;
    margin: 12px 0;
}

/* Score interpretation box */
.score-interpretation {
    border-left: 4px solid #4A90D9;
    background-color: #141824;
    color: #E0E0E0;
    padding: 12px 16px;
    border-radius: 0 8px 8px 0;
    margin: 8px 0;
}

/* Caveat/warning box */
.caveat-box {
    border-left: 4px solid #dc3545;
    background-color: #1C1015;
    color: #E0E0E0;
    padding: 10px 14px;
    border-radius: 0 8px 8px 0;
    margin: 8px 0;
    font-size: 0.9em;
}

/* Branded header */
.branded-header {
    padding-bottom: 0.5rem;
}
.branded-header h1 {
    margin-bottom: 0.2rem;
    color: #FFFFFF;
}
.h2o-gold {
    color: #FEC925;
    font-weight: 700;
}

/* Evaluate button override - clean white outline on dark */
button[data-testid="stBaseButton-secondary"] {
    border: 2px solid #E0E0E0 !important;
    color: #E0E0E0 !important;
    background-color: transparent !important;
    font-weight: 600 !important;
    padding: 0.85rem 1.5rem !important;
    transition: all 0.2s ease !important;
}
button[data-testid="stBaseButton-secondary"]:hover {
    border-color: #FEC925 !important;
    color: #FEC925 !important;
    background-color: rgba(254, 201, 37, 0.08) !important;
}

/* Expander headers */
[data-testid="stExpander"] {
    border-color: #2A2A3E;
}

/* Text area borders */
textarea {
    border-color: #2A2A3E !important;
    background-color: #161922 !important;
    color: #E0E0E0 !important;
}

/* Table styling */
table {
    color: #E0E0E0;
}
th {
    color: #FEC925 !important;
}

/* Links */
a {
    color: #FEC925 !important;
}

/* Info/warning/success boxes */
[data-testid="stAlert"] {
    border-radius: 8px;
}
</style>
"""


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
    if 'has_reference' not in st.session_state:
        st.session_state.has_reference = False
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
    if 'toast_message' not in st.session_state:
        st.session_state.toast_message = None
    if 'custom_judge_criteria' not in st.session_state:
        st.session_state.custom_judge_criteria = ""
    if 'custom_judge_result' not in st.session_state:
        st.session_state.custom_judge_result = None
    if 'custom_judge_running' not in st.session_state:
        st.session_state.custom_judge_running = False


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

    # Check MoverScore - use vendored copy (moverscore_v2_patched) in src/evaluators
    try:
        from src.evaluators.moverscore_wrapper import word_mover_score
        available['era2_moverscore'] = True
    except Exception:
        pass

    # Check Transformers for Era 3
    try:
        from transformers import AutoModelForSequenceClassification
        available['era3'] = True
    except ImportError:
        pass

    return available


def display_metric_info():
    """Display information about available metrics."""
    with st.expander("Why Evaluate Summaries? Understanding the Framework"):
        st.markdown("""
        ## The Problem: How Do You Know if a Summary is Good?

        When an AI generates a summary, we need to answer two fundamental questions:

        1. **Is it accurate?** Does the summary faithfully represent the source without making things up?
        2. **Is it complete?** Does it capture the important information?

        And if you have a "gold standard" reference summary:

        3. **Does it match the expected output?** How close is it to what a human expert would write?
        """)

        st.markdown('<div class="gold-divider"></div>', unsafe_allow_html=True)

        st.markdown("""
        ## Our Two-Stage Evaluation Approach

        ### Stage 1: Source vs. Summary (INTEGRITY CHECK)
        *"Can we trust this summary?"*

        We compare the **Generated Summary** directly against the **Source Text** to check:

        **Faithfulness** -- *Is the summary honest?*
        - Does it only contain information from the source?
        - Does it avoid "hallucinating" facts that don't exist?
        - *Finance example:* If the source says "operating margin improved to 16.2%", the summary shouldn't say "18.5%"
        - *Accident example:* If the report says "4 workers killed", the summary shouldn't say "6 fatalities"

        **Completeness** -- *Did it capture what matters?*
        - Are the main points included?
        - Did important details get lost?
        - *Finance example:* An earnings summary should include revenue, margins, and guidance -- not just the headline number
        - *Accident example:* An incident report should cover cause, casualties, and response -- not just the event
        """)

        st.markdown("""
        ### Stage 2: Generated vs. Reference Summary (CONFORMANCE CHECK)
        *"How does it compare to a human-written summary?"*

        If you have a reference summary (written by an expert), we measure how closely the generated summary matches it:

        **Semantic Match** -- *Same meaning, different words?*
        - Does it convey the same ideas even with different phrasing?
        - Example: "The CEO resigned" vs "The company's leader stepped down" = semantically similar

        **Lexical Match** -- *Same words and structure?*
        - How much word-for-word overlap exists?
        - Useful for checking if specific terminology was preserved (e.g., "EBITDA margin" vs "profitability metric")
        """)

        st.markdown('<div class="gold-divider"></div>', unsafe_allow_html=True)

        st.markdown("""
        ## Why This Matters

        | Use Case | Key Metrics |
        |----------|-------------|
        | **Finance: Fact-checking earnings summaries** | Faithfulness (NLI, AlignScore, FactCC) |
        | **Accident: Report completeness verification** | Completeness (Coverage, G-Eval Relevance) |
        | **Regulatory language matching** | Conformance (ROUGE, chrF++) |
        | **Quality assurance for production** | All metrics combined |
        """)

        st.markdown("""
        <div class="edu-callout">
        <strong>How to Read Your Results (5 Steps)</strong><br>
        1. Start with the <strong>Summary at a Glance</strong> dashboard for a quick health check<br>
        2. Check <strong>Faithfulness</strong> first &mdash; an inaccurate summary is worse than an incomplete one<br>
        3. Review <strong>Coverage</strong> &mdash; low coverage with high quality means a concise but focused summary<br>
        4. Expand <strong>"What does my score mean?"</strong> under any metric for plain-English interpretation<br>
        5. If you have a reference, use <strong>Stage 2</strong> to see how closely your summary matches the gold standard
        </div>
        """, unsafe_allow_html=True)


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
            color = "#FEC925"  # H2O Gold
        else:
            color = "#dc3545"  # Red
        # No decimals for AI Simulator scores
        return f'<span style="color: {color}; font-weight: bold;">{int(round(raw_score))}/10</span>'
    elif max_score == 5.0:
        raw_score = score
        if raw_score >= good_threshold:
            color = "#28a745"  # Green
        elif raw_score >= poor_threshold:
            color = "#FEC925"  # H2O Gold
        else:
            color = "#dc3545"  # Red
        return f'<span style="color: {color}; font-weight: bold;">{int(round(raw_score))}/5</span>'
    else:
        # Standard 0-1 scale
        if score >= good_threshold:
            color = "#28a745"  # Green
        elif score >= poor_threshold:
            color = "#FEC925"  # H2O Gold
        else:
            color = "#dc3545"  # Red
        return f'<span style="color: {color}; font-weight: bold;">{score:.2f}/1.00</span>'


def render_score_interpretation(metric_name: str, score: float):
    """Render an expandable score interpretation block for a given metric."""
    interpretations = {
        "NLI": {
            "green": (0.70, "Claims are well-supported by the source"),
            "yellow": (0.40, "Partial support -- check longer documents for truncation effects"),
            "red": "Possible contradictions detected between source and summary",
            "example": "Finance: If the source says 'revenue grew 12%' and the summary says 'revenue increased 12%', NLI should score high. If the summary says 'revenue doubled', NLI flags this.",
            "caveat": "Truncates input at ~400 words. For long documents (e.g., 10-K filings, full incident reports), faithfulness is only checked against the first portion."
        },
        "FactCC": {
            "green": (0.60, "Summary is factually consistent with the source"),
            "yellow": (0.30, "Borderline -- some claims may not be fully supported"),
            "red": "Factual inconsistencies detected",
            "example": "Accident: If the source reports '4 fatalities' but the summary says '6 people died', FactCC should flag this inconsistency.",
            "caveat": "Trained primarily on news data. Less reliable on highly technical or domain-specific text (financial filings, engineering reports)."
        },
        "AlignScore": {
            "green": (0.70, "Excellent alignment between source and summary"),
            "yellow": (0.50, "Moderate alignment -- some claims may diverge"),
            "red": "Poor alignment -- significant divergence from source",
            "example": "Finance: Correctly paraphrasing 'operating margin improved to 16.2% from 14.8%' as 'margins expanded by 1.4 percentage points' scores high.",
            "caveat": "Best single faithfulness metric (trained on 7 NLP tasks). Trust this when NLI and FactCC disagree."
        },
        "EntityCoverage": {
            "green": (0.80, "All key entities (names, dates, amounts) are present"),
            "yellow": (0.50, "Some important entities are missing"),
            "red": "Many key entities are missing from the summary",
            "example": "Finance: An earnings summary should mention the company name, revenue figure, and reporting period. Missing '$4.2B' or 'Q3 2024' would lower this score.",
            "caveat": "Doesn't understand synonyms or paraphrases -- 'the company' is not matched to 'Meridian Financial'. Only exact or near-exact entity matches count."
        },
        "SemanticCoverage": {
            "green": (0.50, "Good breadth -- summary represents the source well"),
            "yellow": (0.20, "Partial coverage -- some topics are missed"),
            "red": "Very narrow -- only a small fraction of the source is covered",
            "example": "Finance: A 5-sentence summary of a 200-sentence annual report = 2.5% coverage. That can still be excellent if it captures the 5 most important points.",
            "caveat": "Heavily influenced by source length. A 3-sentence summary covering 50 source sentences = 6%, which may actually be quite good. Always interpret alongside quality scores."
        },
        "BERTScoreRecall": {
            "green": (0.75, "Strong semantic recall from the source"),
            "yellow": (0.65, "Moderate recall -- some meaning is lost"),
            "red": "Low recall -- significant source content is not represented",
            "example": "Accident: If the source discusses cause, casualties, and response but the summary only covers casualties, recall will be low.",
            "caveat": "Token-level metric, complementary to sentence-level Semantic Coverage. Use both together for a complete picture."
        },
        "GEval": {
            "green": (8.0, "Excellent quality across the evaluated dimension"),
            "yellow": (5.0, "Acceptable but room for improvement"),
            "red": "Needs significant work",
            "example": "Finance: A coherent earnings summary presents results logically (revenue, then margins, then guidance). Jumping between topics randomly lowers the Coherence score.",
            "caveat": "Scores depend on the LLM model used. Different models may give different scores for the same summary. Consistency is best when using the same model."
        },
        "DAG": {
            "green": (5.0, "Excellent across all three checkpoints"),
            "yellow": (3.0, "Mixed -- some areas need improvement"),
            "red": "Major problems in one or more areas",
            "example": "Accident: Step 1 (Factual) checks if casualty numbers are correct. Step 2 (Complete) checks if cause and response are mentioned. Step 3 (Clear) checks if the sequence of events is understandable.",
            "caveat": "The 3-step breakdown (Factual/Complete/Clear, each 0-2) shows exactly where to focus improvements."
        },
        "Prometheus": {
            "green": (4.0, "Good overall quality"),
            "yellow": (3.0, "Acceptable but not strong"),
            "red": "Poor quality -- significant issues",
            "example": "Finance: A Prometheus score of 4-5 means the summary reads like something an analyst would write. A score of 1-2 means it has major gaps or errors.",
            "caveat": "Holistic teacher-like grading on a 1-5 scale. Considers accuracy, coverage, and readability together."
        },
        "BERTScore": {
            "green": (0.75, "High semantic match with the reference"),
            "yellow": (0.65, "Moderate match -- different phrasing or focus"),
            "red": "Low semantic match -- substantially different from reference",
            "example": "Finance: 'CEO resigned' vs 'leader stepped down' = high BERTScore. 'Revenue grew' vs 'costs declined' = lower score despite both being positive.",
            "caveat": "Measures meaning similarity, not factual correctness. A fluent but wrong summary can still score well against a wrong reference."
        },
        "MoverScore": {
            "green": (0.70, "Close semantic meaning to the reference"),
            "yellow": (0.40, "Some divergence in meaning or emphasis"),
            "red": "Very different meaning from the reference",
            "example": "Finance: Measures the 'effort' to transform your summary's meaning into the reference's meaning. Low effort = high score = similar content.",
            "caveat": "Complements BERTScore by measuring alignment differently (word mover's distance in embedding space)."
        },
        "ROUGE": {
            "green": (0.50, "Good word overlap with reference"),
            "yellow": (0.25, "Moderate overlap"),
            "red": "Low word overlap",
            "example": "Accident: If both summaries use the exact phrase '4 workers killed in explosion', ROUGE-2 captures this bigram match. Paraphrasing to 'explosion claimed 4 lives' would score lower.",
            "caveat": "Counts exact words only, ignoring meaning. 'Revenue increased' vs 'Sales grew' = zero ROUGE overlap despite identical meaning."
        },
        "BLEU": {
            "green": (0.30, "Good n-gram precision (high for summaries)"),
            "yellow": (0.15, "Acceptable overlap"),
            "red": "Low precision -- very different wording from reference",
            "example": "Finance: BLEU was designed for machine translation, so summary scores are naturally lower. A BLEU of 0.30+ is considered good for summarization.",
            "caveat": "Designed for translation evaluation; scores are naturally low for summaries. Don't compare BLEU scores to other metrics on a 0-1 scale."
        },
        "METEOR": {
            "green": (0.70, "Strong match including synonyms and stems"),
            "yellow": (0.40, "Moderate match"),
            "red": "Low match",
            "example": "Finance: METEOR recognizes that 'increased' and 'grew' are related, giving partial credit that ROUGE/BLEU would miss.",
            "caveat": "More forgiving than BLEU because it uses synonym matching and stemming. Generally gives higher scores than BLEU."
        },
        "chrF": {
            "green": (0.70, "Good character-level overlap"),
            "yellow": (0.40, "Moderate overlap"),
            "red": "Low character-level similarity",
            "example": "Finance: Catches partial word matches like 'profitability' vs 'profitable' that word-level metrics miss.",
            "caveat": "Character-level metric -- good for morphological variants and partial matches. Less interpretable than word-level metrics."
        },
        "Levenshtein": {
            "green": (0.70, "Very similar text (few edits needed)"),
            "yellow": (0.40, "Some editing required to match"),
            "red": "Very different text (many edits needed)",
            "example": "Accident: If both summaries are nearly word-for-word identical, Levenshtein will be very high. Heavy paraphrasing will lower it.",
            "caveat": "Raw edit distance ratio. Most useful for detecting near-duplicates or minor variations, less useful for evaluating paraphrased content."
        },
        "Perplexity": {
            "green": (0.70, "Natural, fluent language"),
            "yellow": (0.40, "Somewhat natural"),
            "red": "Unnatural or awkward phrasing",
            "example": "Finance: 'The company reported strong earnings' = low perplexity (natural). 'Earnings strong company reported the' = high perplexity (unnatural).",
            "caveat": "Based on GPT-2 fluency. Measures how 'natural' the text sounds, not whether it's truthful or accurate."
        },
    }

    info = interpretations.get(metric_name)
    if not info:
        return

    green_thresh, green_text = info["green"]
    yellow_thresh, yellow_text = info["yellow"]
    red_text = info["red"]

    # Map internal metric names to friendlier display labels
    display_names = {
        "BERTScoreRecall": "BERTScore Recall",
        "SemanticCoverage": "Semantic Coverage",
        "EntityCoverage": "Entity Coverage",
    }
    display_name = display_names.get(metric_name, metric_name)

    with st.expander(f"What does my {display_name} score mean?"):
        # Determine which tier the score falls into
        if score >= green_thresh:
            tier_color = "#28a745"
            tier_label = "Good"
            tier_text = green_text
        elif score >= yellow_thresh:
            tier_color = "#FEC925"
            tier_label = "Mixed"
            tier_text = yellow_text
        else:
            tier_color = "#dc3545"
            tier_label = "Needs Attention"
            tier_text = red_text

        st.markdown(f"""
        <div class="score-interpretation">
        <strong style="color: {tier_color};">{tier_label}:</strong> {tier_text}<br><br>
        <strong>Score guide:</strong>
        <span style="color: #28a745;">&ge;{green_thresh} = Good</span> |
        <span style="color: #FEC925;">{yellow_thresh}&ndash;{green_thresh} = Mixed</span> |
        <span style="color: #dc3545;">&lt;{yellow_thresh} = Needs Attention</span>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="edu-callout">
        <strong>Domain Example:</strong> {info["example"]}
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="caveat-box">
        <strong>Caveat:</strong> {info["caveat"]}
        </div>
        """, unsafe_allow_html=True)


def compute_summary_dashboard(results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """
    Compute summary metrics for the dashboard.

    Returns a dictionary with:
    - faithfulness_status: emoji + label
    - coverage_status: emoji + label + percentage
    - quality_status: emoji + label + score
    - recommendation: actionable advice
    """
    dashboard = {
        'faithfulness': {'emoji': '⚠️', 'label': 'Unknown', 'detail': ''},
        'coverage': {'emoji': '⚠️', 'label': 'Unknown', 'detail': ''},
        'quality': {'emoji': '⚠️', 'label': 'Unknown', 'detail': ''},
        'recommendation': ''
    }

    # Faithfulness assessment (from local metrics)
    if "faithfulness" in results:
        faith = results["faithfulness"]
        scores = []
        if "NLI" in faith and faith["NLI"].get('nli_score'):
            scores.append(faith["NLI"]['nli_score'])
        if "AlignScore" in faith and faith["AlignScore"].get('score'):
            scores.append(faith["AlignScore"]['score'])
        if "FactCC" in faith and faith["FactCC"].get('score'):
            scores.append(faith["FactCC"]['score'])

        if scores:
            avg = sum(scores) / len(scores)
            if avg >= 0.7:
                dashboard['faithfulness'] = {'emoji': '✅', 'label': 'Good', 'detail': f'{avg:.0%}'}
            elif avg >= 0.4:
                dashboard['faithfulness'] = {'emoji': '⚠️', 'label': 'Mixed', 'detail': f'{avg:.0%}'}
            else:
                dashboard['faithfulness'] = {'emoji': '❌', 'label': 'Low', 'detail': f'{avg:.0%}'}

    # Coverage assessment (from completeness_local)
    if "completeness_local" in results:
        comp = results["completeness_local"]
        if "SemanticCoverage" in comp and comp["SemanticCoverage"].get('score') is not None:
            cov_score = comp["SemanticCoverage"]['score']
            cov_sentences = comp["SemanticCoverage"].get('covered_sentences', 0)
            src_sentences = comp["SemanticCoverage"].get('source_sentences', 1)
            pct = f"{cov_sentences}/{src_sentences} sentences"

            if cov_score >= 0.5:
                dashboard['coverage'] = {'emoji': '✅', 'label': 'Good', 'detail': pct}
            elif cov_score >= 0.2:
                dashboard['coverage'] = {'emoji': '⚠️', 'label': 'Partial', 'detail': pct}
            else:
                dashboard['coverage'] = {'emoji': '❌', 'label': 'Low', 'detail': pct}

    # Quality assessment (from LLM metrics)
    if "completeness" in results:
        comp = results["completeness"]
        llm_scores = []
        for key in ['relevance', 'coherence', 'faithfulness', 'fluency']:
            if key in comp and comp[key].get('raw_score'):
                llm_scores.append(comp[key]['raw_score'])

        if llm_scores:
            avg = sum(llm_scores) / len(llm_scores)
            if avg >= 7:
                dashboard['quality'] = {'emoji': '✅', 'label': 'High', 'detail': f'{avg:.0f}/10'}
            elif avg >= 5:
                dashboard['quality'] = {'emoji': '⚠️', 'label': 'Medium', 'detail': f'{avg:.0f}/10'}
            else:
                dashboard['quality'] = {'emoji': '❌', 'label': 'Low', 'detail': f'{avg:.0f}/10'}

    # Generate recommendation
    recommendations = []
    if dashboard['coverage']['label'] == 'Low':
        recommendations.append("Consider adding more key points from the source")
    if dashboard['faithfulness']['label'] == 'Low':
        recommendations.append("Verify factual claims against the source")
    if dashboard['faithfulness']['label'] == 'Mixed':
        recommendations.append("Double-check specific facts mentioned")
    if not recommendations:
        if dashboard['quality']['label'] == 'High':
            recommendations.append("Summary looks good! Well-written and accurate.")
        else:
            recommendations.append("Review the detailed metrics below for improvement areas")

    dashboard['recommendation'] = recommendations[0] if recommendations else ""

    return dashboard


def display_summary_dashboard(results: Dict[str, Dict[str, Any]]):
    """Display the Summary at a Glance dashboard."""
    dashboard = compute_summary_dashboard(results)

    st.markdown("### Summary at a Glance")

    col1, col2, col3 = st.columns(3)

    with col1:
        f = dashboard['faithfulness']
        st.metric(
            label="Faithfulness",
            value=f"{f['emoji']} {f['label']}",
            delta=f['detail'] if f['detail'] else None,
            delta_color="off"
        )

    with col2:
        c = dashboard['coverage']
        st.metric(
            label="Coverage",
            value=f"{c['emoji']} {c['label']}",
            delta=c['detail'] if c['detail'] else None,
            delta_color="off"
        )

    with col3:
        q = dashboard['quality']
        st.metric(
            label="Quality",
            value=f"{q['emoji']} {q['label']}",
            delta=q['detail'] if q['detail'] else None,
            delta_color="off"
        )

    # Recommendation box
    if dashboard['recommendation']:
        st.info(f"💡 **Recommendation:** {dashboard['recommendation']}")

    # Educational note about metric types
    with st.expander("Understanding the Difference"):
        st.markdown("""
        | Metric Type | What It Measures | Example |
        |-------------|------------------|---------|
        | **Coverage** (Local) | What % of source is captured | "3 of 74 sentences covered" |
        | **Quality** (LLM) | Is what's in the summary good | "Well-written, accurate" |

        **Common Pattern:**
        - 📉 Low Coverage + ✅ High Quality = **Short but accurate** summary
        - 📈 High Coverage + ❌ Low Quality = **Comprehensive but flawed** summary
        - ✅ Both High = **Ideal** summary
        """)


def display_results(results: Dict[str, Dict[str, Any]]):
    """
    Display evaluation results in the new Part 1 / Part 2 structure.

    Part 1: Source-Based Evaluation (INTEGRITY) - Always shown
        - Faithfulness: Is the summary supported by the source?
        - Completeness: Does the summary cover all key points?

    Part 2: Reference-Based Evaluation (CONFORMANCE) - Only if reference provided
        - Semantic: Does it match the meaning/vibe of the reference?
        - Lexical: Does it match the exact words/structure?

    Args:
        results: Dictionary containing metric results.
    """
    # Check for token limit warnings
    source_text = st.session_state.get('source_text', '')
    word_count = len(source_text.split())
    show_token_warning = word_count > 400
    has_reference = st.session_state.get('has_reference', False)

    st.header("Evaluation Results")

    # Show truncation warning if source text exceeds 400 words
    if show_token_warning:
        st.warning(f"""
        ⚠️ **Text Truncation Notice** — Your source text has **{word_count:,} words**.

        Stage 1 faithfulness metrics (NLI, FactCC, AlignScore) truncate source text to ~400 words.
        This may affect accuracy when checking if the summary faithfully represents the full source.
        """)

    # ═══════════════════════════════════════════════════════════════════════════
    # STAGE 1: SOURCE vs SUMMARY (INTEGRITY CHECK)
    # ═══════════════════════════════════════════════════════════════════════════
    st.subheader("Stage 1: Integrity Check")
    st.markdown("""
    *Comparing the **Generated Summary** against the **Source Text** to verify faithfulness (no hallucinations)
    and completeness (key points captured).*
    """)

    # ─────────────────────────────────────────────────────────────────────────
    # FAITHFULNESS (Safety) - Detect hallucinations
    # ─────────────────────────────────────────────────────────────────────────
    if "faithfulness" in results and results["faithfulness"]:
        st.markdown("### Faithfulness -- *Can we trust this summary?*")
        st.markdown("""
        > **Why it matters:** A summary that "hallucinates" facts or contradicts the source is dangerous.
        > These metrics detect if the summary adds false information or misrepresents the source.
        """)

        faith_results = results["faithfulness"]

        # NLI Score
        col1, col2 = st.columns([1, 2])
        with col1:
            nli_score = faith_results.get("NLI", {})
            if nli_score.get('error') is None:
                score_val = nli_score.get('nli_score', 0)
                st.markdown(f"**1. NLI Score:** {format_score_display(score_val, 'general', 1.0)}", unsafe_allow_html=True)
            else:
                st.error("NLI Error")
        with col2:
            st.markdown("**Natural Language Inference** -- *Does the source logically support the summary?*")
            st.caption("Uses DeBERTa to check if claims in the summary can be inferred from the source. Score > 0.7 means 'entailed' (good), < 0.4 means potential contradiction.")
        if nli_score.get('error') is None:
            render_score_interpretation("NLI", nli_score.get('nli_score', 0))

        # FactCC Score
        st.markdown('<div class="metric-divider"></div>', unsafe_allow_html=True)
        if "FactCC" in faith_results:
            col1, col2 = st.columns([1, 2])
            with col1:
                factcc_score = faith_results.get("FactCC", {})
                if factcc_score.get('error') is None and factcc_score.get('score') is not None:
                    st.markdown(f"**2. FactCC:** {format_score_display(factcc_score['score'], 'general', 1.0)}", unsafe_allow_html=True)
                else:
                    st.warning("FactCC unavailable")
            with col2:
                st.markdown("**Factual Consistency Checker** -- *Are there factual errors?*")
                st.caption("A BERT model trained specifically to detect factual inconsistencies in summaries. Low scores flag potential errors.")
            if factcc_score.get('error') is None and factcc_score.get('score') is not None:
                render_score_interpretation("FactCC", factcc_score['score'])

        # AlignScore
        st.markdown('<div class="metric-divider"></div>', unsafe_allow_html=True)
        if "AlignScore" in faith_results:
            col1, col2 = st.columns([1, 2])
            with col1:
                align_score = faith_results.get("AlignScore", {})
                if align_score.get('error') is None and align_score.get('score') is not None:
                    st.markdown(f"**3. AlignScore:** {format_score_display(align_score['score'], 'general', 1.0)}", unsafe_allow_html=True)
                else:
                    st.warning("AlignScore unavailable")
            with col2:
                st.markdown("**Unified Alignment Model** -- *State-of-the-art factual consistency*")
                st.caption("**Recommended** -- Trained on 7 different NLP tasks. Currently the most reliable single metric for factual accuracy.")
            if align_score.get('error') is None and align_score.get('score') is not None:
                render_score_interpretation("AlignScore", align_score['score'])

        # Coverage Score (NER overlap)
        st.markdown('<div class="metric-divider"></div>', unsafe_allow_html=True)
        if "Coverage" in faith_results:
            col1, col2 = st.columns([1, 2])
            coverage_result = faith_results.get("Coverage", {})
            with col1:
                if coverage_result.get('error') is None and coverage_result.get('score') is not None:
                    st.markdown(f"**4. Entity Coverage:** {format_score_display(coverage_result['score'], 'general', 1.0)}", unsafe_allow_html=True)
                    st.caption(f"{coverage_result.get('covered_entities', 0)}/{coverage_result.get('source_entities', 0)} entities")
                else:
                    st.warning("Coverage unavailable")
            with col2:
                st.markdown("**Named Entity Coverage** -- *Are key names, places, dates mentioned?*")
                st.caption("Checks if important entities (people, organizations, locations, dates) from the source appear in the summary.")
                if coverage_result.get('missing_entities'):
                    with st.expander(f"Missing: {', '.join(coverage_result['missing_entities'][:3])}..."):
                        st.write(", ".join(coverage_result['missing_entities']))
            if coverage_result.get('error') is None and coverage_result.get('score') is not None:
                render_score_interpretation("EntityCoverage", coverage_result['score'])

        # Faithfulness Score Guide
        nli_val = faith_results.get("NLI", {}).get('nli_score', 0)
        factcc_val = faith_results.get("FactCC", {}).get('score', 0) if faith_results.get("FactCC", {}).get('error') is None else 0
        align_val = faith_results.get("AlignScore", {}).get('score', 0) if faith_results.get("AlignScore", {}).get('error') is None else 0
        avg_faith = (nli_val + factcc_val + align_val) / 3 if (nli_val and factcc_val and align_val) else 0

        if avg_faith >= 0.7:
            tier_class, icon = "good", "✅"
            msg = f"The summary appears well-supported by the source (avg: {avg_faith:.0%})"
        elif avg_faith >= 0.4:
            tier_class, icon = "mixed", "⚠️"
            msg = f"Some claims may need verification (avg: {avg_faith:.0%})"
        else:
            tier_class, icon = "bad", "❌"
            msg = f"Review carefully for potential errors (avg: {avg_faith:.0%})"
        truncation_note = ""
        if avg_faith < 0.7 and show_token_warning:
            truncation_note = f'<div class="assess-note"><strong>Note:</strong> Your source text ({word_count:,} words) exceeds the ~400 word limit for these models. The faithfulness metrics only evaluated the first ~400 words of the source, which may explain lower scores if key content appears later in the document.</div>'
        st.markdown(f'<div class="assessment-box {tier_class}">{icon} <span class="assess-title">Faithfulness Assessment:</span> {msg}{truncation_note}</div>', unsafe_allow_html=True)

    # ─────────────────────────────────────────────────────────────────────────
    # COMPLETENESS (Substance) - Key points captured
    # ─────────────────────────────────────────────────────────────────────────

    # Combined Completeness Section (Local + LLM metrics together)
    has_completeness_local = "completeness_local" in results and results["completeness_local"]
    has_completeness_llm = "completeness" in results and results["completeness"]

    if has_completeness_local or has_completeness_llm:
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        st.markdown("### Completeness -- *Did the summary capture what matters?*")
        st.markdown("""
        > **Why it matters:** A summary might be accurate but miss important information.
        > These metrics check if the key points from the source are represented.
        """)

        # Show Local Completeness Metrics first (matching Faithfulness style)
        if has_completeness_local:
            local_comp = results["completeness_local"]

            # Semantic Coverage
            if "SemanticCoverage" in local_comp:
                col1, col2 = st.columns([1, 2])
                with col1:
                    sc_result = local_comp["SemanticCoverage"]
                    if sc_result.get('error') is None and sc_result.get('score') is not None:
                        st.markdown(f"**1. Semantic Coverage:** {format_score_display(sc_result['score'], 'general', 1.0)}", unsafe_allow_html=True)
                        st.markdown(f"**Sentences:** {sc_result.get('covered_sentences', 0)}/{sc_result.get('source_sentences', 0)} covered")
                    else:
                        st.warning(f"{sc_result.get('error', 'No result')}")
                with col2:
                    st.markdown("**Sentence-Level Coverage** -- *How many source sentences are represented?*")
                    st.caption("**Recommended** -- Compares each source sentence to the summary using embeddings. Counts how many source sentences have a similar match (>0.7 similarity) in the summary.")
                if sc_result.get('error') is None and sc_result.get('score') is not None:
                    render_score_interpretation("SemanticCoverage", sc_result['score'])

            # BERTScore Recall
            if "BERTScoreRecall" in local_comp:
                st.markdown('<div class="metric-divider"></div>', unsafe_allow_html=True)
                col1, col2 = st.columns([1, 2])
                with col1:
                    bs_result = local_comp["BERTScoreRecall"]
                    if bs_result.get('error') is None and bs_result.get('recall') is not None:
                        st.markdown(f"**2. BERTScore Recall:** {format_score_display(bs_result['recall'], 'bertscore', 1.0)}", unsafe_allow_html=True)
                    else:
                        st.warning(f"{bs_result.get('error', 'No result')}")
                with col2:
                    st.markdown("**Meaning Recall** -- *What fraction of source meaning is captured?*")
                    st.caption("Measures what percentage of the source's semantic content appears in the summary. Low recall = missing content.")
                if bs_result.get('error') is None and bs_result.get('recall') is not None:
                    render_score_interpretation("BERTScoreRecall", bs_result['recall'])

        # Completeness Assessment Summary
        if has_completeness_local:
            local = results.get("completeness_local", {})
            sem_cov = local.get("SemanticCoverage", {})
            cov_score = sem_cov.get('score')

            if cov_score is not None:
                if cov_score >= 0.6:
                    tier_class, icon = "good", "✅"
                    msg = f"Good coverage of source content (Semantic Coverage: {cov_score:.0%})"
                elif cov_score >= 0.3:
                    tier_class, icon = "mixed", "⚠️"
                    msg = f"Partial coverage — some source content may be missing (Semantic Coverage: {cov_score:.0%})"
                else:
                    tier_class, icon = "bad", "❌"
                    msg = f"Low coverage — consider including more key points (Semantic Coverage: {cov_score:.0%})"
                st.markdown(f'<div class="assessment-box {tier_class}">{icon} <span class="assess-title">Completeness Assessment:</span> {msg}</div>', unsafe_allow_html=True)

                with st.expander("Why Coverage May Differ from Quality"):
                    # Get coverage info if available
                    cov_info = ""
                    if "completeness_local" in results:
                        local = results["completeness_local"]
                        if "SemanticCoverage" in local and local["SemanticCoverage"].get('score') is not None:
                            sc = local["SemanticCoverage"]
                            cov_sent = sc.get('covered_sentences', 0)
                            src_sent = sc.get('source_sentences', 1)
                            cov_info = f"Your summary covers **{cov_sent} of {src_sent}** source sentences ({sc['score']:.0%})."

                    st.markdown(f"""
                    {cov_info}

                    **Coverage metrics** (Semantic Coverage, BERTScore Recall) measure **breadth**:
                    - "What percentage of the source is represented in the summary?"
                    - A short summary will naturally have low coverage

                    **Holistic metrics** (below) measure **depth and quality**:
                    - "Are the important points included, and is the summary accurate and clear?"
                    - See the Holistic Assessment section for G-Eval, DAG, and Prometheus scores

                    **Common Interpretation:**
                    - Low coverage + High holistic scores = **Concise, focused summary** (often acceptable)
                    - High coverage + Low holistic scores = **Verbose, but may have issues** (needs review)
                    """)

                    st.markdown("""
                    <div class="edu-callout">
                    <strong>Finance example:</strong> A 10-K filing has 200+ sentences. A 5-sentence summary = 2.5% coverage,
                    but if those 5 sentences capture revenue, margins, guidance, risks, and outlook, holistic quality could be excellent.<br><br>
                    <strong>Accident example:</strong> An investigation report with 30 findings summarized into 5 critical points = 17% coverage,
                    but if those are the 5 most actionable findings, holistic quality could still score highly.
                    </div>
                    """, unsafe_allow_html=True)

    # ─────────────────────────────────────────────────────────────────────────
    # HOLISTIC ASSESSMENT - Metrics that evaluate both faithfulness AND completeness
    # ─────────────────────────────────────────────────────────────────────────

    has_holistic = "completeness" in results and results["completeness"]
    if has_holistic:
        comp_results = results["completeness"]
        has_dag = "dag" in comp_results and comp_results.get("dag", {}).get('error') is None
        has_prometheus = "prometheus" in comp_results and comp_results.get("prometheus", {}).get('error') is None
        has_geval_faith = "faithfulness" in comp_results and comp_results.get("faithfulness", {}).get('error') is None
        has_geval_rel = "relevance" in comp_results and comp_results.get("relevance", {}).get('error') is None
        has_geval_coh = "coherence" in comp_results and comp_results.get("coherence", {}).get('error') is None
        has_geval_flu = "fluency" in comp_results and comp_results.get("fluency", {}).get('error') is None
        has_any_holistic = has_dag or has_prometheus or has_geval_faith or has_geval_rel or has_geval_coh or has_geval_flu

        if has_any_holistic:
            st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
            st.markdown("### Holistic Assessment -- *Faithfulness + Completeness*")
            st.markdown("""
            > These metrics evaluate **both faithfulness and completeness together** — checking that
            > the summary is factually accurate and captures the key points from the source.
            """)

            # 1. DAG results
            if has_dag:
                dag_result = comp_results.get("dag", {})
                st.markdown("**1. DAG** -- *Decision Tree: A 3-step checklist*")
                with st.expander("What is DAG?"):
                    st.markdown("""
                    **DAG** evaluates summaries like a decision tree with 3 checkpoints:

                    | Step | Question | Points | Finance Example | Accident Example |
                    |------|----------|--------|-----------------|-----------------|
                    | 1. Factual | "Does it only state facts from the source?" | 0-2 | Are revenue/margin numbers correct? | Are casualty counts and dates accurate? |
                    | 2. Complete | "Are the main points included?" | 0-2 | Revenue + margins + guidance covered? | Cause + impact + response covered? |
                    | 3. Clear | "Is it easy to understand?" | 0-2 | Logical financial narrative? | Clear chronological sequence? |

                    **Scoring:** 6/6 = Perfect | 4-5 = Good | 2-3 = Issues | 0-1 = Major problems
                    """)
                col1, col2 = st.columns([1, 2])
                with col1:
                    if dag_result.get('score') is not None:
                        raw_score = dag_result.get('raw_score', 0)
                        color = "#28a745" if raw_score >= 5 else "#FEC925" if raw_score >= 3 else "#dc3545"
                        st.markdown(f"**Score:** <span style='color: {color}; font-weight: bold;'>{raw_score}/6</span>", unsafe_allow_html=True)
                        step1 = dag_result.get('step1_factual', 'N/A')
                        step2 = dag_result.get('step2_completeness', 'N/A')
                        step3 = dag_result.get('step3_clarity', 'N/A')
                        st.caption(f"Factual: {step1}/2 | Complete: {step2}/2 | Clear: {step3}/2")
                with col2:
                    st.caption("**Recommended** -- Combines factual accuracy (Step 1), key point coverage (Step 2), and clarity (Step 3) into one structured evaluation.")
                if dag_result.get('raw_score') is not None:
                    render_score_interpretation("DAG", dag_result['raw_score'])

            # 2. Prometheus results
            if has_prometheus:
                st.markdown('<div class="metric-divider"></div>', unsafe_allow_html=True)
                prom_result = comp_results.get("prometheus", {})
                st.markdown("**2. Prometheus** -- *LLM Judge (1-5 scale)*")
                with st.expander("What is Prometheus?"):
                    st.markdown("""
                    **Prometheus** is an LLM-as-a-judge metric that grades summaries on a **1-5 rubric**,
                    much like a teacher grading an essay with a detailed scoring guide.

                    | Score | Meaning | What It Looks Like |
                    |-------|---------|-------------------|
                    | **5** | Excellent | Accurate, complete, clear -- ready to publish |
                    | **4** | Good | Minor omissions or phrasing issues |
                    | **3** | Acceptable | Covers basics but misses important details |
                    | **2** | Poor | Significant gaps or inaccuracies |
                    | **1** | Very Poor | Fails to represent the source meaningfully |

                    **How it works:** The LLM receives a detailed rubric describing each score level,
                    reads both the source and summary, then assigns a score with justification.

                    **Prometheus vs G-Eval:** G-Eval scores 4 separate dimensions (1-10 each).
                    Prometheus gives one holistic score (1-5) considering everything together --
                    useful as a quick overall quality check.
                    """)
                    st.markdown("""
                    <div class="caveat-box">
                    <strong>Caveat:</strong> Like all LLM-based metrics, scores depend on the model selected.
                    Different models may assign different scores for the same summary. For fair comparisons,
                    always use the same model.
                    </div>
                    """, unsafe_allow_html=True)
                col1, col2 = st.columns([1, 2])
                with col1:
                    if prom_result.get('score') is not None:
                        raw_score = prom_result.get('raw_score', prom_result['score'])
                        st.markdown(f"**Score:** {format_score_display(raw_score, 'prometheus', 5.0)}", unsafe_allow_html=True)
                with col2:
                    st.caption("**Recommended** -- Holistic quality assessment considering all aspects. 5 = Excellent | 4 = Good | 3 = Acceptable | 2 = Poor | 1 = Very Poor")
                if prom_result.get('raw_score') is not None:
                    render_score_interpretation("Prometheus", prom_result['raw_score'])

            # 3. G-Eval (grouped as 3a-3d)
            has_any_geval = has_geval_faith or has_geval_rel or has_geval_coh or has_geval_flu
            if has_any_geval:
                st.markdown('<div class="metric-divider"></div>', unsafe_allow_html=True)
                st.markdown("**3. G-Eval** -- *LLM-based quality dimensions (1-10 scale)*")
                with st.expander("What is G-Eval?"):
                    st.markdown("""
                    **G-Eval** uses a large language model (LLM) to evaluate text like a human expert would.
                    It uses chain-of-thought prompting to reason about each dimension before assigning a score (1-10).

                    | Dimension | What It Checks | Example |
                    |-----------|---------------|---------|
                    | **Faithfulness** | Are all claims traceable to the source? | Revenue numbers match the filing? |
                    | **Relevance** | Are the important points included? | Key findings covered? |
                    | **Coherence** | Does it flow logically? | Ideas connect without abrupt jumps? |
                    | **Fluency** | Is it well-written? | Grammar and word choice natural? |

                    **Interpreting Scores:** 9-10: Excellent | 7-8: Good | 5-6: Acceptable | Below 5: Needs improvement
                    """)
                    st.markdown("""
                    <div class="caveat-box">
                    <strong>Caveat:</strong> Scores depend on the LLM model selected. Different models may give different scores
                    for the same summary. For consistency, always compare results using the same model.
                    </div>
                    """, unsafe_allow_html=True)

                # 3a. G-Eval: Faithfulness
                if has_geval_faith:
                    faith_result = comp_results.get("faithfulness", {})
                    col1, col2 = st.columns([1, 2])
                    with col1:
                        if faith_result.get('score') is not None:
                            raw_score = faith_result.get('raw_score', faith_result['score'] * 10)
                            st.markdown(f"**3a. Faithfulness:** {format_score_display(raw_score, 'geval', 10.0)}", unsafe_allow_html=True)
                        else:
                            st.warning(f"{faith_result.get('error', 'No result')}")
                    with col2:
                        st.markdown("**Source Alignment** -- *Can every claim be traced to the source?*")
                        st.caption("LLM reads both texts and verifies each summary claim against the source.")

                # 3b. G-Eval: Relevance
                if has_geval_rel:
                    rel_result = comp_results.get("relevance", {})
                    col1, col2 = st.columns([1, 2])
                    with col1:
                        if rel_result.get('score') is not None:
                            raw_score = rel_result.get('raw_score', rel_result['score'] * 10)
                            st.markdown(f"**3b. Relevance:** {format_score_display(raw_score, 'geval', 10.0)}", unsafe_allow_html=True)
                        else:
                            st.warning(f"{rel_result.get('error', 'No result')}")
                    with col2:
                        st.markdown("**Main Points Check** -- *Are the important points from the source included?*")
                        st.caption("Checks whether the summary captures the key information from the source document.")

                # 3c. G-Eval: Coherence
                if has_geval_coh:
                    coh_result = comp_results.get("coherence", {})
                    col1, col2 = st.columns([1, 2])
                    with col1:
                        if coh_result.get('score') is not None:
                            raw_score = coh_result.get('raw_score', coh_result['score'] * 10)
                            st.markdown(f"**3c. Coherence:** {format_score_display(raw_score, 'geval', 10.0)}", unsafe_allow_html=True)
                        else:
                            st.warning(f"{coh_result.get('error', 'No result')}")
                    with col2:
                        st.markdown("**Logical Flow** -- *Does it flow logically from start to finish?*")
                        st.caption("Checks if ideas connect naturally without abrupt jumps or contradictions.")

                # 3d. G-Eval: Fluency
                if has_geval_flu:
                    flu_result = comp_results.get("fluency", {})
                    col1, col2 = st.columns([1, 2])
                    with col1:
                        if flu_result.get('score') is not None:
                            raw_score = flu_result.get('raw_score', flu_result['score'] * 10)
                            st.markdown(f"**3d. Fluency:** {format_score_display(raw_score, 'geval', 10.0)}", unsafe_allow_html=True)
                        else:
                            st.warning(f"{flu_result.get('error', 'No result')}")
                    with col2:
                        st.markdown("**Writing Quality** -- *Is it grammatically correct and natural?*")
                        st.caption("Evaluates grammar, word choice, and overall readability.")

            # Consolidated G-Eval score interpretation (Faithfulness, Relevance, Coherence, Fluency)
            if has_geval_faith or has_geval_rel or has_geval_coh or has_geval_flu:
                with st.expander("What do my G-Eval scores mean?"):
                    geval_dimensions = [
                        ("Faithfulness", comp_results.get("faithfulness", {})),
                        ("Relevance", comp_results.get("relevance", {})),
                        ("Coherence", comp_results.get("coherence", {})),
                        ("Fluency", comp_results.get("fluency", {})),
                    ]
                    for dim_name, dim_result in geval_dimensions:
                        raw = dim_result.get('raw_score')
                        if raw is not None:
                            if raw >= 8.0:
                                color, tier = "#28a745", "Good"
                            elif raw >= 5.0:
                                color, tier = "#FEC925", "Mixed"
                            else:
                                color, tier = "#dc3545", "Needs Attention"
                            st.markdown(
                                f"**{dim_name}** ({raw:.1f}/10): "
                                f"<span style='color: {color}; font-weight: bold;'>{tier}</span>",
                                unsafe_allow_html=True
                            )

                    st.markdown("""
                    <div class="score-interpretation">
                    <strong>Score guide:</strong>
                    <span style="color: #28a745;">&ge;8 = Good</span> |
                    <span style="color: #FEC925;">5&ndash;8 = Mixed</span> |
                    <span style="color: #dc3545;">&lt;5 = Needs Attention</span>
                    </div>
                    """, unsafe_allow_html=True)

                    st.markdown("""
                    <div class="edu-callout">
                    <strong>Domain Example:</strong> A coherent earnings summary presents results logically (revenue, then margins, then guidance). Jumping between topics randomly lowers the Coherence score.
                    </div>
                    """, unsafe_allow_html=True)

                    st.markdown("""
                    <div class="caveat-box">
                    <strong>Caveat:</strong> Scores depend on the LLM model used. Different models may give different scores for the same summary. Consistency is best when using the same model.
                    </div>
                    """, unsafe_allow_html=True)

            # Holistic Assessment Summary
            summary_scores = []
            # Include G-Eval scores (normalized to 10-point scale)
            for geval_key in ['faithfulness', 'relevance', 'coherence', 'fluency']:
                geval_result = comp_results.get(geval_key, {})
                if geval_result.get('raw_score') is not None:
                    summary_scores.append((geval_key.title(), geval_result['raw_score'], 10))
            if has_dag and dag_result.get('raw_score') is not None:
                summary_scores.append(('DAG', dag_result['raw_score'], 6))
            if has_prometheus and prom_result.get('raw_score') is not None:
                summary_scores.append(('Prometheus', prom_result['raw_score'], 5))

            if summary_scores:
                # Calculate weighted average
                total_weighted = sum(s[1]/s[2] for s in summary_scores)
                avg_holistic = total_weighted / len(summary_scores)

                if avg_holistic >= 0.75:
                    tier_class, icon = "good", "✅"
                    msg = "The summary scores well across all dimensions (accuracy, coverage, clarity)"
                elif avg_holistic >= 0.5:
                    tier_class, icon = "mixed", "⚠️"
                    msg = "The summary has room for improvement in some areas"
                else:
                    tier_class, icon = "bad", "❌"
                    msg = "The summary may need significant revision"
                st.markdown(f'<div class="assessment-box {tier_class}">{icon} <span class="assess-title">Holistic Assessment:</span> {msg}</div>', unsafe_allow_html=True)

    # ═══════════════════════════════════════════════════════════════════════════
    # STAGE 2: GENERATED vs. REFERENCE SUMMARY (CONFORMANCE)
    # Only shown if reference summary was provided
    # ═══════════════════════════════════════════════════════════════════════════
    if has_reference and (("semantic" in results and results["semantic"]) or ("lexical" in results and results["lexical"])):
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        st.subheader("Stage 2: Conformance Check")
        st.markdown("""
        *Comparing your **Generated Summary** against a **Reference Summary** (human-written gold standard)
        to measure both semantic similarity (same meaning) and lexical overlap (same words).*
        """)

        with st.expander("Understanding Stage 2: Semantic vs Lexical"):
            st.markdown("""
            **When to focus on semantic metrics** (BERTScore, MoverScore):
            - General quality assessment where paraphrasing is acceptable
            - Creative or editorial summaries where different wording is fine
            - Example: "EBITDA margin" vs "profitability metric" -- same meaning, different words

            **When to focus on lexical metrics** (ROUGE, BLEU, chrF++):
            - Regulatory or compliance contexts where exact terminology matters
            - Technical reports where specific terms must be preserved
            - Example: A safety report must say "confined space" not "small area"

            **Best practice:** Use both together. High semantic + low lexical = good paraphrasing.
            High lexical + low semantic = copied words but missed the point (rare but possible).
            """)

        # ─────────────────────────────────────────────────────────────────────
        # SEMANTIC CONFORMANCE (Vibe/Meaning)
        # ─────────────────────────────────────────────────────────────────────
        if "semantic" in results and results["semantic"]:
            st.markdown("### Semantic Conformance -- *Same meaning, different words?*")
            st.markdown("""
            These metrics understand synonyms and paraphrasing. "The CEO resigned" and
            "The company's leader stepped down" would score high because the meaning is the same.
            """)

            with st.expander("ℹ️ Understanding Semantic Metrics"):
                st.markdown("""
                **BERTScore** uses AI embeddings to compare meanings:
                - **Precision**: "How much of my summary is relevant to the reference?"
                - **Recall**: "How much of the reference did my summary capture?"
                - **F1**: The balanced average of both (your main number)

                **MoverScore** measures the "effort" to transform one meaning into another.
                Think of it like: "How much would I need to change my summary to make it identical to the reference?"
                """)

            sem_results = results["semantic"]
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**1. BERTScore**")
                st.caption("Semantic similarity via embeddings")

                bert_scores = sem_results.get("BERTScore", {})
                if bert_scores.get('error') is None:
                    st.markdown(f"- Precision: {format_score_display(bert_scores.get('precision', 0), 'bertscore')}", unsafe_allow_html=True)
                    st.markdown(f"- Recall: {format_score_display(bert_scores.get('recall', 0), 'bertscore')}", unsafe_allow_html=True)
                    st.markdown(f"- F1: {format_score_display(bert_scores.get('f1', 0), 'bertscore')}", unsafe_allow_html=True)
                else:
                    st.error(f"Error: {bert_scores['error']}")

            with col2:
                st.markdown("**2. MoverScore**")
                st.caption("Semantic alignment distance")

                mover_score = sem_results.get("MoverScore", {})
                if mover_score.get('error') is None:
                    st.markdown(f"- Score: {format_score_display(mover_score.get('moverscore', 0))}", unsafe_allow_html=True)
                else:
                    st.error(f"Error: {mover_score['error']}")

            # Combined semantic score interpretation
            if bert_scores.get('error') is None or mover_score.get('error') is None:
                with st.expander("What do my BERTScore & MoverScore mean?"):
                    # Show actual scores with color coding
                    if bert_scores.get('error') is None:
                        bert_f1 = bert_scores.get('f1', 0)
                        if bert_f1 >= 0.75:
                            b_color, b_tier = "#28a745", "Good"
                        elif bert_f1 >= 0.5:
                            b_color, b_tier = "#FEC925", "Mixed"
                        else:
                            b_color, b_tier = "#dc3545", "Needs Attention"
                        st.markdown(
                            f"**BERTScore F1** ({bert_f1:.3f}): "
                            f"<span style='color: {b_color}; font-weight: bold;'>{b_tier}</span>",
                            unsafe_allow_html=True
                        )
                    if mover_score.get('error') is None:
                        mover_val = mover_score.get('moverscore', 0)
                        if mover_val >= 0.75:
                            m_color, m_tier = "#28a745", "Good"
                        elif mover_val >= 0.5:
                            m_color, m_tier = "#FEC925", "Mixed"
                        else:
                            m_color, m_tier = "#dc3545", "Needs Attention"
                        st.markdown(
                            f"**MoverScore** ({mover_val:.3f}): "
                            f"<span style='color: {m_color}; font-weight: bold;'>{m_tier}</span>",
                            unsafe_allow_html=True
                        )

                    st.markdown("""
                    ---
                    **What each measures:**

                    | Metric | Approach | Analogy |
                    |--------|----------|---------|
                    | **BERTScore** | Token-level embedding similarity (precision/recall/F1) | "How much meaning overlaps between the two texts?" |
                    | **MoverScore** | Optimal transport distance between embeddings | "How much effort to transform one meaning into the other?" |

                    **Why both matter:** BERTScore checks token-by-token overlap and can miss issues
                    where the same words appear in a different order or context. MoverScore captures
                    semantic flow differences by measuring the minimum "cost" to rearrange meaning,
                    so it catches re-ordering and structural issues that BERTScore may miss.
                    """)

                    st.markdown("""
                    <div class="score-interpretation">
                    <strong>Score guide:</strong>
                    <span style="color: #28a745;">&ge;0.75 = Good</span> |
                    <span style="color: #FEC925;">0.50&ndash;0.75 = Mixed</span> |
                    <span style="color: #dc3545;">&lt;0.50 = Needs Attention</span>
                    </div>
                    """, unsafe_allow_html=True)

                    st.markdown("""
                    <div class="edu-callout">
                    <strong>Domain Examples:</strong><br>
                    <strong>Finance:</strong> "EBITDA margin expanded 200bps" vs "profitability improved significantly" —
                    BERTScore captures the shared concept of improvement; MoverScore penalises the loss of the specific metric.<br><br>
                    <strong>Accident report:</strong> "3 fatalities occurred before emergency response" vs "emergency response followed 3 fatalities" —
                    BERTScore sees the same tokens; MoverScore detects the subtle causal reordering.
                    </div>
                    """, unsafe_allow_html=True)

        # ─────────────────────────────────────────────────────────────────────
        # LEXICAL CONFORMANCE (Format/Structure)
        # ─────────────────────────────────────────────────────────────────────
        if "lexical" in results and results["lexical"]:
            st.markdown('<div class="gold-divider"></div>', unsafe_allow_html=True)
            st.markdown("### Lexical Conformance -- *Same words and structure?*")
            st.markdown("""
            These metrics count exact word matches. Useful for checking if the summary
            uses required terminology, follows a specific format, or matches brand voice.
            """)

            with st.expander("ℹ️ Understanding Lexical Metrics"):
                st.markdown("""
                **ROUGE** (Recall-Oriented Understudy for Gisting Evaluation):
                - **ROUGE-1**: Single word overlap (unigrams) — "How many individual words match?"
                - **ROUGE-2**: Two-word phrase overlap (bigrams) — "How many word pairs match?"
                - **ROUGE-L**: Longest common subsequence — "What's the longest matching sequence?"

                **BLEU** (Bilingual Evaluation Understudy):
                - Originally designed for machine translation
                - Measures n-gram precision with a brevity penalty
                - Scores tend to be lower (0.3+ is good for summaries)

                **METEOR**: Considers stemming and synonyms (more forgiving than BLEU)

                **chrF++**: Character-level F-score (handles morphology well)

                **Levenshtein**: Edit distance — "How many changes needed to match?"

                **Perplexity**: Measures fluency — "How natural does the text sound?"
                """)

            lex_results = results["lexical"]
            col1, col2, col3 = st.columns(3)

            with col1:
                st.markdown("**1. ROUGE Scores**")
                rouge_scores = lex_results.get("ROUGE", {})
                if rouge_scores.get('error') is None:
                    st.markdown(f"- ROUGE-1: {format_score_display(rouge_scores.get('rouge1', 0))}", unsafe_allow_html=True)
                    st.markdown(f"- ROUGE-2: {format_score_display(rouge_scores.get('rouge2', 0))}", unsafe_allow_html=True)
                    st.markdown(f"- ROUGE-L: {format_score_display(rouge_scores.get('rougeL', 0))}", unsafe_allow_html=True)
                else:
                    st.error(f"Error: {rouge_scores['error']}")

            with col2:
                st.markdown("**2. BLEU Score**")
                bleu_score = lex_results.get("BLEU", {})
                if bleu_score.get('error') is None:
                    st.markdown(f"- BLEU: {format_score_display(bleu_score.get('bleu', 0), 'bleu')}", unsafe_allow_html=True)
                else:
                    st.error(f"Error: {bleu_score['error']}")

                st.markdown("**3. METEOR Score**")
                meteor_score = lex_results.get("METEOR", {})
                if meteor_score.get('error') is None:
                    st.markdown(f"- METEOR: {format_score_display(meteor_score.get('meteor', 0))}", unsafe_allow_html=True)
                else:
                    st.error(f"Error: {meteor_score['error']}")

                st.markdown("**4. chrF++ Score**")
                chrf_score = lex_results.get("chrF++", {})
                if chrf_score.get('error') is None:
                    st.markdown(f"- chrF++: {format_score_display(chrf_score.get('chrf', 0))}", unsafe_allow_html=True)
                else:
                    st.error(f"Error: {chrf_score['error']}")

            with col3:
                st.markdown("**5. Levenshtein Similarity**")
                lev_score = lex_results.get("Levenshtein", {})
                if lev_score.get('error') is None:
                    st.markdown(f"- Similarity: {format_score_display(lev_score.get('levenshtein', 0))}", unsafe_allow_html=True)
                else:
                    st.error(f"Error: {lev_score['error']}")

                st.markdown("**6. Perplexity (Fluency)**")
                perp_score = lex_results.get("Perplexity", {})
                if perp_score.get('error') is None:
                    st.markdown(f"- Fluency: {format_score_display(perp_score.get('normalized_score', 0))}", unsafe_allow_html=True)
                else:
                    st.warning(f"{perp_score.get('error', 'N/A')}")

            # Single combined interpretation for all lexical metrics
            with st.expander("What do my lexical scores mean?"):
                if rouge_scores.get('error') is None:
                    r1 = rouge_scores.get('rouge1', 0)
                    tier = "Good" if r1 >= 0.50 else "Moderate" if r1 >= 0.25 else "Low"
                    st.markdown(f"**ROUGE-1** ({r1:.2f}): {tier} word overlap with reference. Counts exact word matches only -- ignores meaning.")

                if bleu_score.get('error') is None:
                    bl = bleu_score.get('bleu', 0)
                    tier = "Good (high for summaries)" if bl >= 0.30 else "Acceptable" if bl >= 0.15 else "Low"
                    st.markdown(f"**BLEU** ({bl:.2f}): {tier}. Designed for translation -- summary scores are naturally low. 0.30+ is good.")

                if meteor_score.get('error') is None:
                    mt = meteor_score.get('meteor', 0)
                    tier = "Strong" if mt >= 0.70 else "Moderate" if mt >= 0.40 else "Low"
                    st.markdown(f"**METEOR** ({mt:.2f}): {tier}. More forgiving than BLEU -- uses synonym matching and stemming.")

                if chrf_score.get('error') is None:
                    ch = chrf_score.get('chrf', 0)
                    tier = "Good" if ch >= 0.70 else "Moderate" if ch >= 0.40 else "Low"
                    st.markdown(f"**chrF++** ({ch:.2f}): {tier}. Character-level -- catches partial word matches like 'profitable' vs 'profitability'.")

                if lev_score.get('error') is None:
                    lv = lev_score.get('levenshtein', 0)
                    tier = "Very similar" if lv >= 0.70 else "Some differences" if lv >= 0.40 else "Very different"
                    st.markdown(f"**Levenshtein** ({lv:.2f}): {tier}. Raw edit distance -- how many character changes to match the reference.")

                if perp_score.get('error') is None:
                    pp = perp_score.get('normalized_score', 0)
                    tier = "Natural" if pp >= 0.70 else "Somewhat natural" if pp >= 0.40 else "Awkward phrasing"
                    st.markdown(f"**Perplexity** ({pp:.2f}): {tier}. GPT-2 fluency score -- measures how natural the text sounds, not accuracy.")

                st.markdown("""
                <div class="caveat-box">
                <strong>Note:</strong> Lexical metrics count surface-level overlap (words, characters, n-grams).
                They cannot detect paraphrases -- "revenue grew" vs "sales increased" scores zero overlap
                despite identical meaning. Always use alongside semantic metrics.
                </div>
                """, unsafe_allow_html=True)

    elif not has_reference:
        st.markdown('<div class="gold-divider"></div>', unsafe_allow_html=True)
        st.info("**Stage 2 (Conformance)** skipped -- no reference summary provided. Add a reference summary to enable ROUGE, BLEU, BERTScore comparisons.")

    # ── LLM-as-a-Judge (optional) ────────────────────────────────────────
    if H2OGPTE_AVAILABLE:
        st.markdown('<div class="gold-divider"></div>', unsafe_allow_html=True)
        st.subheader("LLM-as-a-Judge")
        st.caption(
            "Customize the prompt template for LLM-as-a-Judge evaluation. "
            "Available placeholders: **{PROMPT}** for the input question, "
            "**{PREDICTED_TEXT}** for the model's response, and "
            "**{TARGET_TEXT}** for reference answers (ground truth). "
            'Make sure to specify the scoring scale and criteria in the prompt template '
            '(e.g. "Score the response on a scale of 1 to 10").'
        )

        @st.fragment
        def _custom_judge_fragment():
            DEFAULT_JUDGE_TEMPLATE = """\
[Instruction]
Evaluate the summary against the source document (and reference if provided) on a scale of 1 (worst) to 10 (best). Consider faithfulness, completeness, and clarity.

Be specific in your explanation: cite any missing key points, unsupported claims, or errors by name. Avoid vague statements.

[Source Document]
{PROMPT}

[Reference Summary]
{TARGET_TEXT}

[Summary to Evaluate]
{PREDICTED_TEXT}"""

            criteria_text = st.text_area(
                "LLM-as-a-Judge Prompt Template",
                value=DEFAULT_JUDGE_TEMPLATE,
                height=300,
                key="custom_judge_criteria_input"
            )
            st.session_state.custom_judge_criteria = criteria_text

            # Run button
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                run_judge = st.button(
                    "⚖️ Run LLM-as-a-Judge",
                    type="secondary",
                    use_container_width=True,
                    disabled=not criteria_text.strip(),
                    key="run_custom_judge"
                )

            if run_judge:
                with st.spinner("Running LLM-as-a-Judge..."):
                    source_text = st.session_state.get('source_text', '')
                    summary_text = st.session_state.get('summary_text', '')
                    reference_text = st.session_state.get('reference_text', '')

                    result = evaluate_custom(
                        summary=summary_text,
                        source=source_text,
                        criteria=criteria_text,
                        reference_summary=reference_text,
                        model_name=st.session_state.selected_model,
                        timeout=90
                    )
                    st.session_state.custom_judge_result = result

            # Display results
            cj_result = st.session_state.custom_judge_result
            if cj_result is not None:
                if cj_result.get('error') and cj_result.get('score') is None:
                    st.error(f"Custom Judge error: {cj_result['error']}")
                else:
                    raw = cj_result.get('raw_score', 0)
                    explanation = cj_result.get('explanation', 'No explanation provided')

                    res_col1, res_col2 = st.columns([1, 3])
                    with res_col1:
                        score_html = format_score_display(raw, metric_type="geval", max_score=10.0)
                        st.markdown(f"**Score:** {score_html}", unsafe_allow_html=True)
                    with res_col2:
                        st.markdown(f"**Explanation:** {explanation}")

                    if raw >= 7:
                        st.success(f"**Good** — The summary scores {int(raw)}/10 on your custom criteria.")
                    elif raw >= 4:
                        st.warning(f"**Mixed** — The summary scores {int(raw)}/10 on your custom criteria. Review the explanation for areas to improve.")
                    else:
                        st.error(f"**Poor** — The summary scores {int(raw)}/10 on your custom criteria. Significant gaps identified.")

                    st.markdown("""
                    <div class="caveat-box">
                    <strong>Note:</strong> Custom judge scores depend on the selected LLM model and prompt phrasing.
                    Different models or slightly different criteria may produce different scores. Use as a directional signal alongside built-in metrics.
                    </div>
                    """, unsafe_allow_html=True)

        _custom_judge_fragment()

    # Batch evaluation button (show at end of results if dataset uploaded)
    if st.session_state.uploaded_dataset is not None and \
       st.session_state.source_column and st.session_state.summary_column and \
       st.session_state.columns_selected and H2OGPTE_AVAILABLE:
        st.markdown('<div class="gold-divider"></div>', unsafe_allow_html=True)
        st.subheader("Batch Evaluation")
        st.caption("Evaluate the entire dataset with API metrics (G-Eval, DAG, Prometheus)")

        def start_batch_evaluation_main():
            st.session_state.start_batch_eval = True

        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.button(
                "🚀 Evaluate Entire Dataset",
                type="secondary",
                use_container_width=True,
                key="batch_eval_main",
                on_click=start_batch_evaluation_main
            )


def batch_evaluate_dataset(df: pd.DataFrame, source_col: str, reference_col: str, summary_col: str, model_name: str, progress_bar, status_text, preview_placeholder=None, custom_criteria: str = ""):
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

    # Custom judge columns (only if criteria provided)
    has_custom = bool(custom_criteria and custom_criteria.strip())
    if has_custom:
        results_df['custom_judge_score'] = None
        results_df['custom_judge_explanation'] = None

    total_rows = len(df)

    for row_num, (idx, row) in enumerate(df.iterrows(), start=1):
        source_text = str(row[source_col])
        summary_text = str(row[summary_col])
        reference_text = str(row[reference_col]) if reference_col else ""

        # Update progress
        progress_bar.progress(row_num / total_rows)
        status_text.text(f"Processing row {row_num}/{total_rows}")

        try:
            # G-Eval Faithfulness (this serves as our fact-checking metric)
            faithfulness_result = evaluate_faithfulness(
                summary=summary_text,
                source=source_text,
                model_name=model_name
            )
            results_df.at[idx, 'geval_faithfulness'] = faithfulness_result.get('score', None)

            # G-Eval Coherence
            coherence_result = evaluate_coherence(
                summary=summary_text,
                source=source_text,
                model_name=model_name
            )
            results_df.at[idx, 'geval_coherence'] = coherence_result.get('score', None)

            # G-Eval Relevance
            relevance_result = evaluate_relevance(
                summary=summary_text,
                source=source_text,
                model_name=model_name
            )
            results_df.at[idx, 'geval_relevance'] = relevance_result.get('score', None)

            # G-Eval Fluency
            fluency_result = evaluate_fluency(
                summary=summary_text,
                source=source_text,
                model_name=model_name
            )
            results_df.at[idx, 'geval_fluency'] = fluency_result.get('score', None)

            # DAG
            dag_result = evaluate_dag(
                summary=summary_text,
                source=source_text,
                model_name=model_name
            )
            results_df.at[idx, 'dag_score'] = dag_result.get('raw_score', None)

            # Prometheus
            prometheus_result = evaluate_prometheus(
                summary=summary_text,
                reference_summary=reference_text,
                model_name=model_name
            )
            results_df.at[idx, 'prometheus_score'] = prometheus_result.get('score', None)

            # Custom Judge (only if criteria provided)
            if has_custom:
                custom_result = evaluate_custom(
                    summary=summary_text,
                    source=source_text,
                    criteria=custom_criteria,
                    reference_summary=reference_text if reference_text else None,
                    model_name=model_name
                )
                results_df.at[idx, 'custom_judge_score'] = custom_result.get('raw_score', None)
                results_df.at[idx, 'custom_judge_explanation'] = custom_result.get('explanation', None)

        except Exception as e:
            st.error(f"Error processing row {row_num}: {str(e)}")
            # Fill with None on error
            results_df.at[idx, 'geval_faithfulness'] = None
            results_df.at[idx, 'geval_coherence'] = None
            results_df.at[idx, 'geval_relevance'] = None
            results_df.at[idx, 'geval_fluency'] = None
            results_df.at[idx, 'dag_score'] = None
            results_df.at[idx, 'prometheus_score'] = None
            if has_custom:
                results_df.at[idx, 'custom_judge_score'] = None
                results_df.at[idx, 'custom_judge_explanation'] = None

        # Update live preview for first 10 rows
        if preview_placeholder is not None and row_num <= 10:
            # Show preview of first 10 rows with all columns
            preview_df = results_df.head(10).copy()
            preview_placeholder.dataframe(preview_df, use_container_width=True)

    progress_bar.progress(1.0)
    status_text.text(f"✅ Completed! Processed {total_rows}/{total_rows} rows")

    return results_df


def export_results(df: pd.DataFrame, original_format: str, _original_filename: str) -> BytesIO:
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
            st.title("📊 H2O SumBench - Batch Evaluation")
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
            results_df = batch_evaluate_dataset(df, source_col, reference_col, summary_col, model_name, progress_bar, status_text, preview_placeholder, custom_criteria=st.session_state.custom_judge_criteria)

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

    # Inject custom CSS
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    # Check available metrics
    available = check_metric_availability()

    # Branded Header
    st.markdown(f"""
    <div class="branded-header">
    <h1 style="margin-bottom:0;">H2O SumBench</h1>
    <p style="color:#888; margin-top:4px; margin-bottom:0;">
    Summarization Evaluation Framework by <span class="h2o-gold">H2O.ai</span>
    </p>
    </div>
    """, unsafe_allow_html=True)

    # Toast notification for file upload (fades automatically via CSS animation)
    if st.session_state.toast_message:
        st.markdown(f'<div class="toast-notification">{st.session_state.toast_message}</div>', unsafe_allow_html=True)
        st.session_state.toast_message = None

    # --- How It Works section ---
    st.markdown("")
    st.markdown("#### How It Works")
    st.markdown("""
    H2O SumBench evaluates any text summary through a two-stage pipeline.
    You provide a **source document** and a **generated summary**. The framework then runs multiple key metrics to give you a complete picture
    of how well your summary performs.
    """)

    col_s1, col_s2 = st.columns(2)
    with col_s1:
        st.markdown("""
        <div class="edu-callout">
        <strong style="color:#FEC925;">Stage 1: Integrity Check</strong><br>
        <em>Source &rarr; Summary</em><br><br>
        <strong>Faithfulness</strong> &mdash; Does the summary only state facts from the source?
        Detects hallucinated numbers, inverted claims, and fabricated details.<br><br>
        <strong>Completeness</strong> &mdash; Did the summary capture what matters?
        Measures how many key points, entities, and sentences from the source
        are represented.<br><br>
        <span style="color:#888;">Always runs. No reference needed.</span>
        </div>
        """, unsafe_allow_html=True)
    with col_s2:
        st.markdown("""
        <div class="edu-callout">
        <strong style="color:#FEC925;">Stage 2: Conformance Check</strong><br>
        <em>Summary &rarr; Reference</em><br><br>
        <strong>Semantic Match</strong> &mdash; Does the summary convey the same meaning
        as a human-written reference, even with different words?<br><br>
        <strong>Lexical Match</strong> &mdash; Does the summary use the same words and
        structure? 
        Critical when terminology must be preserved
        (e.g., regulatory language, technical terms).<br><br>
        <span style="color:#888;">Only runs if you provide a reference summary.</span>
        </div>
        """, unsafe_allow_html=True)

    # --- Domain context (only show when using built-in samples, not uploaded data) ---
    if st.session_state.uploaded_dataset is None or st.session_state.dataset_cleared:
        st.markdown("")
        st.markdown("#### Built-In Sample Data")
        st.markdown("""
        The framework ships with **10 pre-loaded examples** across two domains, each crafted
        to demonstrate specific evaluation behaviors:
        """)

        col_d1, col_d2 = st.columns(2)
        with col_d1:
            st.markdown("""
            <div class="score-interpretation">
            <strong>Finance (5 examples)</strong><br><br>
            &bull; <strong>Q3 Earnings Report</strong> &mdash; summary hallucinates an operating margin figure<br>
            &bull; <strong>M&A Announcement</strong> &mdash; summary omits regulatory timeline and termination fee<br>
            &bull; <strong>Central Bank Rate Decision</strong> &mdash; good paraphrase but loses precise policy language<br>
            &bull; <strong>IPO Filing</strong> &mdash; captures numbers but misses risk factors<br>
            &bull; <strong>Annual Report</strong> &mdash; accurate but extremely brief 
            </div>
            """, unsafe_allow_html=True)
        with col_d2:
            st.markdown("""
            <div class="score-interpretation">
            <strong>Accidents (5 examples)</strong><br><br>
            &bull; <strong>Industrial Explosion</strong> &mdash; inverts the cause-and-effect of the incident<br>
            &bull; <strong>Highway Collision</strong> &mdash; reports wrong casualty numbers<br>
            &bull; <strong>Aviation Near-Miss</strong> &mdash; omits ATC communications detail<br>
            &bull; <strong>Workplace Safety</strong> &mdash; states wrong penalty amount ($2.4M vs $1.24M)<br>
            &bull; <strong>Disaster Response</strong> &mdash; accurate but non-chronological (coherence test)
            </div>
            """, unsafe_allow_html=True)

        st.markdown("""
        <p style="color:#888; font-size:0.9em;">
        Select a sample from the sidebar to load it, or paste your own text below.
        Each sample is designed so you can see how different metrics respond to
        specific types of errors &mdash; hallucination, omission, paraphrasing, and more.
        </p>
        """, unsafe_allow_html=True)

    display_metric_info()

    # Sidebar - Logo & Branding
    if os.path.exists(LOGO_PATH):
        st.sidebar.image(LOGO_PATH, width=120)
        st.sidebar.caption("Summarization Evaluation Framework")

    # Quick Start Guide
    with st.sidebar.expander("Quick Start Guide"):
        st.markdown("""
        **1.** Select a sample from the dropdown below (finance or accident domain)

        **2.** Review the source, summary, and reference texts loaded into the input areas

        **3.** Click **Evaluate Summary** to run all metrics

        **4.** Explore results: check the dashboard first, then drill into individual metrics
        """)

    st.sidebar.markdown("---")
    st.sidebar.header("Configuration")

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
                # Set toast notification for new file upload (no sidebar duplicate)
                st.session_state.toast_message = f"Loaded: {filename} ({len(df)} rows, {len(df.columns)} columns)"

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

                # Rerun immediately to show toast notification
                st.rerun()
        else:
            pass  # File already processed, no duplicate notification needed

    # Column selection (only show if dataset is uploaded)
    if st.session_state.uploaded_dataset is not None and not st.session_state.dataset_cleared:
        st.sidebar.markdown("---")
        st.sidebar.subheader("🔧 Map Columns")

        def on_column_change():
            """Mark that user has selected columns."""
            if st.session_state.source_column and st.session_state.summary_column:
                st.session_state.columns_selected = True

        # Source and Summary columns (required)
        col1, col2 = st.sidebar.columns(2)

        with col1:
            source_col = st.sidebar.selectbox(
                "Source Column:",
                options=st.session_state.dataset_columns,
                key="source_col_selector",
                help="Select the column containing the source documents",
                on_change=on_column_change
            )
            st.session_state.source_column = source_col

        with col2:
            # Filter out the source column from summary options
            summary_options = [col for col in st.session_state.dataset_columns if col != source_col]
            if summary_options:
                summary_col = st.sidebar.selectbox(
                    "Summary Column:",
                    options=summary_options,
                    key="summary_col_selector",
                    help="Select the column containing the summaries",
                    on_change=on_column_change
                )
                st.session_state.summary_column = summary_col
            else:
                st.sidebar.warning("⚠️ Need at least 2 columns")

        # Reference column (optional) - in sidebar
        reference_options = ["None (Skip Part 2)"] + [col for col in st.session_state.dataset_columns
                            if col != source_col and col != st.session_state.summary_column]
        reference_col = st.sidebar.selectbox(
            "Reference Column (Optional):",
            options=reference_options,
            key="reference_col_selector",
            help="Optional: Select reference summary column for Part 2 metrics",
            on_change=on_column_change
        )
        if reference_col == "None (Skip Part 2)":
            st.session_state.reference_column = None
        else:
            st.session_state.reference_column = reference_col

        # Mark columns as selected when source and summary are set
        if st.session_state.source_column and st.session_state.summary_column:
            st.session_state.columns_selected = True

        # Show preview of column mapping
        if st.session_state.source_column and st.session_state.summary_column:
            st.sidebar.caption(f"✅ Source: `{st.session_state.source_column}`")
            st.sidebar.caption(f"✅ Summary: `{st.session_state.summary_column}`")
            if st.session_state.reference_column:
                st.sidebar.caption(f"✅ Reference: `{st.session_state.reference_column}`")
            else:
                st.sidebar.caption("ℹ️ Reference: None (Part 2 skipped)")

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
            st.session_state.source_column and st.session_state.summary_column and \
            st.session_state.columns_selected:
            # Use uploaded dataset
            df = st.session_state.uploaded_dataset
            source_col = st.session_state.source_column
            reference_col = st.session_state.reference_column  # May be None
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
                    st.session_state.reference_text = str(row[reference_col]) if reference_col else ""
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
            # Build options list with descriptive labels
            sample_labels = get_sample_labels()
            all_options = ["-- Select a sample --"] + sample_labels

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

    # Model selection for LLM-as-a-Judge (if API available)
    if H2OGPTE_AVAILABLE:
        st.sidebar.markdown("---")
        st.sidebar.subheader("LLM Model Selection")

        # Available models list (only the 3 tested working models)
        available_models = [
            'meta-llama/Llama-3.3-70B-Instruct',  # Default
            'meta-llama/Meta-Llama-3.1-70B-Instruct',
            'deepseek-ai/DeepSeek-R1',
        ]

        model_display_names = {
            'meta-llama/Llama-3.3-70B-Instruct': 'Llama 3.3 70B (Recommended)',
            'meta-llama/Meta-Llama-3.1-70B-Instruct': 'Llama 3.1 70B',
            'deepseek-ai/DeepSeek-R1': 'DeepSeek R1',
        }

        selected_model = st.sidebar.selectbox(
            "Select LLM Model:",
            options=available_models,
            index=0,
            format_func=lambda x: model_display_names.get(x, x.split('/')[-1]),
            help="Choose the LLM model for API metrics (G-Eval, DAG, Prometheus)"
        )
        st.session_state.selected_model = selected_model

    # Batch evaluation button (only show if dataset uploaded and API available)
    # Moved AFTER LLM selection so user selects model first
    if st.session_state.uploaded_dataset is not None and \
        st.session_state.source_column and st.session_state.summary_column and \
        st.session_state.columns_selected and H2OGPTE_AVAILABLE:
        st.sidebar.markdown("---")
        st.sidebar.subheader("📊 Batch Evaluation")
        st.sidebar.caption("Evaluate entire dataset with G-Eval, DAG, Prometheus")

        # Use callback to set state immediately when button is clicked
        def start_batch_evaluation():
            st.session_state.start_batch_eval = True

        st.sidebar.button(
            "🚀 Evaluate Entire Dataset",
            type="secondary",
            use_container_width=True,
            on_click=start_batch_evaluation,
            key="batch_eval_sidebar"
        )

    # Automatically set which metrics to run based on availability
    run_era1 = True  # Always available
    run_era2 = available['era2_bertscore']  # Run if BERTScore available
    run_era3 = available['era3']  # Run if Era 3 available
    use_factcc = available['era3']  # Use FactCC if Era 3 available
    use_alignscore = available['era3']  # Use AlignScore if Era 3 available
    use_coverage = True  # Coverage Score always available (uses spaCy)
    run_era3b = H2OGPTE_AVAILABLE  # Run if API configured
    use_dag = H2OGPTE_AVAILABLE  # Use DAG if API configured
    use_prometheus = H2OGPTE_AVAILABLE  # Use Prometheus if API configured

    # Main content
    st.markdown('<div class="gold-divider"></div>', unsafe_allow_html=True)

    # Source Text on top (full width)
    source_text = st.text_area(
        "Source Text",
        value=st.session_state.source_text,
        height=180,
        placeholder="Paste the original source document here...",
        help="The original source text. Stage 1 faithfulness metrics check if the summary is supported by this text."
    )
    source_word_count = len(source_text.split()) if source_text.strip() else 0
    if source_word_count > 400:
        st.caption(f":red[{source_word_count:,} words -- exceeds ~400 word limit for faithfulness metrics]")
    elif source_word_count > 0:
        st.caption(f"{source_word_count:,} words")

    # Generated Summary and Reference Summary side by side
    col_left, col_right = st.columns(2)

    with col_left:
        summary_text = st.text_area(
            "Generated Summary",
            value=st.session_state.summary_text,
            height=180,
            placeholder="Paste the summary to evaluate...",
            help="The summary to evaluate against the source (and reference if provided)."
        )
        summary_word_count = len(summary_text.split()) if summary_text.strip() else 0
        if summary_word_count > 0:
            st.caption(f"{summary_word_count:,} words")

    with col_right:
        reference_text = st.text_area(
            "Reference Summary (optional)",
            value=st.session_state.reference_text,
            height=180,
            placeholder="Paste a reference summary for Stage 2...",
            help="Optional human-written reference. Enables Stage 2 conformance metrics (ROUGE, BLEU, BERTScore)."
        )
        ref_word_count = len(reference_text.split()) if reference_text.strip() else 0
        if ref_word_count > 0:
            st.caption(f"{ref_word_count:,} words")
        else:
            st.caption("Stage 2 skipped without reference")

    # Update session state with current text area values
    st.session_state.source_text = source_text
    st.session_state.reference_text = reference_text
    st.session_state.summary_text = summary_text

    # Evaluation button
    st.markdown('<div class="gold-divider"></div>', unsafe_allow_html=True)
    _col1, col2, _col3 = st.columns([1, 2, 1])

    with col2:
        evaluate_button = st.button(
            "📊 Evaluate Summary",
            type="secondary",
            use_container_width=True
        )

    # Run evaluation
    if evaluate_button:
        if not source_text.strip() or not summary_text.strip():
            st.error("⚠️ Please provide both source text and generated summary.")
        else:
            results = {}
            has_reference = bool(reference_text.strip())

            with st.spinner("Computing evaluation metrics..."):
                # ═══════════════════════════════════════════════════════════════
                # PART 1: SOURCE-BASED EVALUATION (INTEGRITY)
                # Always runs - compares Generated Summary ↔ Source Text
                # ═══════════════════════════════════════════════════════════════

                # Part 1A: Faithfulness (Safety) - Detect hallucinations
                if run_era3:
                    spinner_text = "🛡️ Part 1: Faithfulness Check (NLI"
                    if use_factcc:
                        spinner_text += " + FactCC"
                    if use_alignscore:
                        spinner_text += " + AlignScore"
                    if use_coverage:
                        spinner_text += " + Coverage"
                    spinner_text += ")..."

                    with st.spinner(spinner_text):
                        results["faithfulness"] = compute_all_era3_metrics(
                            summary=summary_text,
                            source=source_text,
                            use_factcc=use_factcc,
                            use_alignscore=use_alignscore,
                            use_coverage=use_coverage,
                            use_factchecker=False,  # Moved to API section
                            factchecker_model=None
                        )

                # Part 1B: Completeness (Local) - Semantic Coverage metrics
                with st.spinner("📦 Part 1: Completeness Check (Semantic Coverage + BERTScore Recall)..."):
                    results["completeness_local"] = compute_all_completeness_metrics(
                        summary=summary_text,
                        source=source_text,
                        use_semantic_coverage=True,
                        use_bertscore_recall=True,
                        use_bartscore=False  # Skip BARTScore for now (large model)
                    )

                # Part 1C: Completeness (LLM) - via LLM Judge
                if run_era3b:
                    spinner_text = f"📦 Part 1: Completeness Check (G-Eval"
                    if use_prometheus:
                        spinner_text += " + Prometheus"
                    spinner_text += f") using {st.session_state.selected_model.split('/')[-1]}..."

                    with st.spinner(spinner_text):
                        try:
                            results["completeness"] = evaluate_all(
                                summary=summary_text,
                                source=source_text,
                                reference_summary=reference_text if has_reference else None,
                                model_name=st.session_state.selected_model,
                                timeout=90,
                                include_dag=use_dag,
                                include_prometheus=use_prometheus
                            )
                        except Exception as e:
                            results["completeness"] = {"error": str(e)}

                # ═══════════════════════════════════════════════════════════════
                # PART 2: REFERENCE-BASED EVALUATION (CONFORMANCE)
                # Only runs if Reference Summary is provided
                # Compares Generated Summary ↔ Reference Summary
                # ═══════════════════════════════════════════════════════════════

                if has_reference:
                    # Part 2A: Semantic Conformance (BERTScore, MoverScore)
                    if run_era2:
                        with st.spinner("🧠 Part 2: Semantic Conformance (BERTScore + MoverScore)..."):
                            results["semantic"] = compute_all_era2_metrics(
                                summary=summary_text,
                                reference_summary=reference_text,  # Compare against reference, not source
                            )

                    # Part 2B: Lexical Conformance (ROUGE, BLEU, METEOR)
                    if run_era1:
                        with st.spinner("📝 Part 2: Lexical Conformance (ROUGE, BLEU, METEOR)..."):
                            results["lexical"] = compute_all_era1_metrics(
                                summary=summary_text,
                                reference_summary=reference_text,  # Compare against reference, not source
                            )

            st.session_state.results = results
            st.session_state.has_reference = has_reference
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
    st.markdown('<div class="gold-divider"></div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div style='text-align: center; color: #888;'>
        <p>H2O SumBench v3.0 | Built with Streamlit |
        <a href='https://h2o.ai' style='color: #FEC925; font-weight: 600;'>H2O.ai</a></p>
        </div>
        """,
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
