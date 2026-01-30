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
from src.evaluators.completeness_metrics import compute_all_completeness_metrics
from src.utils.data_loader import load_sample_data, get_sample_by_index
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
        evaluate_all
    )
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

# Toast notification CSS
TOAST_CSS = """
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
    background: linear-gradient(135deg, #28a745 0%, #20c997 100%);
    color: white;
    border-radius: 8px;
    font-weight: 500;
    box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    z-index: 9999;
    animation: fadeInOut 4s ease-in-out forwards;
    max-width: 350px;
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
    with st.expander("📚 Why Evaluate Summaries? Understanding the Framework"):
        st.markdown("""
        ## The Problem: How Do You Know if a Summary is Good?

        When an AI generates a summary, we need to answer two fundamental questions:

        1. **Is it accurate?** Does the summary faithfully represent the source without making things up?
        2. **Is it complete?** Does it capture the important information?

        And if you have a "gold standard" reference summary:

        3. **Does it match the expected output?** How close is it to what a human expert would write?

        ---

        ## Our Two-Stage Evaluation Approach

        ### 📄 Stage 1: Source vs. Summary (INTEGRITY CHECK)
        *"Can we trust this summary?"*

        We compare the **Generated Summary** directly against the **Source Text** to check:

        **🛡️ Faithfulness** — *Is the summary honest?*
        - Does it only contain information from the source?
        - Does it avoid "hallucinating" facts that don't exist?
        - Example: If the source says "sales grew 10%", the summary shouldn't say "sales doubled"

        **📦 Completeness** — *Did it capture what matters?*
        - Are the main points included?
        - Did important details get lost?
        - Example: A news summary should include who, what, when, where—not just the headline

        ### 📊 Stage 2: Generated vs. Reference Summary (CONFORMANCE CHECK)
        *"How does it compare to a human-written summary?"*

        If you have a reference summary (written by an expert), we measure how closely the generated summary matches it:

        **🧠 Semantic Match** — *Same meaning, different words?*
        - Does it convey the same ideas even with different phrasing?
        - Example: "The CEO resigned" vs "The company's leader stepped down" = semantically similar

        **📝 Lexical Match** — *Same words and structure?*
        - How much word-for-word overlap exists?
        - Useful for checking if specific terminology was preserved

        ---

        ## Why This Matters

        | Use Case | Key Metrics |
        |----------|-------------|
        | **Fact-checking AI outputs** | Faithfulness (NLI, AlignScore, FactCC) |
        | **Ensuring nothing important is missed** | Completeness (Coverage, G-Eval Relevance) |
        | **Matching house style/format** | Conformance (ROUGE, BERTScore) |
        | **Quality assurance for production** | All metrics combined |

        💡 **Tip:** Stage 1 always runs. Stage 2 only runs if you provide a Reference Summary.
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

    st.markdown("""
    <style>
    .dashboard-box {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("### 📋 Summary at a Glance")

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
    with st.expander("📊 Understanding the Difference"):
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

    st.markdown("---")
    st.header("📊 Evaluation Results")

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
    st.subheader("📄 Stage 1: Source Text vs. Generated Summary")
    st.caption("*Checking if the summary is accurate and complete based on the original source*")

    # ─────────────────────────────────────────────────────────────────────────
    # FAITHFULNESS (Safety) - Detect hallucinations
    # ─────────────────────────────────────────────────────────────────────────
    if "faithfulness" in results and results["faithfulness"]:
        st.markdown("### 🛡️ Faithfulness — *Can we trust this summary?*")
        st.markdown("""
        > **Why it matters:** A summary that "hallucinates" facts or contradicts the source is dangerous.
        > These metrics detect if the summary adds false information or misrepresents the source.
        """)

        faith_results = results["faithfulness"]

        # NLI Score
        st.markdown("---")
        col1, col2 = st.columns([1, 2])
        with col1:
            nli_score = faith_results.get("NLI", {})
            if nli_score.get('error') is None:
                score_val = nli_score.get('nli_score', 0)
                st.markdown(f"**NLI Score:** {format_score_display(score_val, 'general', 1.0)}", unsafe_allow_html=True)
            else:
                st.error("NLI Error")
        with col2:
            st.markdown("**Natural Language Inference** — *Does the source logically support the summary?*")
            st.caption("Uses DeBERTa to check if claims in the summary can be inferred from the source. Score > 0.7 means 'entailed' (good), < 0.4 means potential contradiction.")

        # FactCC Score
        if "FactCC" in faith_results:
            col1, col2 = st.columns([1, 2])
            with col1:
                factcc_score = faith_results.get("FactCC", {})
                if factcc_score.get('error') is None and factcc_score.get('score') is not None:
                    st.markdown(f"**FactCC:** {format_score_display(factcc_score['score'], 'general', 1.0)}", unsafe_allow_html=True)
                else:
                    st.warning("FactCC unavailable")
            with col2:
                st.markdown("**Factual Consistency Checker** — *Are there factual errors?*")
                st.caption("A BERT model trained specifically to detect factual inconsistencies in summaries. Low scores flag potential errors.")

        # AlignScore
        if "AlignScore" in faith_results:
            col1, col2 = st.columns([1, 2])
            with col1:
                align_score = faith_results.get("AlignScore", {})
                if align_score.get('error') is None and align_score.get('score') is not None:
                    st.markdown(f"**AlignScore:** {format_score_display(align_score['score'], 'general', 1.0)}", unsafe_allow_html=True)
                else:
                    st.warning("AlignScore unavailable")
            with col2:
                st.markdown("**Unified Alignment Model** ⭐ — *State-of-the-art factual consistency*")
                st.caption("⭐ **Recommended** — Trained on 7 different NLP tasks. Currently the most reliable single metric for factual accuracy.")

        # Coverage Score (NER overlap)
        if "Coverage" in faith_results:
            col1, col2 = st.columns([1, 2])
            coverage_result = faith_results.get("Coverage", {})
            with col1:
                if coverage_result.get('error') is None and coverage_result.get('score') is not None:
                    st.markdown(f"**Entity Coverage:** {format_score_display(coverage_result['score'], 'general', 1.0)}", unsafe_allow_html=True)
                    st.caption(f"{coverage_result.get('covered_entities', 0)}/{coverage_result.get('source_entities', 0)} entities")
                else:
                    st.warning("Coverage unavailable")
            with col2:
                st.markdown("**Named Entity Coverage** — *Are key names, places, dates mentioned?*")
                st.caption("Checks if important entities (people, organizations, locations, dates) from the source appear in the summary.")
                if coverage_result.get('missing_entities'):
                    with st.expander(f"⚠️ Missing: {', '.join(coverage_result['missing_entities'][:3])}..."):
                        st.write(", ".join(coverage_result['missing_entities']))

        # Faithfulness Score Guide
        st.markdown("---")
        nli_val = faith_results.get("NLI", {}).get('nli_score', 0)
        factcc_val = faith_results.get("FactCC", {}).get('score', 0) if faith_results.get("FactCC", {}).get('error') is None else 0
        align_val = faith_results.get("AlignScore", {}).get('score', 0) if faith_results.get("AlignScore", {}).get('error') is None else 0
        avg_faith = (nli_val + factcc_val + align_val) / 3 if (nli_val and factcc_val and align_val) else 0

        if avg_faith >= 0.7:
            st.success(f"✅ **Faithfulness Assessment:** The summary appears well-supported by the source (avg: {avg_faith:.0%})")
        elif avg_faith >= 0.4:
            truncation_note = ""
            if show_token_warning:
                truncation_note = f"\n\n**Note:** Your source text ({word_count:,} words) exceeds the ~400 word limit for these models. The faithfulness metrics only evaluated the first ~400 words of the source, which may explain lower scores if key content appears later in the document."
            st.warning(f"⚠️ **Faithfulness Assessment:** Some claims may need verification (avg: {avg_faith:.0%}){truncation_note}")
        else:
            truncation_note = ""
            if show_token_warning:
                truncation_note = f"\n\n**Note:** Your source text ({word_count:,} words) exceeds the ~400 word limit for these models. The faithfulness metrics only evaluated the first ~400 words of the source, which may explain lower scores if key content appears later in the document."
            st.error(f"❌ **Faithfulness Assessment:** Review carefully for potential errors (avg: {avg_faith:.0%}){truncation_note}")

    # ─────────────────────────────────────────────────────────────────────────
    # COMPLETENESS (Substance) - Key points captured
    # ─────────────────────────────────────────────────────────────────────────

    # Combined Completeness Section (Local + LLM metrics together)
    has_completeness_local = "completeness_local" in results and results["completeness_local"]
    has_completeness_llm = "completeness" in results and results["completeness"]

    if has_completeness_local or has_completeness_llm:
        st.markdown("---")
        st.markdown("### 📦 Completeness — *Did the summary capture what matters?*")
        st.markdown("""
        > **Why it matters:** A summary might be accurate but miss important information.
        > These metrics check if the key points from the source are represented.
        """)

        # Show Local Completeness Metrics first (matching Faithfulness style)
        if has_completeness_local:
            local_comp = results["completeness_local"]

            # Semantic Coverage
            if "SemanticCoverage" in local_comp:
                st.markdown("---")
                col1, col2 = st.columns([1, 2])
                with col1:
                    sc_result = local_comp["SemanticCoverage"]
                    if sc_result.get('error') is None and sc_result.get('score') is not None:
                        st.markdown(f"**Semantic Coverage:** {format_score_display(sc_result['score'], 'general', 1.0)}", unsafe_allow_html=True)
                        st.markdown(f"**Sentences:** {sc_result.get('covered_sentences', 0)}/{sc_result.get('source_sentences', 0)} covered")
                    else:
                        st.warning(f"⚠️ {sc_result.get('error', 'No result')}")
                with col2:
                    st.markdown("**Sentence-Level Coverage** ⭐ — *How many source sentences are represented?*")
                    st.caption("⭐ **Recommended** — Compares each source sentence to the summary using embeddings. Counts how many source sentences have a similar match (>0.7 similarity) in the summary.")

            # BERTScore Recall
            if "BERTScoreRecall" in local_comp:
                col1, col2 = st.columns([1, 2])
                with col1:
                    bs_result = local_comp["BERTScoreRecall"]
                    if bs_result.get('error') is None and bs_result.get('recall') is not None:
                        st.markdown(f"**BERTScore Recall:** {format_score_display(bs_result['recall'], 'bertscore', 1.0)}", unsafe_allow_html=True)
                    else:
                        st.warning(f"⚠️ {bs_result.get('error', 'No result')}")
                with col2:
                    st.markdown("**Meaning Recall** — *What fraction of source meaning is captured?*")
                    st.caption("Measures what percentage of the source's semantic content appears in the summary. Low recall = missing content.")

        # Show LLM Completeness Metrics (G-Eval, DAG, Prometheus)
        if has_completeness_llm:
            st.markdown("---")
            comp_results = results["completeness"]
            if "error" in comp_results:
                st.error(f"Error: {comp_results['error']}")
            else:
                # G-Eval: Relevance
                col1, col2 = st.columns([1, 2])
                with col1:
                    rel_result = comp_results.get("relevance", {})
                    if rel_result.get('error') is None and rel_result.get('score') is not None:
                        raw_score = rel_result.get('raw_score', rel_result['score'] * 10)
                        st.markdown(f"**G-Eval Relevance:** {format_score_display(raw_score, 'geval', 10.0)}", unsafe_allow_html=True)
                    else:
                        st.warning(f"⚠️ {rel_result.get('error', 'No result')}")
                with col2:
                    st.markdown("**Main Points Check** — *Are the important points from the source included?*")
                    if rel_result.get('explanation'):
                        st.caption(f"💬 {rel_result['explanation']}")

                # G-Eval: Coherence
                col1, col2 = st.columns([1, 2])
                with col1:
                    coh_result = comp_results.get("coherence", {})
                    if coh_result.get('error') is None and coh_result.get('score') is not None:
                        raw_score = coh_result.get('raw_score', coh_result['score'] * 10)
                        st.markdown(f"**G-Eval Coherence:** {format_score_display(raw_score, 'geval', 10.0)}", unsafe_allow_html=True)
                    else:
                        st.warning(f"⚠️ {coh_result.get('error', 'No result')}")
                with col2:
                    st.markdown("**Logical Flow** — *Does it flow logically from start to finish?*")
                    st.caption("Checks if ideas connect naturally without abrupt jumps or contradictions.")

                # G-Eval: Faithfulness
                col1, col2 = st.columns([1, 2])
                with col1:
                    faith_result = comp_results.get("faithfulness", {})
                    if faith_result.get('error') is None and faith_result.get('score') is not None:
                        raw_score = faith_result.get('raw_score', faith_result['score'] * 10)
                        st.markdown(f"**G-Eval Faithfulness:** {format_score_display(raw_score, 'geval', 10.0)}", unsafe_allow_html=True)
                    else:
                        st.warning(f"⚠️ {faith_result.get('error', 'No result')}")
                with col2:
                    st.markdown("**Source Alignment** — *Can every claim be traced to the source?*")
                    st.caption("LLM reads both texts and verifies each summary claim against the source.")

                # G-Eval: Fluency
                col1, col2 = st.columns([1, 2])
                with col1:
                    flu_result = comp_results.get("fluency", {})
                    if flu_result.get('error') is None and flu_result.get('score') is not None:
                        raw_score = flu_result.get('raw_score', flu_result['score'] * 10)
                        st.markdown(f"**G-Eval Fluency:** {format_score_display(raw_score, 'geval', 10.0)}", unsafe_allow_html=True)
                    else:
                        st.warning(f"⚠️ {flu_result.get('error', 'No result')}")
                with col2:
                    st.markdown("**Writing Quality** — *Is it grammatically correct and natural?*")
                    st.caption("Evaluates grammar, word choice, and overall readability.")

                with st.expander("💡 What is G-Eval?"):
                    st.markdown("""
                    **G-Eval** uses a large language model (LLM) to evaluate text like a human expert would.

                    Instead of counting word matches or computing embeddings, G-Eval actually *reads* your summary
                    and gives a score based on understanding, just like a teacher grading an essay.

                    **The 4 Dimensions (1-10 scale):**
                    | Dimension | Question Asked |
                    |-----------|----------------|
                    | **Relevance** | "Did it cover the important points?" |
                    | **Coherence** | "Does it flow logically?" |
                    | **Faithfulness** | "Is everything accurate?" |
                    | **Fluency** | "Is it well-written?" |

                    **Interpreting Scores:**
                    - 9-10: Excellent
                    - 7-8: Good
                    - 5-6: Acceptable
                    - Below 5: Needs improvement
                    """)

        # Completeness Assessment Summary
        if has_completeness_llm and "error" not in results.get("completeness", {}):
            comp = results["completeness"]
            qual_scores = []
            for k in ['relevance', 'coherence', 'faithfulness', 'fluency']:
                if k in comp and comp[k].get('raw_score'):
                    qual_scores.append(comp[k]['raw_score'])
            avg_qual = sum(qual_scores) / len(qual_scores) if qual_scores else 0

            st.markdown("---")
            if avg_qual >= 8:
                st.success(f"✅ **Completeness Assessment:** High-quality summary with good coverage of key points (avg G-Eval: {avg_qual:.0f}/10)")
            elif avg_qual >= 6:
                st.warning(f"⚠️ **Completeness Assessment:** Acceptable quality but may miss some points (avg G-Eval: {avg_qual:.0f}/10)")
            else:
                st.error(f"❌ **Completeness Assessment:** Consider revising for better coverage and clarity (avg G-Eval: {avg_qual:.0f}/10)")

        # Completeness Interpretation Guide
        with st.expander("💡 Why Coverage May Differ from Quality"):
            # Get coverage info if available
            cov_info = ""
            if "completeness_local" in results:
                local = results["completeness_local"]
                if "SemanticCoverage" in local and local["SemanticCoverage"].get('score') is not None:
                    cov_score = local["SemanticCoverage"]['score']
                    cov_sent = local["SemanticCoverage"].get('covered_sentences', 0)
                    src_sent = local["SemanticCoverage"].get('source_sentences', 1)
                    cov_info = f"Your summary covers **{cov_sent} of {src_sent}** source sentences ({cov_score:.0%})."

            # Get quality info
            avg_qual = 0
            if has_completeness_llm and "error" not in results["completeness"]:
                qual_scores = []
                comp = results["completeness"]
                for k in ['relevance', 'coherence', 'faithfulness', 'fluency']:
                    if k in comp and comp[k].get('raw_score'):
                        qual_scores.append(comp[k]['raw_score'])
                avg_qual = sum(qual_scores) / len(qual_scores) if qual_scores else 0

            st.markdown(f"""
            {cov_info}

            **Coverage metrics** measure **breadth**:
            - "What percentage of the source is represented in the summary?"
            - A short summary will naturally have low coverage

            **Quality metrics** measure **depth**:
            - "Is what's in the summary accurate, coherent, and well-written?"
            - Average quality score: **{avg_qual:.0f}/10**

            **Common Interpretation:**
            - Low coverage + High quality = **Concise, focused summary** (often acceptable)
            - High coverage + Low quality = **Verbose, but may have issues** (needs review)
            """)

    # ─────────────────────────────────────────────────────────────────────────
    # HOLISTIC ASSESSMENT - Metrics that evaluate both faithfulness AND completeness
    # ─────────────────────────────────────────────────────────────────────────

    has_holistic = "completeness" in results and results["completeness"]
    if has_holistic:
        comp_results = results["completeness"]
        has_dag = "dag" in comp_results and comp_results.get("dag", {}).get('error') is None
        has_prometheus = "prometheus" in comp_results and comp_results.get("prometheus", {}).get('error') is None

        if has_dag or has_prometheus:
            st.markdown("---")
            st.markdown("### 🔄 Holistic Assessment — *End-to-end quality evaluation*")
            st.markdown("""
            > **Why separate?** These metrics evaluate **both** faithfulness and completeness together,
            > giving you a single score that considers accuracy, coverage, and clarity as one unit.
            """)

            # DAG results
            if has_dag:
                dag_result = comp_results.get("dag", {})
                col1, col2 = st.columns([1, 2])
                with col1:
                    if dag_result.get('score') is not None:
                        raw_score = dag_result.get('raw_score', 0)
                        color = "#28a745" if raw_score >= 5 else "#ffc107" if raw_score >= 3 else "#dc3545"
                        st.markdown(f"**DAG Score:** <span style='color: {color}; font-weight: bold;'>{raw_score}/6</span>", unsafe_allow_html=True)
                        step1 = dag_result.get('step1_factual', 'N/A')
                        step2 = dag_result.get('step2_completeness', 'N/A')
                        step3 = dag_result.get('step3_clarity', 'N/A')
                        st.caption(f"Factual: {step1}/2 | Complete: {step2}/2 | Clear: {step3}/2")
                with col2:
                    st.markdown("**Decision Tree** ⭐ — *A 3-step checklist: Is it factual? Complete? Clear?*")
                    st.caption("⭐ **Recommended** — Combines factual accuracy (Step 1), key point coverage (Step 2), and clarity (Step 3) into one structured evaluation.")

                with st.expander("💡 What is DAG?"):
                    st.markdown("""
                    **DAG** evaluates summaries like a decision tree with 3 checkpoints:

                    | Step | Question | Points |
                    |------|----------|--------|
                    | 1. Factual | "Does it only state facts from the source?" | 0-2 |
                    | 2. Complete | "Are the main points included?" | 0-2 |
                    | 3. Clear | "Is it easy to understand?" | 0-2 |

                    **Scoring:** 6/6 = Perfect | 4-5 = Good | 2-3 = Issues | 0-1 = Major problems
                    """)

            # Prometheus results
            if has_prometheus:
                st.markdown("---")
                prom_result = comp_results.get("prometheus", {})
                col1, col2 = st.columns([1, 2])
                with col1:
                    if prom_result.get('score') is not None:
                        raw_score = prom_result.get('raw_score', prom_result['score'])
                        st.markdown(f"**Prometheus:** {format_score_display(raw_score, 'prometheus', 5.0)}", unsafe_allow_html=True)
                with col2:
                    st.markdown("**LLM Judge** ⭐ — *An AI that grades summaries like a teacher (1-5 scale)*")
                    st.caption("⭐ **Recommended** — Holistic quality assessment considering all aspects. 5 = Excellent | 4 = Good | 3 = Acceptable | 2 = Poor | 1 = Very Poor")

            # Holistic Assessment Summary
            st.markdown("---")
            summary_scores = []
            if has_dag and dag_result.get('raw_score') is not None:
                summary_scores.append(('DAG', dag_result['raw_score'], 6))
            if has_prometheus and prom_result.get('raw_score') is not None:
                summary_scores.append(('Prometheus', prom_result['raw_score'], 5))

            if summary_scores:
                # Calculate weighted average
                total_weighted = sum(s[1]/s[2] for s in summary_scores)
                avg_holistic = total_weighted / len(summary_scores)

                if avg_holistic >= 0.75:
                    st.success(f"✅ **Holistic Assessment:** The summary scores well across all dimensions (accuracy, coverage, clarity)")
                elif avg_holistic >= 0.5:
                    st.warning(f"⚠️ **Holistic Assessment:** The summary has room for improvement in some areas")
                else:
                    st.error(f"❌ **Holistic Assessment:** The summary may need significant revision")

    # ═══════════════════════════════════════════════════════════════════════════
    # STAGE 2: GENERATED vs. REFERENCE SUMMARY (CONFORMANCE)
    # Only shown if reference summary was provided
    # ═══════════════════════════════════════════════════════════════════════════
    if has_reference and (("semantic" in results and results["semantic"]) or ("lexical" in results and results["lexical"])):
        st.markdown("---")
        st.subheader("📊 Stage 2: Generated vs. Reference Summary (CONFORMANCE)")
        st.markdown("""
        *How does your summary compare to a human-written "gold standard"?*

        These metrics measure how closely your generated summary matches the reference.
        High scores mean the AI is producing output similar to what a human expert would write.
        """)

        # ─────────────────────────────────────────────────────────────────────
        # SEMANTIC CONFORMANCE (Vibe/Meaning)
        # ─────────────────────────────────────────────────────────────────────
        if "semantic" in results and results["semantic"]:
            st.markdown("### 🧠 Semantic Conformance — *Same meaning, different words?*")
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
                st.markdown("**BERTScore**")
                st.caption("Semantic similarity via embeddings")

                bert_scores = sem_results.get("BERTScore", {})
                if bert_scores.get('error') is None:
                    st.markdown(f"- Precision: {format_score_display(bert_scores.get('precision', 0), 'bertscore')}", unsafe_allow_html=True)
                    st.markdown(f"- Recall: {format_score_display(bert_scores.get('recall', 0), 'bertscore')}", unsafe_allow_html=True)
                    st.markdown(f"- F1: {format_score_display(bert_scores.get('f1', 0), 'bertscore')}", unsafe_allow_html=True)
                else:
                    st.error(f"Error: {bert_scores['error']}")

            with col2:
                st.markdown("**MoverScore**")
                st.caption("Semantic alignment distance")

                mover_score = sem_results.get("MoverScore", {})
                if mover_score.get('error') is None:
                    st.markdown(f"- Score: {format_score_display(mover_score.get('moverscore', 0))}", unsafe_allow_html=True)
                else:
                    st.error(f"Error: {mover_score['error']}")

        # ─────────────────────────────────────────────────────────────────────
        # LEXICAL CONFORMANCE (Format/Structure)
        # ─────────────────────────────────────────────────────────────────────
        if "lexical" in results and results["lexical"]:
            st.markdown("---")
            st.markdown("### 📝 Lexical Conformance — *Same words and structure?*")
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
                st.markdown("**ROUGE Scores**")
                rouge_scores = lex_results.get("ROUGE", {})
                if rouge_scores.get('error') is None:
                    st.markdown(f"- ROUGE-1: {format_score_display(rouge_scores.get('rouge1', 0))}", unsafe_allow_html=True)
                    st.markdown(f"- ROUGE-2: {format_score_display(rouge_scores.get('rouge2', 0))}", unsafe_allow_html=True)
                    st.markdown(f"- ROUGE-L: {format_score_display(rouge_scores.get('rougeL', 0))}", unsafe_allow_html=True)
                else:
                    st.error(f"Error: {rouge_scores['error']}")

            with col2:
                st.markdown("**BLEU Score**")
                bleu_score = lex_results.get("BLEU", {})
                if bleu_score.get('error') is None:
                    st.markdown(f"- BLEU: {format_score_display(bleu_score.get('bleu', 0), 'bleu')}", unsafe_allow_html=True)
                else:
                    st.error(f"Error: {bleu_score['error']}")

                st.markdown("**METEOR Score**")
                meteor_score = lex_results.get("METEOR", {})
                if meteor_score.get('error') is None:
                    st.markdown(f"- METEOR: {format_score_display(meteor_score.get('meteor', 0))}", unsafe_allow_html=True)
                else:
                    st.error(f"Error: {meteor_score['error']}")

                st.markdown("**chrF++ Score**")
                chrf_score = lex_results.get("chrF++", {})
                if chrf_score.get('error') is None:
                    st.markdown(f"- chrF++: {format_score_display(chrf_score.get('chrf', 0))}", unsafe_allow_html=True)
                else:
                    st.error(f"Error: {chrf_score['error']}")

            with col3:
                st.markdown("**Levenshtein Similarity**")
                lev_score = lex_results.get("Levenshtein", {})
                if lev_score.get('error') is None:
                    st.markdown(f"- Similarity: {format_score_display(lev_score.get('levenshtein', 0))}", unsafe_allow_html=True)
                else:
                    st.error(f"Error: {lev_score['error']}")

                st.markdown("**Perplexity (Fluency)**")
                perp_score = lex_results.get("Perplexity", {})
                if perp_score.get('error') is None:
                    st.markdown(f"- Fluency: {format_score_display(perp_score.get('normalized_score', 0))}", unsafe_allow_html=True)
                else:
                    st.warning(f"⚠️ {perp_score.get('error', 'N/A')}")

    elif not has_reference:
        st.markdown("---")
        st.info("ℹ️ **Part 2 (Reference-Based)** skipped - no reference summary provided. Add a reference summary to enable ROUGE, BLEU, BERTScore comparisons.")

    # Batch evaluation button (show at end of results if dataset uploaded)
    if st.session_state.uploaded_dataset is not None and \
       st.session_state.source_column and st.session_state.summary_column and \
       st.session_state.columns_selected and H2OGPTE_AVAILABLE:
        st.markdown("---")
        st.subheader("📊 Batch Evaluation")
        st.caption("Evaluate the entire dataset with API metrics (G-Eval, DAG, Prometheus)")

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
    st.markdown("### Comprehensive Summarization Evaluation Framework")
    if H2OGPTE_AVAILABLE:
        st.markdown("**24 metrics** across 2 evaluation dimensions: **INTEGRITY** (Source-Based) + **CONFORMANCE** (Reference-Based)")
    else:
        st.markdown("**14 local metrics** for faithfulness, completeness, and conformance evaluation")

    # Toast notification for file upload (fades automatically via CSS animation)
    if st.session_state.toast_message:
        st.markdown(TOAST_CSS, unsafe_allow_html=True)
        st.markdown(f'<div class="toast-notification">{st.session_state.toast_message}</div>', unsafe_allow_html=True)
        # Clear toast after showing (it will fade via CSS animation)
        st.session_state.toast_message = None

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
                # Set toast notification for new file upload
                st.session_state.toast_message = f"✅ Loaded: {filename} ({len(df)} rows)"
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

                # Rerun immediately to show toast notification
                st.rerun()
        else:
            # Already processed this file - just show info in sidebar (no toast)
            if st.session_state.uploaded_dataset is not None:
                st.sidebar.info(f"📊 {filename} | {len(st.session_state.uploaded_dataset)} rows × {len(st.session_state.dataset_columns)} columns")

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
            help="Choose the LLM model for API metrics (G-Eval, DAG, Prometheus)"
        )
        st.session_state.selected_model = selected_model
        st.sidebar.caption(f"✅ Using: {selected_model.split('/')[-1]}")

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
            type="primary",
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
    use_unieval = False  # Disabled - UniEval fallback not reliable
    use_factchecker = available['era3'] and H2OGPTE_AVAILABLE  # Use if both available
    run_era3b = H2OGPTE_AVAILABLE  # Run if API configured
    use_dag = H2OGPTE_AVAILABLE  # Use DAG if API configured
    use_prometheus = H2OGPTE_AVAILABLE  # Use Prometheus if API configured

    # Main content
    st.header("📝 Input Texts")

    # Source Text on top (full width)
    st.subheader("Source Text")
    source_text = st.text_area(
        "Enter the original source document:",
        value=st.session_state.source_text,
        height=200,
        help="The source text to summarize"
    )

    # Generated Summary and Reference Summary side by side
    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("Generated Summary")
        summary_text = st.text_area(
            "Enter the summary to evaluate:",
            value=st.session_state.summary_text,
            height=200,
            help="The generated summary to evaluate"
        )

    with col_right:
        st.subheader("Reference Summary (Optional)")
        reference_text = st.text_area(
            "Enter a reference summary for comparison:",
            value=st.session_state.reference_text,
            height=200,
            help="Optional: Add a reference summary to enable Part 2 (conformance) metrics"
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
                    if use_unieval:
                        spinner_text += " + UniEval"
                    spinner_text += ")..."

                    with st.spinner(spinner_text):
                        results["faithfulness"] = compute_all_era3_metrics(
                            summary=summary_text,
                            source=source_text,
                            use_factcc=use_factcc,
                            use_alignscore=use_alignscore,
                            use_coverage=use_coverage,
                            use_unieval=use_unieval,
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
