"""
Build an MCP server for H2O.ai SumBench.
"""

import os
import sys
import subprocess

# Install dependencies before importing local modules
def install_dependencies():
    """Install dependencies from requirements.txt if present."""
    server_dir = os.path.dirname(os.path.abspath(__file__))
    requirements_path = os.path.join(server_dir, 'requirements.txt')

    if os.path.exists(requirements_path):
        print(f"[MCP Server] Installing dependencies from {requirements_path}...")
        try:
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", "-q", "-r", requirements_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            print("[MCP Server] Dependencies installed successfully.")
        except subprocess.CalledProcessError as e:
            print(f"[MCP Server] Warning: Failed to install dependencies: {e}")
            print("[MCP Server] Continuing with existing packages...")
    else:
        print(f"[MCP Server] No requirements.txt found at {requirements_path}")

# Install dependencies before any local imports
install_dependencies()

from mcp.server.fastmcp import FastMCP

# Try/except imports for both development and bundled modes
try:
    # Development mode: src/ is in parent directory
    from src.evaluators.tool_logic import (
        list_available_metrics,
        run_multiple_metrics,
        get_metric_info,
    )
except ImportError:
    # Bundled mode: evaluators/ is directly accessible
    from evaluators.tool_logic import (
        list_available_metrics,
        run_multiple_metrics,
        get_metric_info,
    )

mcp = FastMCP("H2O.ai SumBench MCP Server")

# ---------------------------------------------------------------------------
# Metric catalog — single source of truth for every metric exposed via MCP.
# ---------------------------------------------------------------------------
METRIC_CATALOG = {
    # Word Overlap
    'rouge': {
        'category': 'Word Overlap',
        'score_range': '0-1 (F1)',
        'description': 'Word and phrase overlap using ROUGE-1/2/L',
        'recommended_for': ['source+reference', 'reference_only'],
    },
    'bleu': {
        'category': 'Word Overlap',
        'score_range': '0-1',
        'description': 'N-gram precision (BLEU)',
        'recommended_for': ['source+reference', 'reference_only'],
    },
    'meteor': {
        'category': 'Word Overlap',
        'score_range': '0-1',
        'description': 'Alignment-based overlap with synonyms and stemming',
        'recommended_for': ['source+reference', 'reference_only'],
    },
    'levenshtein': {
        'category': 'Word Overlap',
        'score_range': '0-1 (normalized similarity)',
        'description': 'Character-level edit distance similarity',
        'recommended_for': ['source+reference', 'reference_only'],
    },
    'chrf': {
        'category': 'Word Overlap',
        'score_range': '0-100',
        'description': 'Character n-gram F-score (chrF++)',
        'recommended_for': ['source+reference', 'reference_only'],
    },
    # Fluency
    'perplexity': {
        'category': 'Fluency',
        'score_range': '1+ (lower is better)',
        'description': 'Language model perplexity — measures fluency',
        'recommended_for': ['source+reference', 'reference_only', 'neither'],
    },
    # Semantic
    'bertscore': {
        'category': 'Semantic',
        'score_range': '0-1 (F1)',
        'description': 'Contextual embedding similarity (BERTScore F1)',
        'recommended_for': ['source+reference', 'reference_only'],
    },
    # Completeness
    'entity_coverage': {
        'category': 'Completeness',
        'score_range': '0-1',
        'description': 'Fraction of source named entities retained in summary',
        'recommended_for': ['source+reference', 'source_only'],
    },
    'semantic_coverage': {
        'category': 'Completeness',
        'score_range': '0-1',
        'description': 'Sentence-level semantic coverage of source content',
        'recommended_for': ['source+reference', 'source_only'],
    },
    'bertscore_recall': {
        'category': 'Completeness',
        'score_range': '0-1',
        'description': 'BERTScore recall — how much source content is captured',
        'recommended_for': ['source+reference', 'source_only'],
    },
    # LLM Judge
    'llm_faithfulness': {
        'category': 'LLM Judge',
        'score_range': '1-5',
        'description': 'G-Eval faithfulness — factual consistency with source',
        'recommended_for': ['source+reference', 'source_only'],
    },
    'llm_coherence': {
        'category': 'LLM Judge',
        'score_range': '1-5',
        'description': 'G-Eval coherence — logical flow and structure',
        'recommended_for': ['source+reference', 'reference_only'],
    },
    'llm_relevance': {
        'category': 'LLM Judge',
        'score_range': '1-5',
        'description': 'G-Eval relevance — pertinence to the source topic',
        'recommended_for': ['source+reference', 'source_only'],
    },
    'llm_fluency': {
        'category': 'LLM Judge',
        'score_range': '1-5',
        'description': 'G-Eval fluency — grammar and readability',
        'recommended_for': ['source+reference', 'neither'],
    },
    'llm_dag': {
        'category': 'LLM Judge',
        'score_range': '1-5',
        'description': 'DAG — holistic quality assessment via LLM',
        'recommended_for': ['source+reference'],
    },
    'llm_prometheus': {
        'category': 'LLM Judge',
        'score_range': '1-5',
        'description': 'Prometheus — fine-grained LLM evaluation',
        'recommended_for': ['source+reference'],
    },
    'factchecker_api': {
        'category': 'LLM Judge',
        'score_range': '0-1',
        'description': 'LLM-based fact-checking against the source document',
        'recommended_for': ['source+reference', 'source_only'],
    },
}

SUPPORTED_METRICS = set(METRIC_CATALOG.keys())


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _detect_scenario(source, reference):
    """Return the scenario tag based on which inputs are provided."""
    if source and reference:
        return 'source+reference'
    if source:
        return 'source_only'
    if reference:
        return 'reference_only'
    return 'neither'


def _metrics_for_scenario(scenario):
    """Return the list of metric names appropriate for a scenario."""
    return [
        name for name, info in METRIC_CATALOG.items()
        if scenario in info['recommended_for']
    ]


def _extract_primary_score(metric_name: str, result: dict) -> str:
    """Return a single representative score string from a metric result."""
    scores = result.get('scores', {})
    if not scores:
        return 'N/A'

    for key in ('f1', 'score', 'rougeL', 'rouge1', 'bleu', 'meteor',
                'similarity', 'coverage', 'chrf', 'perplexity'):
        if key in scores:
            val = scores[key]
            return f"{val:.4f}" if isinstance(val, float) else str(val)

    for val in scores.values():
        if isinstance(val, (int, float)):
            return f"{val:.4f}" if isinstance(val, float) else str(val)
    return str(next(iter(scores.values())))


def _build_summary(results: dict) -> dict:
    """Build a concise summary dict appended to evaluate_summary output."""
    rows = []
    scored_values = []

    for name, res in results.items():
        if name.startswith('_'):
            continue
        info = METRIC_CATALOG.get(name, {})
        primary = _extract_primary_score(name, res)
        rows.append({
            'metric': name,
            'category': info.get('category', ''),
            'score': primary,
            'range': info.get('score_range', ''),
        })
        try:
            val = float(primary)
            score_range = info.get('score_range', '')
            if score_range.startswith('0-1'):
                scored_values.append(val)
            elif score_range.startswith('1-5'):
                scored_values.append((val - 1) / 4)  # normalize to 0-1
        except (ValueError, TypeError):
            pass

    if scored_values:
        avg = sum(scored_values) / len(scored_values)
        if avg >= 0.85:
            quality = f"Excellent ({avg:.2f} avg normalized)"
        elif avg >= 0.70:
            quality = f"Good ({avg:.2f} avg normalized)"
        elif avg >= 0.55:
            quality = f"Moderate ({avg:.2f} avg normalized)"
        elif avg >= 0.40:
            quality = f"Fair ({avg:.2f} avg normalized)"
        else:
            quality = f"Poor ({avg:.2f} avg normalized)"
    else:
        quality = "Unable to compute (no normalizable scores)"

    return {
        'score_table': rows,
        'overall_quality': quality,
        'guidance': (
            "Present these results as a Markdown table with columns: "
            "Category | Metric | Score | Interpretation. "
            "Add 3-4 bullet-point insights and an overall assessment. "
            "Keep the full response under 700 words."
        ),
    }


# ---------------------------------------------------------------------------
# MCP Tools
# ---------------------------------------------------------------------------

@mcp.tool()
def evaluate_summary(summary: str, source: str = None, reference: str = None):
    """Evaluate an LLM-generated summary using H2O.ai SumBench.

    Call this tool ONCE with the full text of each input. The tool
    automatically selects and runs all appropriate metrics — you do
    NOT need to pick metrics yourself.

    What to pass:
        summary   (REQUIRED) — The generated summary you want to evaluate.
                    Pass the FULL summary text, not a filename or URL.
        source    (REQUIRED) — The original source document that was summarized.
                    Enables faithfulness and completeness checks.
        reference (Optional) — A human-written or gold-standard reference summary.
                    Enables word-overlap and semantic similarity metrics.

    Provide as many inputs as you have. More inputs = more metrics:
        summary + source + reference → 17 metrics (full diagnostic)
        summary + source             →  6 metrics (faithfulness & completeness)
        summary + reference          →  8 metrics (word overlap & semantic)
        summary only                 →  2 metrics (fluency only)

    Returns a dict with:
        - Per-metric results (scores, interpretation, errors)
        - _scenario: which input combination was detected
        - _metrics_used: list of metric names that were run
        - _summary: score_table, overall_quality, and formatting guidance

    Present the results as a Markdown table with columns:
    Category | Metric | Score | Interpretation.
    Add 3-4 bullet-point insights and an overall assessment.
    """
    scenario = _detect_scenario(source, reference)
    metrics = _metrics_for_scenario(scenario)
    results = run_multiple_metrics(metrics, summary, source, reference)
    results['_scenario'] = scenario
    results['_metrics_used'] = metrics
    results['_summary'] = _build_summary(results)
    return results


@mcp.tool()
def list_metrics():
    """List all 17 evaluation metrics available in SumBench.

    Returns a list of metric objects with: name, category, score_range,
    description, and recommended_for (which scenarios use the metric).
    """
    all_metrics = list_available_metrics()
    enriched = []
    for m in all_metrics:
        name = m.get('name', m) if isinstance(m, dict) else m
        if name not in SUPPORTED_METRICS:
            continue
        info = METRIC_CATALOG[name]
        entry = dict(m) if isinstance(m, dict) else {'name': name}
        entry['score_range'] = info['score_range']
        entry['recommended_for'] = info['recommended_for']
        if 'category' not in entry:
            entry['category'] = info['category']
        if 'description' not in entry:
            entry['description'] = info['description']
        enriched.append(entry)
    return enriched


@mcp.tool()
def get_info(metric_name: str):
    """Get detailed information about a specific metric."""
    return get_metric_info(metric_name)


def main():
    mcp.run()


if __name__ == "__main__":
    main()
