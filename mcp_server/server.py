"""
Build an MCP server for H2O SumBench.
"""

import os
import sys
import subprocess

# Install dependencies before importing local modules
def _deps_already_installed():
    """Quick check: can we import key packages needed by the MCP server?"""
    try:
        import mcp          # noqa: F401 – MCP framework
        import rouge_score   # noqa: F401 – ROUGE metric
        return True
    except ImportError:
        return False


def install_dependencies():
    """Install dependencies from vendor/ dir, local wheels, or requirements.txt.

    Skips installation entirely if key packages are already importable
    (e.g. pre-installed by the agent from deps.zip).
    """
    server_dir = os.path.dirname(os.path.abspath(__file__))
    vendor_dir = os.path.join(server_dir, 'vendor')
    wheels_dir = os.path.join(server_dir, 'wheels')
    requirements_path = os.path.join(server_dir, 'requirements.txt')
    nltk_data_dir = os.path.join(server_dir, 'nltk_data')

    if _deps_already_installed():
        print("[MCP Server] Dependencies already installed, skipping.")
    elif os.path.isdir(vendor_dir) and os.listdir(vendor_dir):
        # Vendored mode: add vendor/ to sys.path (no pip needed)
        sys.path.insert(0, vendor_dir)
        print(f"[MCP Server] Using vendored packages from {vendor_dir}")
    elif os.path.isdir(wheels_dir) and os.listdir(wheels_dir):
        # Airgapped mode: install from bundled wheels
        print(f"[MCP Server] Found bundled wheels at {wheels_dir}")
        print(f"[MCP Server] Installing dependencies offline (--no-index)...")
        try:
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", "-q",
                 "--no-index", "--find-links", wheels_dir,
                 "-r", requirements_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            print("[MCP Server] Dependencies installed successfully (offline).")
        except subprocess.CalledProcessError as e:
            print(f"[MCP Server] Warning: Offline install failed: {e}")
            print("[MCP Server] Continuing with existing packages...")
    elif os.path.exists(requirements_path):
        # Online mode: original behavior
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

    # Set NLTK_DATA if bundled nltk_data directory exists
    # Search: server dir first, then CWD (where agent extracts deps.zip)
    nltk_search_paths = [
        nltk_data_dir,
        os.path.join(os.getcwd(), 'nltk_data'),
    ]
    for nltk_path in nltk_search_paths:
        if os.path.isdir(nltk_path):
            os.environ['NLTK_DATA'] = nltk_path
            print(f"[MCP Server] Using NLTK data at {nltk_path}")
            break

# Install dependencies before any local imports
install_dependencies()


def _is_airgapped():
    """Detect whether the server is running in an airgapped environment.

    Detection priority:
      1. SUMBENCH_FORCE_ALL_METRICS=1 → always False (operator override)
      2. SUMBENCH_AIRGAPPED=1         → always True  (explicit flag)
      3. vendor/ directory exists      → True          (auto-detect)
    """
    if os.environ.get('SUMBENCH_FORCE_ALL_METRICS', '').strip() == '1':
        return False
    if os.environ.get('SUMBENCH_AIRGAPPED', '').strip() == '1':
        return True
    server_dir = os.path.dirname(os.path.abspath(__file__))
    vendor_dir = os.path.join(server_dir, 'vendor')
    return os.path.isdir(vendor_dir) and bool(os.listdir(vendor_dir))


_AIRGAPPED_MODE = _is_airgapped()

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

mcp = FastMCP("H2O SumBench MCP Server")

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
        'requires_model': None,
    },
    'bleu': {
        'category': 'Word Overlap',
        'score_range': '0-1',
        'description': 'N-gram precision (BLEU)',
        'recommended_for': ['source+reference', 'reference_only'],
        'requires_model': None,
    },
    'meteor': {
        'category': 'Word Overlap',
        'score_range': '0-1',
        'description': 'Alignment-based overlap with synonyms and stemming',
        'recommended_for': ['source+reference', 'reference_only'],
        'requires_model': None,
    },
    'levenshtein': {
        'category': 'Word Overlap',
        'score_range': '0-1 (normalized similarity)',
        'description': 'Character-level edit distance similarity',
        'recommended_for': ['source+reference', 'reference_only'],
        'requires_model': None,
    },
    'chrf': {
        'category': 'Word Overlap',
        'score_range': '0-100',
        'description': 'Character n-gram F-score (chrF++)',
        'recommended_for': ['source+reference', 'reference_only'],
        'requires_model': None,
    },
    # Fluency
    'perplexity': {
        'category': 'Fluency',
        'score_range': '1+ (lower is better)',
        'description': 'Language model perplexity — measures fluency',
        'recommended_for': ['source+reference', 'reference_only', 'neither'],
        'requires_model': 'hf:gpt2',
    },
    # Semantic
    'bertscore': {
        'category': 'Semantic',
        'score_range': '0-1 (F1)',
        'description': 'Contextual embedding similarity (BERTScore F1)',
        'recommended_for': ['source+reference', 'reference_only'],
        'requires_model': 'hf:distilbert-base-uncased',
    },
    # Completeness
    'entity_coverage': {
        'category': 'Completeness',
        'score_range': '0-1',
        'description': 'Fraction of source named entities retained in summary',
        'recommended_for': ['source+reference', 'source_only'],
        'requires_model': 'spacy:en_core_web_sm',
    },
    'semantic_coverage': {
        'category': 'Completeness',
        'score_range': '0-1',
        'description': 'Sentence-level semantic coverage of source content',
        'recommended_for': ['source+reference', 'source_only'],
        'requires_model': 'hf:all-MiniLM-L6-v2',
    },
    'bertscore_recall': {
        'category': 'Completeness',
        'score_range': '0-1',
        'description': 'BERTScore recall — how much source content is captured',
        'recommended_for': ['source+reference', 'source_only'],
        'requires_model': 'hf:distilbert-base-uncased',
    },
    # LLM Judge
    'llm_faithfulness': {
        'category': 'LLM Judge',
        'score_range': '1-5',
        'description': 'G-Eval faithfulness — factual consistency with source',
        'recommended_for': ['source+reference', 'source_only'],
        'requires_model': None,
    },
    'llm_coherence': {
        'category': 'LLM Judge',
        'score_range': '1-5',
        'description': 'G-Eval coherence — logical flow and structure',
        'recommended_for': ['source+reference', 'reference_only'],
        'requires_model': None,
    },
    'llm_relevance': {
        'category': 'LLM Judge',
        'score_range': '1-5',
        'description': 'G-Eval relevance — pertinence to the source topic',
        'recommended_for': ['source+reference', 'source_only'],
        'requires_model': None,
    },
    'llm_fluency': {
        'category': 'LLM Judge',
        'score_range': '1-5',
        'description': 'G-Eval fluency — grammar and readability',
        'recommended_for': ['source+reference', 'neither'],
        'requires_model': None,
    },
    'llm_dag': {
        'category': 'LLM Judge',
        'score_range': '1-5',
        'description': 'DAG — holistic quality assessment via LLM',
        'recommended_for': ['source+reference'],
        'requires_model': None,
    },
    'llm_prometheus': {
        'category': 'LLM Judge',
        'score_range': '1-5',
        'description': 'Prometheus — fine-grained LLM evaluation',
        'recommended_for': ['source+reference'],
        'requires_model': None,
    },
    'factchecker_api': {
        'category': 'LLM Judge',
        'score_range': '0-1',
        'description': 'LLM-based fact-checking against the source document',
        'recommended_for': ['source+reference', 'source_only'],
        'requires_model': None,
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
    """Return (metrics_to_run, skipped_metrics) for a scenario.

    In airgapped mode, metrics requiring model downloads are skipped.
    """
    all_metrics = [
        name for name, info in METRIC_CATALOG.items()
        if scenario in info['recommended_for']
    ]
    if not _AIRGAPPED_MODE:
        return all_metrics, []
    run = [m for m in all_metrics if METRIC_CATALOG[m].get('requires_model') is None]
    skipped = [m for m in all_metrics if METRIC_CATALOG[m].get('requires_model') is not None]
    return run, skipped


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
    """Evaluate an LLM-generated summary using H2O SumBench.

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

    In airgapped environments (no internet), metrics requiring model downloads
    (perplexity, bertscore, entity_coverage, semantic_coverage, bertscore_recall)
    are automatically skipped. Set SUMBENCH_FORCE_ALL_METRICS=1 to override
    if models are pre-cached.

    Returns a dict with:
        - Per-metric results (scores, interpretation, errors)
        - _scenario: which input combination was detected
        - _metrics_used: list of metric names that were run
        - _skipped_metrics: (airgapped only) metrics skipped and reason
        - _summary: score_table, overall_quality, and formatting guidance

    Present the results as a Markdown table with columns:
    Category | Metric | Score | Interpretation.
    Add 3-4 bullet-point insights and an overall assessment.
    """
    scenario = _detect_scenario(source, reference)
    metrics, skipped = _metrics_for_scenario(scenario)
    results = run_multiple_metrics(metrics, summary, source, reference)
    results['_scenario'] = scenario
    results['_metrics_used'] = metrics
    if skipped:
        results['_skipped_metrics'] = {
            'metrics': skipped,
            'reason': 'Airgapped mode: these metrics require model downloads '
                      'not available offline. Set SUMBENCH_FORCE_ALL_METRICS=1 '
                      'if models are pre-cached.',
        }
    results['_summary'] = _build_summary(results)

    # Surface any metric errors clearly so they're visible in the response
    errors = {}
    for name, res in results.items():
        if name.startswith('_'):
            continue
        err = res.get('error') if isinstance(res, dict) else None
        if err:
            errors[name] = err
    if errors:
        results['_errors'] = errors

    return results


@mcp.tool()
def list_metrics():
    """List all 17 evaluation metrics available in SumBench.

    Returns a list of metric objects with: name, category, score_range,
    description, recommended_for, and available_in_airgap.
    In airgapped mode, model-dependent metrics show status 'disabled (airgapped)'.
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
        needs_model = info.get('requires_model') is not None
        entry['available_in_airgap'] = not needs_model
        if _AIRGAPPED_MODE and needs_model:
            entry['status'] = 'disabled (airgapped)'
        enriched.append(entry)
    return enriched


@mcp.tool()
def get_info(metric_name: str):
    """Get detailed information about a specific metric."""
    return get_metric_info(metric_name)


@mcp.tool()
def check_environment():
    """Check MCP server environment: credentials, NLTK data, airgapped mode.

    Call this FIRST to verify everything is configured before running evaluations.
    Returns a dict with the status of each component.
    """
    status = {
        'airgapped_mode': _AIRGAPPED_MODE,
        'nltk_data_path': os.environ.get('NLTK_DATA', 'NOT SET'),
    }

    # Check NLTK data availability
    try:
        import nltk
        nltk_ok = True
        for path, name in [('corpora/wordnet', 'wordnet'),
                           ('tokenizers/punkt_tab', 'punkt_tab')]:
            try:
                nltk.data.find(path)
            except LookupError:
                nltk_ok = False
                status[f'nltk_{name}'] = 'MISSING'
        status['nltk_data_available'] = nltk_ok
    except ImportError:
        status['nltk_data_available'] = False
        status['nltk_error'] = 'nltk package not installed'

    # Check H2OGPTE client connectivity
    try:
        from evaluators.h2ogpte_client import is_configured, get_credentials, get_h2ogpte_client
    except ImportError:
        try:
            from src.evaluators.h2ogpte_client import is_configured, get_credentials, get_h2ogpte_client
        except ImportError:
            status['h2ogpte_connection'] = 'FAILED: could not import h2ogpte_client'
            return status

    status['h2ogpte_configured'] = is_configured()
    if is_configured():
        try:
            client = get_h2ogpte_client()
            session_id = client.create_chat_session()
            status['h2ogpte_connection'] = 'OK'
            status['test_session_id'] = session_id
        except Exception as e:
            status['h2ogpte_connection'] = f'FAILED: {str(e)}'
    else:
        api_key, address = get_credentials()
        status['h2ogpte_connection'] = 'NOT CONFIGURED'
        status['h2ogpte_api_key_set'] = bool(api_key)
        status['h2ogpte_address_set'] = bool(address)

    return status


def main():
    mcp.run()


if __name__ == "__main__":
    main()
