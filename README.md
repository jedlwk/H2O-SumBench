# SumOmniEval

**Comprehensive Summarization Evaluation Framework**

24 metrics across 2 evaluation stages to answer two questions:
1. **Is this summary trustworthy?** (Stage 1: Source vs Summary)
2. **Does it match expectations?** (Stage 2: Generated vs Reference)

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download spaCy model
python -m spacy download en_core_web_sm

# 3. (Optional) Configure API for LLM metrics
cp .env.example .env
# Edit .env with your H2OGPTE credentials

# 4. Launch
streamlit run ui/app.py
```

The app opens at `http://localhost:8501`. Use the sidebar to navigate between:
- **Standalone Evaluators** - Manual evaluation with all metrics
- **Agent Evaluation** - AI agent-driven evaluation
- **MCP Dashboard** - Build bundles and test tools

---

## Four Ways to Use SumOmniEval

### 1. Standalone Evaluators (Streamlit App)
Use the interactive web interface to evaluate summaries manually.

```bash
streamlit run ui/app.py
```

Supports CSV, JSON, Excel (.xlsx), and TSV file uploads. Map your columns to Source, Summary, and Reference (optional).

### 2. Python Library
Import and use metrics directly in your code.

```python
from src.evaluators.tool_logic import run_metric, run_multiple_metrics, list_available_metrics

# Run a single metric
result = run_metric(
    metric_name="rouge",
    summary="The cat sat on the mat.",
    reference_summary="A cat was sitting on a mat."
)
print(result["scores"])  # {'rouge1': 0.67, 'rouge2': 0.5, 'rougeL': 0.67}

# Run multiple metrics at once
results = run_multiple_metrics(
    metric_names=["rouge", "bertscore", "bleu"],
    summary="Generated summary text",
    source="Original source document",
    reference_summary="Reference summary"
)

# List all available metrics
metrics = list_available_metrics()
```

### 3. Agent with Code Execution Tools
Let an AI agent use the evaluation metrics as callable tools via H2OGPTE.

```bash
python agents/h2o/orchestrator.py --agent-type agent --sample grenfell_tower
```

The agent uploads `tool_logic.py` and executes metrics directly through code execution.

### 4. Agent with MCP Server
Use the Model Context Protocol (MCP) for structured tool access.

```bash
# First, bundle the MCP server
python mcp_server/bundle.py

# Run agent with MCP
python agents/h2o/orchestrator.py --agent-type agent_with_mcp --sample grenfell_tower
```

The MCP server exposes these tools:
- `list_metrics()` - List all available metrics
- `recommend_metrics(has_source, has_reference)` - Get recommended metrics
- `run_single_metric(metric_name, summary, source, reference)` - Run one metric
- `run_multiple(metrics, summary, source, reference)` - Run multiple metrics
- `get_info(metric_name)` - Get metric details

---

## Metrics Overview

### Stage 1: Source vs Summary (12 metrics)
Checks if the summary is **accurate** and **complete** based on the original source.

| Metric | Type | What It Checks |
|--------|------|----------------|
| NLI | Local | Does source logically support summary? |
| FactCC | Local | Is it factually consistent? |
| AlignScore | Local | Unified consistency score |
| Coverage Score | Local | Are named entities preserved? |
| Semantic Coverage | Local | How many source sentences covered? |
| BERTScore Recall | Local | What % of source meaning captured? |
| G-Eval (4 dims) | API | Relevance, Coherence, Faithfulness, Fluency |
| DAG | API | 3-step decision tree evaluation |
| Prometheus | API | Open-source LLM judge |

### Stage 2: Generated vs Reference (12 metrics)
Compares your summary against a "gold standard" reference.

| Metric | Type | What It Checks |
|--------|------|----------------|
| BERTScore | Local | Semantic similarity |
| MoverScore | Local | Meaning transformation distance |
| ROUGE-1/2/L | Local | Word and phrase overlap |
| BLEU | Local | N-gram precision |
| METEOR | Local | Matching with synonyms |
| chrF++ | Local | Character-level F-score |
| Levenshtein | Local | Edit distance |
| Perplexity | Local | Fluency score |

---

## Evaluation Modes

| Mode | Metrics | Time | Requirements |
|------|---------|------|--------------|
| **Local Only** | 12 | ~30s | None |
| **Full Suite** | 24 | ~2min | H2OGPTE API key |

---

## Model Storage

All local models download automatically on first use:

| Model | Size | Used By |
|-------|------|---------|
| roberta-large | ~1.4GB | BERTScore, AlignScore |
| deberta-v3-base | ~440MB | NLI |
| deberta-base-mnli | ~440MB | FactCC |
| distilbert | ~260MB | MoverScore |
| GPT-2 | ~600MB | Perplexity |
| MiniLM-L6 | ~80MB | Semantic Coverage |
| spaCy en_core_web_sm | ~12MB | Coverage Score |

**Total: ~5-6GB** (models cached after first download)

---

## Project Structure

```
SumOmniEval/
├── requirements.txt            # Python dependencies
├── METRICS.md                  # Complete metrics documentation
├── .env.example                # API configuration template
│
├── ui/                         # Streamlit application
│   ├── app.py                  # Main entry point (standalone evaluators)
│   └── pages/
│       ├── 1_Agent_Evaluation.py   # Agent-based evaluation UI
│       └── 2_MCP_Dashboard.py      # MCP server management & testing
│
├── src/evaluators/
│   ├── tool_logic.py           # Unified tool interface (CLI + library)
│   ├── era1_word_overlap.py    # ROUGE, BLEU, METEOR, etc.
│   ├── era2_embeddings.py      # BERTScore, MoverScore
│   ├── era3_logic_checkers.py  # NLI, FactCC, AlignScore
│   ├── era3_llm_judge.py       # G-Eval, DAG, Prometheus
│   └── completeness_metrics.py # Semantic Coverage, BERTScore Recall
│
├── agents/
│   ├── h2o/
│   │   └── orchestrator.py     # H2OGPTE agent orchestrator
│   ├── prompts/
│   │   ├── system.md           # Agent system prompt
│   │   └── user.md             # User prompt template
│   └── shared_utils.py         # Shared utilities for agents
│
├── mcp_server/
│   ├── server.py               # MCP server implementation
│   └── bundle.py               # Bundle server for deployment
│
├── data/
│   └── sample_summaries.json   # Sample evaluation data
│
├── tests/
│   └── test_all_metrics.py     # Comprehensive test suite
│
└── examples/                   # Sample datasets (CSV, JSON, Excel)
```

---

## API Configuration

For G-Eval, DAG, and Prometheus metrics:

```bash
# .env file
H2OGPTE_API_KEY=your_key_here
H2OGPTE_ADDRESS=https://your-instance.h2ogpte.com
```

**Available LLM Models:**
- `meta-llama/Llama-3.3-70B-Instruct` (default)
- `meta-llama/Meta-Llama-3.1-70B-Instruct`
- `deepseek-ai/DeepSeek-R1`

---

## Running Tests

```bash
# Run all metric tests
python -m pytest tests/test_all_metrics.py -v

# Quick syntax check
python -m py_compile ui/app.py
```

---

## Requirements

- Python 3.8+
- 8GB+ RAM recommended
- ~6GB disk space (for models)
- Internet connection (for API metrics)

---

## Documentation

- **[METRICS.md](METRICS.md)** - Complete guide to all 24 metrics
- **[docs/SETUP.md](docs/SETUP.md)** - Installation troubleshooting

---

## Metrics Not Implemented

| Metric | Reason |
|--------|--------|
| BLEURT | TensorFlow/PyTorch conflicts |
| QuestEval | Cython dependency issues |
| UniEval | Fallback implementation unreliable |

See [METRICS.md](METRICS.md) for alternatives.

---

## Version

- **v2.1** - Added agent integration and MCP server support
- **v2.0** - 24 metrics, educational UI

## License

MIT License - see [LICENSE](LICENSE) for details.
