# H2O SumBench

**Comprehensive Summarization Evaluation Framework**

24 built-in metrics across 5 evaluation dimensions, plus a customizable **LLM-as-a-Judge** for your own criteria.

1. **Faithfulness:** Does the summary stick to the source without hallucinating?
2. **Completeness:** How much of the essential source meaning was captured?
3. **Semantic Alignment:** How well does the summary match the reference summary?
4. **Surface Overlap:** How many specific words/phrases match the reference?
5. **Linguistic Quality:** Is the output readable, logical and well structured?
6. **LLM-as-a-Judge:** Custom prompt template — define your own evaluation criteria and have the LLM score 1-10 with explanation.

---

## Quick Start

```bash
# 1. One-shot install (creates .venv, installs all dependencies)
python setup.py

# 2. Activate the virtual environment
source .venv/bin/activate        # macOS / Linux
.venv\Scripts\activate           # Windows

# 3. Launch
streamlit run ui/app.py
```

---

## Three Ways to Use H2O SumBench

### 1. Standalone Evaluators
Use metrics directly in your code or as an interactive web app.

**Streamlit UI:**
```bash
streamlit run ui/app.py
```

**Python Library:**
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

### 2. Agent with Code Execution Tools
Let an AI agent use the evaluation metrics as callable tools via H2OGPTE.

```bash
# Default: Run agent on CNN/DM dataset
python agents/h2o/orchestrator.py --agent-type agent --sample-idx 0

# Run agent on custom dataset
python agents/h2o/orchestrator.py --agent-type agent --sample-idx 0 --data-file data/processed/YOUR_FILE.json
```

The agent uploads `tool_logic.py` and executes metrics directly through code execution.

### 3. Agent with MCP Server
Use the Model Context Protocol (MCP) for structured tool access.

```bash
# Bundle the MCP server
python mcp_server/bundle.py

# Default: Run agent on CNN/DM dataset
python agents/h2o/orchestrator.py --agent-type agent_with_mcp --sample-idx 0

# Run agent on custom dataset
python agents/h2o/orchestrator.py --agent-type agent_with_mcp --sample-idx 0 --data-file data/processed/YOUR_FILE.json
```

**MCP Server Tools:**

- `evaluate_summary(summary, source, reference)` - Auto-selects and runs all appropriate metrics based on available inputs
- `list_metrics()` - List all 17 available metrics with categories and score ranges
- `get_info(metric_name)` - Get detailed information about a specific metric

**Note:** The orchestrator automatically creates Agent Keys via the H2OGPTE API to inject `H2OGPTE_API_KEY` and `H2OGPTE_ADDRESS` into the MCP server process. Keys are reused across runs.

---

## Metrics Overview

### 1. Faithfulness
Does the summary stick to the source without hallucinating?

| Metric | Type | Description |
|--------|------|-------------|
| NLI | Local | Natural language inference - does source entail summary? |
| FactCC | Local | BERT-based factual consistency classifier |
| AlignScore | Local | Unified alignment score via RoBERTa |
| G-Eval Faithfulness | API | LLM-judged factual accuracy |
| FactChecker | API | LLM-based fact-checking against source |

### 2. Completeness
How much of the essential source meaning was captured?

| Metric | Type | Description |
|--------|------|-------------|
| Entity Coverage | Local | Are named entities preserved? |
| Semantic Coverage | Local | % of source sentences semantically covered |
| BERTScore Recall | Local | What % of source meaning captured? |
| G-Eval Relevance | API | LLM-judged information coverage |

### 3. Semantic Alignment
How well does the summary match the reference summary?

| Metric | Type | Description |
|--------|------|-------------|
| BERTScore | Local | Contextual embedding similarity |
| MoverScore | Local | Earth Mover's Distance on embeddings |
| BARTScore | Local | Generation likelihood score |

### 4. Surface Overlap
How many specific words/phrases match the reference?

| Metric | Type | Description |
|--------|------|-------------|
| ROUGE-1/2/L | Local | Unigram, bigram, and longest common subsequence |
| BLEU | Local | N-gram precision with brevity penalty |
| METEOR | Local | Alignment with synonyms and stemming |
| chrF++ | Local | Character-level F-score |
| Levenshtein | Local | Edit distance ratio |

### 5. Linguistic Quality
Is the output readable, logical and well structured?

| Metric | Type | Description |
|--------|------|-------------|
| Perplexity | Local | GPT-2 language model fluency |
| G-Eval Fluency | API | LLM-judged grammatical quality |
| G-Eval Coherence | API | LLM-judged logical flow |
| DAG | API | Decision tree: accuracy, completeness, clarity |
| Prometheus | API | Open-source LLM judge |

### 6. LLM-as-a-Judge (Custom)
Define your own evaluation criteria using a prompt template.

| Feature | Description |
|---------|-------------|
| Custom Prompt Template | Editable prompt with `{PROMPT}`, `{PREDICTED_TEXT}`, `{TARGET_TEXT}` placeholders |
| Scoring | LLM scores 1-10 with explanation |
| Batch Support | Automatically included in batch evaluation when criteria are set |

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
H2O SumBench/
├── setup.py                        # One-shot install script
├── requirements.txt                # Python dependencies
├── .env.example                    # API credentials (for LLM Judge metrics)
│
├── ui/                             # Streamlit application
│   └── app.py                      # Main entry point (standalone evaluators)
│
├── src/
│   ├── evaluators/
│   │   ├── tool_logic.py           # Unified tool interface (CLI + library)
│   │   ├── h2ogpte_client.py       # Shared H2OGPTe client module
│   │   ├── era1_word_overlap.py    # ROUGE, BLEU, METEOR, etc.
│   │   ├── era2_embeddings.py      # BERTScore, MoverScore
│   │   ├── era3_logic_checkers.py  # NLI, FactCC, AlignScore, Entity Coverage
│   │   ├── era3_llm_judge.py       # G-Eval, DAG, Prometheus, Custom Judge
│   │   └── completeness_metrics.py # Semantic Coverage, BERTScore Recall
│   └── utils/
│       ├── force_cpu.py            # Force CPU-only PyTorch mode
│       └── data_loader.py          # Data loading utilities
│
├── agents/
│   ├── h2o/
│   │   └── orchestrator.py         # H2OGPTE agent orchestrator
│   ├── prompts/
│   │   ├── system_base.md          # Shared foundation prompt
│   │   ├── system_mcp.md           # MCP mode prompt
│   │   ├── system_agent.md         # Agent mode prompt
│   │   └── user.md                 # User prompt template
│   └── shared_utils.py             # Shared utilities for agents
│
├── mcp_server/
│   ├── server.py                   # MCP server implementation
│   ├── envs.json                   # Environment variables JSON
│   └── bundle.py                   # Bundle server for deployment
│
├── data/
│   ├── examples/                   # Template files (CSV, JSON, XLSX)
│   ├── processed/                  # Processed data with AI summaries
│   ├── raw/                        # Raw downloaded data
│   └── scripts/                    # Data processing pipeline
│
├── tests/
│   ├── test_all_metrics.py         # Evaluator metric tests
│   └── test_mcp_server.py          # MCP server layer tests
│
└── docs/
    ├── GETTING_STARTED.md          # First-time user guide
    ├── METRICS.md                  # Complete metrics documentation
    ├── SETUP.md                    # Installation & troubleshooting
    ├── FILE_FORMATS.md             # Dataset upload guide
    └── CHANGELOG.md                # Version history
```

---

## Documentation

- **[docs/GETTING_STARTED.md](docs/GETTING_STARTED.md)** - First-time setup walkthrough
- **[docs/METRICS.md](docs/METRICS.md)** - Complete guide to all 24 metrics + custom judge
- **[docs/SETUP.md](docs/SETUP.md)** - Installation troubleshooting

---

## Version

- **v3.1** - LLM-as-a-Judge: custom prompt template evaluation with batch support
- **v3.0** - GitHub-ready release, setup script, documentation refresh
- **v2.4** - New prompt architecture, enhanced documentation
- **v2.3** - MCP warmup, system installation and dynamic Jinja2 prompt
- **v2.2** - Restructure data folder, pipeline and documentation
- **v2.1** - Added agent integration and MCP server support
- **v2.0** - 24 metrics, educational UI
- **v1.0** - 15 metrics

## License

MIT License - see [LICENSE](LICENSE) for details.
