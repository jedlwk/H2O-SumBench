# H2O SumBench

Comprehensive summarization evaluation framework — **23 built-in metrics** across two evaluation stages, plus a customizable **LLM-as-a-Judge** for your own criteria.

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)
![AI](https://img.shields.io/badge/AI-H2OGPTE-FEC925?style=flat-square)
![PyTorch](https://img.shields.io/badge/Framework-PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)
![Platform](https://img.shields.io/badge/Platform-macOS%20%7C%20Windows%20%7C%20Linux-blue?style=flat-square)

---

## Features

| Feature | Description |
|---------|-------------|
| **Stage 1 — Integrity Check** | Source vs Summary (always runs) — faithfulness, completeness, holistic quality |
| **Stage 2 — Conformance Check** | Generated vs Reference (requires reference summary) — semantic and lexical similarity |
| **LLM-as-a-Judge** | Custom prompt template — define your own evaluation criteria, scored 1–10 with explanation |
| **Standalone Library** | Use metrics directly in Python code via `tool_logic.py` |
| **Agent + Code Execution** | Let an H2OGPTE agent call evaluation metrics as tools |
| **Agent + MCP Server** | Structured tool access via Model Context Protocol |

---

## Prerequisites

> **Using Docker?** Skip straight to [Docker Setup](#docker-setup) — no local Python needed.

| Requirement | Version | Notes |
|-------------|---------|-------|
| **Python** | 3.10+ | Tested on 3.10, 3.11, 3.13 |
| **pip** | 20+ | Comes with Python |
| **H2OGPTE** | 1.6.47+	 | Required for LLM-judged metrics (G-Eval, DAG, Prometheus) and Agent Modes |

> **Why Python 3.10 specifically?** Key packages (pandas, numpy, uvicorn) don't ship pre-built Windows wheels for 3.9. On 3.10+, everything installs cleanly on both macOS and Windows — no C compiler needed.
---

## Quick Start (Local)

These steps are the same on **macOS**, **Windows**, and **Linux**. Where a command differs, both versions are shown.

### 1. One-shot install

```bash
python setup.py
```

This creates a `.venv`, installs all dependencies, and downloads NLP models.

### 2. Activate the virtual environment

| macOS / Linux | Windows |
|---------------|---------|
| `source .venv/bin/activate` | `.venv\Scripts\activate` |

### 3. Create your `.env` file (optional — only for LLM-judged metrics)

| macOS / Linux | Windows |
|---------------|---------|
| `cp .env.example .env` | `copy .env.example .env` |

Then open `.env` and add your H2OGPTE credentials:

```
H2OGPTE_API_KEY=your_api_key_here
H2OGPTE_ADDRESS=https://h2ogpte.genai.h2o.ai/
```

> **Where to get these:** Refer to the [Step-by-Step Guide (PDF)](./HOW_TO_GET_H2OGPTE_API.pdf)

### 4. Launch

```bash
streamlit run ui/app.py
```

Go to **http://localhost:8501** in your browser. Done.

---

## Docker Setup

Works on **macOS**, **Windows**, and **Linux** — anywhere Docker runs. No local Python installation required.

### 1. Create your `.env` file

| macOS / Linux | Windows |
|---------------|---------|
| `cp .env.example .env` | `copy .env.example .env` |

Edit `.env` with your H2OGPTE credentials (same as step 3 above).

### 2. Build and run

```bash
docker build -t sumbench .
docker run -p 8501:8501 --env-file .env sumbench
```

### 3. Open the app

Go to **http://localhost:8501**.

### Stop

```bash
docker stop $(docker ps -q --filter ancestor=sumbench)
```

---

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `H2OGPTE_API_KEY` | For LLM metrics | Your H2OGPTE API key |
| `H2OGPTE_ADDRESS` | For LLM metrics | Your H2OGPTE instance URL |
| `OPENAI_API_KEY` | No | Optional — if using OpenAI models |
| `ANTHROPIC_API_KEY` | No | Optional — if using Claude models |

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

### Stage 1: Integrity Check (Source vs Summary)

These metrics always run — they only need the source document and the generated summary.

#### 1.1 Faithfulness
Does the summary stick to the source without hallucinating?

| Metric | Type | Description |
|--------|------|-------------|
| NLI | Local | Natural language inference - does source entail summary? |
| FactCC | Local | BERT-based factual consistency classifier |
| AlignScore | Local | Unified alignment score via RoBERTa |
| Entity Coverage | Local | Are named entities preserved? |

#### 1.2 Completeness
How much of the essential source meaning was captured?

| Metric | Type | Description |
|--------|------|-------------|
| Semantic Coverage | Local | % of source sentences semantically covered |
| BERTScore Recall | Local | What % of source meaning captured? |

#### 1.3 Holistic Assessment
LLM-judged quality across multiple dimensions.

| Metric | Type | Description |
|--------|------|-------------|
| DAG | API | Decision tree: accuracy, completeness, clarity |
| Prometheus | API | Open-source LLM judge |
| G-Eval Faithfulness | API | LLM-judged factual accuracy |
| G-Eval Relevance | API | LLM-judged information coverage |
| G-Eval Coherence | API | LLM-judged logical flow |
| G-Eval Fluency | API | LLM-judged grammatical quality |

### Stage 2: Conformance Check (Generated vs Reference)

These metrics require a reference summary for comparison.

#### 2.1 Semantic Similarity
How well does the summary match the reference in meaning?

| Metric | Type | Description |
|--------|------|-------------|
| BERTScore | Local | Contextual embedding similarity |
| MoverScore | Local | Earth Mover's Distance on embeddings |

#### 2.2 Lexical Similarity
How many specific words/phrases match the reference?

| Metric | Type | Description |
|--------|------|-------------|
| ROUGE-1/2/L | Local | Unigram, bigram, and longest common subsequence |
| BLEU | Local | N-gram precision with brevity penalty |
| METEOR | Local | Alignment with synonyms and stemming |
| chrF++ | Local | Character-level F-score |
| Levenshtein | Local | Edit distance ratio |
| Perplexity | Local | GPT-2 language model fluency |

### LLM-as-a-Judge (Custom)
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

**Total: ~5–6 GB** (models cached after first download)

---

## Tech Stack

| Layer | Technology |
|-------|------------|
| UI | Streamlit |
| Deep Learning | PyTorch |
| Embeddings | HuggingFace Transformers |
| NLP | spaCy, NLTK |
| AI / LLM | H2OGPTE (LLM-judged metrics + agent orchestration) |

---

## Project Structure

```
H2O SumBench/
├── setup.py                        # One-shot install script
├── requirements.txt                # Python dependencies
├── Dockerfile                      # Docker container build
├── .env.example                    # API credentials (for LLM Judge metrics)
├── HOW_TO_GET_H2OGPTE_API.pdf      # Step-by-step guide to get your H2OGPTe API key
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

- **[docs/GETTING_STARTED.md](docs/GETTING_STARTED.md)** — First-time setup walkthrough
- **[docs/METRICS.md](docs/METRICS.md)** — Complete guide to all 23 metrics + custom judge
- **[docs/SETUP.md](docs/SETUP.md)** — Installation troubleshooting

---

## Version

| Version | Highlights |
|---------|------------|
| **v3.0** (final) | LLM-as-a-Judge, MCP server, agent integration, one-shot setup |
| **v2.2** | Data pipeline restructure, documentation overhaul |
| **v2.1** | Agent integration, MCP server support |
| **v2.0** | 24 metrics, educational Streamlit UI |
| **v1.1** | Semantic and faithfulness metrics |
| **v1.0** | Initial release — 15 lexical metrics |

See [CHANGELOG.md](docs/CHANGELOG.md) for full details.

---

## Author

**Jed Lee** — [jed.lee@h2o.ai](mailto:jed.lee@h2o.ai)

---

## License

MIT License — see [LICENSE](LICENSE) for details.
