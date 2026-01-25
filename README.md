# SumOmniEval - Comprehensive Summary Evaluation Tool

A complete toolkit for evaluating text summarization quality using **15 different metrics** across 3 evaluation eras.

## Quick Start

### 1. Install Dependencies
```bash
pip3 install -r requirements.txt
```

### 2. Configure API (Optional - for Era 3 metrics)
Create a `.env` file or update the `.env.example` file in the project root:
```bash
H2OGPTE_API_KEY=your_api_key_here
```

### 3. Launch Application
```bash
streamlit run app.py
```
or

```
python3 -m streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

---

## What This Tool Does

SumOmniEval evaluates summary quality using **15 metrics** organized into 3 evaluation eras:

| Era | Metrics | Type | Time | Purpose |
|-----|---------|------|------|---------|
| **Era 1: Word Overlap** | 5 | Local | ~2s | Basic n-gram matching |
| **Era 2: Embeddings** | 2 | Local | ~10s | Semantic similarity |
| **Era 3A: Logic Checkers** | 3 | 2 Local + 1 API | ~40s | Factual consistency |
| **Era 3B: AI Simulators** | 5 | API | ~7min | Human-like evaluation |

**Total: 15 metrics (9 local + 6 API)**

---

## Evaluation Workflows

### Fast & Free (Local Only - ~40 seconds)
```
✓ Era 1: ROUGE, BLEU, METEOR, Levenshtein, Perplexity
✓ Era 2: BERTScore, MoverScore
✓ Era 3A: NLI + FactCC (local models)
Result: 9 metrics, no API calls
```

### Balanced (+ API Fact-Check - ~70 seconds)
```
✓ All local metrics
✓ Era 3A: + FactChecker (API)
Result: 10 metrics, 1 API call
```

### Comprehensive (Full Suite - ~8 minutes)
```
✓ All local metrics
✓ Era 3A: All fact-checkers
✓ Era 3B: G-Eval (4 dimensions) + DAG
Result: 15 metrics, 6 API calls
```

---

## Available Metrics

### Era 1: Word Overlap (5 metrics - Local)
- **ROUGE** (1/2/L): N-gram overlap with reference
- **BLEU**: Precision-based machine translation metric
- **METEOR**: Semantic matching with synonyms
- **Levenshtein**: Edit distance similarity
- **Perplexity**: Language model fluency score

### Era 2: Embeddings (2 metrics - Local)
- **BERTScore**: Contextual embedding similarity (Precision/Recall/F1)
- **MoverScore**: Optimal word alignment via Earth Mover's Distance

### Era 3A: Logic Checkers (3 metrics)
- **NLI** (DeBERTa-v3): Natural Language Inference - Local (~400MB)
- **FactCC** (BERT): BERT-based consistency checker - Local (~400MB)
- **FactChecker** (LLM): AI-powered fact-checking - API (0MB)

### Era 3B: AI Simulators (5 metrics - API)
**G-Eval (4 dimensions):**
- **Faithfulness**: Are facts accurate and supported?
- **Coherence**: Does the summary flow logically?
- **Relevance**: Are main points captured?
- **Fluency**: Is the writing clear and grammatical?

**Decision Tree:**
- **DAG** (DeepEval): Step-by-step evaluation (factual → completeness → clarity)

---

## Using the Application

1. **Enter your text**:
   - Source Document: The original text to summarize
   - Summary: The summary to evaluate

2. **Load sample data** (optional):
   - Use the sidebar to load pre-configured examples
   - Helpful for testing and understanding the tool

3. **Configure model** (if using API metrics):
   - Select your preferred LLM from the sidebar
   - Default: `meta-llama/Llama-3.3-70B-Instruct`
   - Other options: Meta-Llama-3.1-70B, DeepSeek-R1

4. **Click "Evaluate Summary"**
   - All available metrics run automatically
   - Local metrics (Era 1, 2, 3A) run first
   - API metrics (Era 3B) run if API key is configured

5. **View results**:
   - Scores range from 0.00-1.00 (Era 1-3A) or 1-10 (Era 3B)
   - Color-coded: 🟢 Green (good) | 🟡 Yellow (fair) | 🔴 Red (poor)
   - Detailed explanations and reasoning for each metric
   - Expand sections to learn more about each metric era

---

## Project Structure

```
SumOmniEval/
├── app.py                          # Main Streamlit application
├── requirements.txt                # Python dependencies
├── .env                            # API configuration (create this)
│
├── src/
│   ├── evaluators/
│   │   ├── era1_basic.py          # ROUGE, BLEU, METEOR, etc.
│   │   ├── era2_embeddings.py     # BERTScore, MoverScore
│   │   ├── era3_logic_checkers.py # NLI, FactCC, FactChecker
│   │   └── era3_llm_judge.py      # G-Eval, DAG
│   └── utils/
│       └── helpers.py             # Shared utilities
│
├── tests/                          # All test scripts
│   ├── README.md                  # Testing guide
│   ├── test_all_new_metrics.py   # Comprehensive test suite
│   ├── test_era3a_factchecker.py # Era 3A tests
│   ├── test_era3b_individual.py  # Era 3B tests
│   └── ...                        # Other test files
│
└── docs/                           # Documentation
    ├── METRICS.md                 # Detailed metric explanations
    ├── SETUP.md                   # Installation & troubleshooting
    └── CHANGELOG.md               # Version history
```

---

## Running Tests

```bash
# Test all metrics (comprehensive)
python3 tests/test_all_new_metrics.py

# Test specific eras
python3 tests/test_era3a_factchecker.py
python3 tests/test_era3b_individual.py

# Test API connectivity
python3 tests/test_h2ogpte_api.py
```

See **[tests/README.md](tests/README.md)** for detailed testing documentation.

---

## Documentation

- **[METRICS.md](docs/METRICS.md)** - Detailed metric explanations and scoring guidelines
- **[SETUP.md](docs/SETUP.md)** - Installation, API configuration, troubleshooting
- **[CHANGELOG.md](docs/CHANGELOG.md)** - Version history and recent updates

---

## Requirements

- **Python**: 3.8 or higher
- **Disk Space**: ~3GB (for local models)
- **RAM**: 8GB+ recommended
- **Internet**: Required for API metrics (Era 3A FactChecker, Era 3B)
- **API Key**: Optional (H2OGPTE for Era 3 API metrics)

---

## Implementation Coverage

| Metric | Status | Implementation |
|--------|--------|----------------|
| Era 1: Word Overlap | ✅ Complete | All 5 metrics (ROUGE, BLEU, METEOR, Levenshtein, Perplexity) |
| Era 2: Embeddings | ✅ Complete | BERTScore + MoverScore |
| Era 3A: NLI | ✅ Complete | DeBERTa-v3 (~400MB) |
| Era 3A: FactCC | ✅ Complete | BERT-based (~400MB) |
| Era 3A: FactChecker | ✅ Complete | LLM-powered (API) |
| Era 3A: AlignScore | ❌ Skipped | Model size exceeds 1GB budget |
| Era 3A: QuestEval | ❌ Skipped | Cython dependency conflicts |
| Era 3B: G-Eval | ✅ Complete | All 4 dimensions (Faithfulness, Coherence, Relevance, Fluency) |
| Era 3B: DAG | ✅ Complete | Decision tree evaluation |
| Era 3B: Prometheus | ❌ Skipped | Complex local model setup |

**Total**: 15 metrics implemented (9 local + 6 API)
**Skipped**: 3 metrics due to technical constraints

---

## Technical Details

### Local Metrics (Era 1, 2, 3A)
- Run on CPU, no internet required
- Models auto-download on first use
- Cached for future runs

### API Metrics (Era 3A FactChecker, Era 3B)
- Require H2OGPTE API key and internet
- Use state-of-the-art LLMs (Llama-3.3-70B by default)
- Configurable model selection

### Performance
- **Local only**: ~40 seconds for 9 metrics
- **+ FactChecker**: ~70 seconds for 10 metrics
- **Full suite**: ~8 minutes for 15 metrics (API latency dependent)

### Model Sizes
- Era 1: ~50MB
- Era 2: ~1.2GB (BERTScore + MoverScore)
- Era 3A NLI: ~400MB
- Era 3A FactCC: ~400MB
- Era 3B: 0MB (API only)

**Total local storage**: ~2.05GB

---

## Quick Reference

### No API Key?
Use **Era 1 + 2 + 3A (NLI + FactCC)** for 9 free local metrics.

### Have API Key?
Enable **Era 3B** for human-like AI evaluation across 4 dimensions.

### Need Fast Results?
Use **Era 1 + 2** for instant evaluation (12 seconds).

### Need Maximum Quality?
Enable **all 15 metrics** for comprehensive multi-perspective evaluation.

---

## License

See LICENSE file for details.

## Contributing

Questions or contributions? Check the documentation in `docs/` or create an issue.

---

**Version**: 1.0
**Last Updated**: 2026-01-25
**Total Metrics**: 15 (9 local + 6 API)
**Ready to use**: `streamlit run app.py`
