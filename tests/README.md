# H2O SumBench - Testing Guide

This directory contains all test scripts for verifying metric functionality.

## Quick Test

Run the comprehensive test suite to verify all metrics:

```bash
python3 -m pytest tests/test_all_metrics.py -v
```

**Expected output**: All test cases should pass with PASSED indicators.

---

## Test Files

### Main Test Suite

**`test_all_metrics.py`** - Comprehensive pytest test suite
- Tests Era 1: Lexical metrics (ROUGE, BLEU, METEOR, chrF++, Levenshtein, Perplexity)
- Tests Era 2: Semantic metrics (BERTScore, MoverScore)
- Tests Era 3A: Logic checkers (NLI, FactCC, AlignScore, Coverage Score)
- Tests Era 3B: LLM-as-a-Judge (G-Eval, DAG, Prometheus)
- Tests Completeness metrics (Semantic Coverage, BERTScore Recall)
- Includes edge case testing and full pipeline integration
- **Run this first** to verify everything works

```bash
python3 -m pytest tests/test_all_metrics.py -v
```

### API & Utility Tests

**`test_h2ogpte_api.py`** - API Connectivity Test
- Tests H2OGPTE API connection
- Verifies API key and address configuration
- Simple query to confirm API access works

```bash
python3 tests/test_h2ogpte_api.py
```

### Agent Integration Tests

**`test_h2ogpte_agent.py`** - Agent Tool Integration
- Tests H2OGPTE agent with custom tool ingestion
- Demonstrates agent_only ingest mode
- Tests tool calling capabilities with metrics

```bash
python3 tests/test_h2ogpte_agent.py
```

**`test_simple_agent.py`** - Simple Agent Framework
- Implements a simple agent that calls evaluation metrics as tools
- Demonstrates tool schema definition for LLM function calling
- Multi-step evaluation workflow example

```bash
python3 tests/test_simple_agent.py
```

---

## Expected Test Results

### test_all_metrics.py

When running the comprehensive test suite with pytest:

```bash
python3 -m pytest tests/test_all_metrics.py -v
```

**Expected output**: All tests should pass with ✅ indicators, covering:
- 6 Era 1 metrics (ROUGE, BLEU, METEOR, chrF++, Levenshtein, Perplexity)
- 2 Era 2 metrics (BERTScore, MoverScore)
- 4 Era 3A metrics (NLI, FactCC, AlignScore, Coverage Score)
- 5 Era 3B metrics (G-Eval dimensions + DAG + Prometheus)
- 2 Completeness metrics (Semantic Coverage, BERTScore Recall)
- Edge case handling tests
- Full pipeline integration tests

**Total**: 40+ test cases covering all 15+ metrics

---

## Troubleshooting

### API Tests Failing

**Error**: "H2OGPTE_API_KEY not found"
**Solution**: Create `.env` file in project root:
```bash
H2OGPTE_API_KEY=your_key_here
H2OGPTE_ADDRESS=https://your-instance.h2ogpte.com
```

**Error**: "Invalid model"
**Solution**: Run `test_corrected_models.py` to check available models

### Local Tests Failing

**Error**: "Model not found" or "Download error"
**Solution**: Ensure internet connection for first-time model downloads

**Error**: "Out of memory"
**Solution**: Close other applications, or run fewer metrics at once

### Import Errors

**Error**: "ModuleNotFoundError"
**Solution**: Install requirements:
```bash
pip install -r requirements.txt
```

---

## Test Data

All tests use consistent test data:

**Good Summary**: Accurate facts matching source
- Should score high (>0.7) on all metrics

**Bad Summary**: Multiple factual errors
- Should score low (<0.5) on factual consistency metrics
- May still score okay on fluency/coherence

---

## Adding New Tests

When adding new metrics:

1. Create test in this directory
2. Follow naming convention: `test_<feature>.py`
3. Include both positive and negative test cases
4. Update this README with test description

---

## Performance

Test execution times (approximate):

| Test File | Time | API Calls | Notes |
|-----------|------|-----------|-------|
| test_all_metrics.py | ~5-10min | 6-8 | Comprehensive suite, includes API metrics |
| test_h2ogpte_api.py | ~5s | 1 | Quick connectivity check |
| test_h2ogpte_agent.py | ~30-60s | 2-3 | Agent tool integration |
| test_simple_agent.py | ~30s | 0-1 | Simple agent framework demo |

**Notes**:
- Local-only metrics (Era 1, Era 2) run quickly (<30s total)
- API-dependent metrics (Era 3B) take longer due to API calls
- First run may be slower due to model downloads

---

**Last Updated**: 2026-02-07
**Total Test Files**: 5
**Coverage**: All 23 metrics across 2 stages (Integrity Check, Conformance Check) + LLM-as-a-Judge
