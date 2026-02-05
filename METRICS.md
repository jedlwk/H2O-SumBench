# SumOmniEval: Complete Metrics Guide

**24 metrics** across 5 evaluation dimensions.
All local models are under 2GB each. API metrics use H2OGPTE.

---

## Quick Reference Table

| # | Metric | Type | Model Size | Dimension |
|---|--------|------|------------|-----------|
| **1. Faithfulness** |||||
| 1 | NLI (DeBERTa-v3) | Local | ~440MB | Does source logically support summary? |
| 2 | FactCC | Local | ~440MB | Binary: consistent or inconsistent? |
| 3 | AlignScore | Local | ~1.4GB | Unified factual consistency score |
| 4 | G-Eval Faithfulness | API | - | Are claims accurate? (1-10) |
| **2. Completeness** |||||
| 5 | Coverage Score | Local | ~12MB | Are named entities preserved? |
| 6 | Semantic Coverage | Local | ~80MB | How many source sentences covered? |
| 7 | BERTScore Recall | Local | ~1.4GB | What % of source meaning captured? |
| 8 | G-Eval Relevance | API | - | Are main points included? (1-10) |
| **3. Semantic Alignment** |||||
| 9 | BERTScore | Local | ~1.4GB | Semantic similarity (P/R/F1) |
| 10 | MoverScore | Local | ~260MB | Meaning transformation effort |
| 11 | BARTScore | Local | ~1.6GB | Generation likelihood score |
| **4. Surface Overlap** |||||
| 12 | ROUGE-1 | Local | <10MB | Single word overlap |
| 13 | ROUGE-2 | Local | <10MB | Two-word phrase overlap |
| 14 | ROUGE-L | Local | <10MB | Longest common subsequence |
| 15 | BLEU | Local | <1MB | N-gram precision |
| 16 | METEOR | Local | ~100MB | Word match with synonyms/stemming |
| 17 | chrF++ | Local | <1MB | Character-level F-score |
| 18 | Levenshtein | Local | <1MB | Edit distance similarity |
| **5. Linguistic Quality** |||||
| 19 | Perplexity | Local | ~600MB | How natural does it sound? |
| 20 | G-Eval Fluency | API | - | Is it well-written? (1-10) |
| 21 | G-Eval Coherence | API | - | Does it flow logically? (1-10) |
| 22 | DAG | API | - | 3-step decision tree (0-6) |
| 23 | Prometheus | API | - | Open-source LLM judge (1-5) |

**Total Local Storage:** ~6-8GB (models downloaded on first use)

---

## 1. Faithfulness

**Question:** Does the summary stick to the source without hallucinating?

These metrics detect made-up facts, contradictions, and claims that can't be traced back to the source.

---

### NLI - Natural Language Inference
**Model:** `microsoft/deberta-v3-base` (~440MB)

**What it does:** Uses natural language inference to check if the source text *logically supports* the summary.

**How it works:**
```
Source: "Apple released iPhone 15 in September 2023"
Summary: "Apple launched a new iPhone last fall"
→ Result: ENTAILMENT (0.85) - Source supports the claim
```

**Scores:**
- 0.7-1.0: Highly consistent (source supports summary)
- 0.4-0.7: Neutral (neither proven nor disproven)
- 0.0-0.4: Inconsistent (potential contradictions)

**Limitation:** Truncates text to 400 words. Long documents may lose context.

---

### FactCC - Factual Consistency Checker
**Model:** `microsoft/deberta-base-mnli` (~440MB)

**What it does:** Binary classification - is the summary consistent with the source?

**How it works:** Fine-tuned on synthetic errors (swapped dates, names, numbers) to detect factual inconsistencies.

**Scores:**
- 0.6-1.0: Consistent
- 0.0-0.6: Inconsistent (flags potential issues)

**Note:** This implementation uses DeBERTa-MNLI as the original FactCC checkpoint has compatibility issues.

---

### AlignScore - Unified Alignment Metric
**Model:** `liuyanyi/AlignScore-large-hf` (~1.4GB, RoBERTa-large based)

**What it does:** State-of-the-art factual consistency metric trained on 7 diverse alignment tasks.

**Why it's good:** Combines insights from NLI, QA, and paraphrasing tasks into one score.

**Scores:**
- 0.7-1.0: Highly consistent
- 0.5-0.7: Mostly consistent
- 0.3-0.5: Partially consistent
- 0.0-0.3: Inconsistent

**Paper:** "AlignScore: Evaluating Factual Consistency with a Unified Alignment Function" (ACL 2023)

---

### G-Eval Faithfulness
**Model:** API-based (default: `meta-llama/Llama-3.3-70B-Instruct`)

**What it does:** Uses a large language model to evaluate factual accuracy like a human expert.

**Question Asked:** "Can every claim in the summary be traced back to the source?"

**Scores (1-10):**
- 9-10: Excellent - all claims supported
- 7-8: Good - minor unsupported details
- 5-6: Acceptable - some claims unverifiable
- 1-4: Poor - significant hallucinations

---

## 2. Completeness

**Question:** How much of the essential source meaning was captured?

These metrics check if key information, entities, and main points from the source are preserved in the summary.

---

### Coverage Score (Named Entity Overlap)
**Model:** `spaCy/en_core_web_sm` (~12MB)

**What it does:** Checks if named entities (people, places, organizations, dates) from the source appear in the summary.

**Example:**
```
Source entities: ["Apple", "Tim Cook", "September 2023", "iPhone 15"]
Summary entities: ["Apple", "iPhone 15"]
→ Coverage: 50% (2/4 entities)
```

**Why it matters:** If key entities are missing, the summary likely missed important information.

**Scores:**
- 0.7-1.0: Excellent coverage
- 0.5-0.7: Good coverage
- 0.3-0.5: Partial coverage
- 0.0-0.3: Poor coverage

---

### Semantic Coverage
**Model:** `all-MiniLM-L6-v2` (~80MB, sentence-transformers)

**What it does:** Counts how many source sentences are semantically represented in the summary.

**How it works:**
1. Split source into sentences
2. For each source sentence, find most similar summary sentence
3. If similarity > 0.7, mark as "covered"
4. Score = covered sentences / total sentences

**Example:**
```
Source: 10 sentences
Covered: 3 sentences (similarity > 0.7)
→ Score: 0.30 (30% coverage)
```

**Interpretation:** Low coverage doesn't mean bad - a concise summary naturally covers less. Check this alongside quality metrics.

---

### BERTScore Recall (vs Source)
**Model:** `roberta-large` (~1.4GB)

**What it does:** Measures what fraction of the source's *meaning* is captured in the summary.

**Why "Recall"?**
- Precision = "How much of the summary is relevant?"
- **Recall** = "How much of the source did we capture?"

For completeness checking, recall is what matters.

**Scores:** (after baseline rescaling)
- 0.6-1.0: Good recall
- 0.4-0.6: Moderate recall
- 0.0-0.4: Low recall (missing content)

---

### G-Eval Relevance
**Model:** API-based (default: `meta-llama/Llama-3.3-70B-Instruct`)

**What it does:** Uses a large language model to evaluate information coverage.

**Question Asked:** "Are the important points from the source included?"

**Scores (1-10):**
- 9-10: Excellent - all main points included
- 7-8: Good - most important points covered
- 5-6: Acceptable - key points present but gaps
- 1-4: Poor - major information missing

---

## 3. Semantic Alignment

**Question:** How well does the summary match the reference summary?

These metrics measure whether your summary captures the same meaning as a "gold standard" reference, even if using different words.

---

### BERTScore (vs Reference)
**Model:** `roberta-large` (~1.4GB)

**What it does:** Computes semantic similarity using contextual embeddings.

**Three Scores:**
- **Precision:** "How much of my summary is relevant to the reference?"
- **Recall:** "How much of the reference did my summary capture?"
- **F1:** Balanced average (your main number)

**Why it's good:** Understands synonyms and paraphrasing.
```
"The CEO resigned" ≈ "The company's leader stepped down"
→ High BERTScore despite different words
```

---

### MoverScore
**Model:** `distilbert-base-uncased` (~260MB)

**What it does:** Measures the "effort" to transform one meaning into another using Earth Mover's Distance.

**Intuition:** Imagine moving piles of sand - how much work to rearrange the summary's meaning to match the reference?

**Limitation:** 400-word limit per text.

---

### BARTScore
**Model:** `facebook/bart-large-cnn` (~1.6GB)

**What it does:** Uses BART's generation probability to score how likely the summary is given the reference (and vice versa).

**Why it's useful:** Captures fluency and semantic alignment together.

---

## 4. Surface Overlap

**Question:** How many specific words/phrases match the reference?

These metrics count exact lexical matches - same words, same phrases, same structure.

---

### ROUGE-1, ROUGE-2, ROUGE-L
**Model:** None (rule-based)

**What they measure:**

| Metric | What It Counts | Good For |
|--------|----------------|----------|
| **ROUGE-1** | Single word overlap | Basic vocabulary check |
| **ROUGE-2** | Two-word phrase overlap | Phrase-level matching |
| **ROUGE-L** | Longest matching sequence | Structure and word order |

**Example:**
```
Reference: "The quick brown fox jumps"
Summary: "A quick brown fox leaps"

ROUGE-1: 4/5 words match (quick, brown, fox, the→a)
ROUGE-2: 2/4 bigrams match (quick brown, brown fox)
ROUGE-L: "quick brown fox" = 3 words in sequence
```

---

### BLEU
**Model:** None (rule-based)

**What it does:** Precision-focused n-gram matching, originally for machine translation.

**Caution:** BLEU penalizes short outputs heavily. Scores tend to be lower for summaries.
- 0.3+ is good for summarization
- Don't compare BLEU across different tasks

---

### METEOR
**Model:** WordNet (~100MB via NLTK)

**What it does:** Word matching that allows:
- Stemming: "running" = "run"
- Synonyms: "fast" = "quick"

**Why it's useful:** More forgiving than pure word matching.

---

### chrF++
**Model:** None (rule-based)

**What it does:** Character-level F-score. Matches at the character level rather than word level.

**Why it's useful:**
- Handles typos better
- Works well for morphologically rich languages
- Robust to tokenization differences

---

### Levenshtein Similarity
**Model:** None (rule-based)

**What it does:** Edit distance - how many character changes to transform summary into reference?

**Score:** Normalized to 0-1 (1 = identical, 0 = completely different)

---

## 5. Linguistic Quality

**Question:** Is the output readable, logical, and well structured?

These metrics evaluate the summary's writing quality independent of content accuracy.

---

### Perplexity
**Model:** `GPT-2` (~600MB)

**What it does:** Measures how "surprised" a language model is by the text. Lower perplexity = more natural.

**Score:** Normalized using `1 / (1 + log(perplexity))`
- Higher = more fluent
- Lower = awkward phrasing

---

### G-Eval Fluency
**Model:** API-based (default: `meta-llama/Llama-3.3-70B-Instruct`)

**Question Asked:** "Is the summary grammatically correct and natural?"

**Scores (1-10):**
- 9-10: Excellent - publication-ready
- 7-8: Good - minor issues
- 5-6: Acceptable - noticeable errors
- 1-4: Poor - significant grammar/style problems

---

### G-Eval Coherence
**Model:** API-based (default: `meta-llama/Llama-3.3-70B-Instruct`)

**Question Asked:** "Does the summary flow logically from start to finish?"

**Scores (1-10):**
- 9-10: Excellent - clear logical structure
- 7-8: Good - mostly coherent
- 5-6: Acceptable - some disjointed sections
- 1-4: Poor - confusing or illogical

---

### DAG - Decision Tree Evaluation
**Model:** API-based

**What it does:** Evaluates summaries using a 3-step decision tree:

```
Step 1: Is it FACTUAL? (0-2 points)
   └─ "Does the summary only state facts from the source?"

Step 2: Is it COMPLETE? (0-2 points)
   └─ "Are the main points included?"

Step 3: Is it CLEAR? (0-2 points)
   └─ "Is it easy to understand?"

Total: 0-6 points
```

**Example Scoring:**
- 6/6: Perfect - accurate, complete, and clear
- 4/6: Good but missing something
- 2/6: Major issues to address

---

### Prometheus
**Model:** API-based

**What it does:** Open-source LLM judge that grades summaries on a 1-5 scale.

**Scale:**
- 5: Excellent
- 4: Good
- 3: Acceptable
- 2: Poor
- 1: Very Poor

**Paper:** "Prometheus: Inducing Fine-Grained Evaluation Capability in Language Models" (2023)

---

## Metrics NOT Implemented

| Metric | Reason | Alternative |
|--------|--------|-------------|
| **BLEURT** | TensorFlow/PyTorch conflicts | Use BERTScore |
| **QuestEval** | Cython dependency issues | Use Semantic Coverage |
| **UniEval** | Fallback implementation unreliable | Use G-Eval |

---

## Model Storage Summary

| Dimension | Models | Total Size |
|-----------|--------|------------|
| Faithfulness | DeBERTa-v3, DeBERTa-MNLI, AlignScore | ~2.3GB |
| Completeness | spaCy, MiniLM, RoBERTa | ~1.5GB |
| Semantic Alignment | RoBERTa, DistilBERT, BART | ~3.3GB |
| Surface Overlap | Python libraries, WordNet | ~100MB |
| Linguistic Quality | GPT-2 | ~600MB |
| **Total** | | **~6-8GB** |

All models are under 2GB individually and download automatically on first use.

---

## API Configuration

For G-Eval, DAG, and Prometheus, you need H2OGPTE API access:

```bash
# .env file
H2OGPTE_API_KEY=your_key_here
H2OGPTE_ADDRESS=https://your-instance.h2ogpte.com
```

**Available Models:**
- `meta-llama/Llama-3.3-70B-Instruct` (default)
- `meta-llama/Meta-Llama-3.1-70B-Instruct`
- `deepseek-ai/DeepSeek-R1`

---

## Quick Decision Guide

| Scenario | Recommended Metrics |
|----------|---------------------|
| **Quick fact-check** | NLI + AlignScore (Faithfulness) |
| **Completeness check** | Semantic Coverage + G-Eval Relevance |
| **Reference matching** | BERTScore + ROUGE-L |
| **Writing quality** | Perplexity + G-Eval Fluency/Coherence |
| **Full evaluation** | All 24 metrics |
| **No API access** | Local metrics only (14 metrics) |

---

## Version History

- **v2.3** (2026-02-06): Reorganized by 5 evaluation dimensions
- **v2.0** (2026-01-29): 24 metrics, educational UI
- **v1.0** (2026-01-25): Initial release with 15 metrics
