# EVALUATION TASK
Evaluate the quality of the generated summary provided below.

### DATA INPUTS
**Generated Summary:** {{ generated_summary }}

{% if source %}
**Source Text:** {{ source }}
{% endif %}

{% if reference_summary %}
**Reference Summary:** {{ reference_summary }}
{% endif %}

### INSTRUCTIONS
1. **Scenario Detection:** You have been provided with
   {%- if source and reference_summary %} both the Source and a Reference summary.
   {%- elif source %} only the Source text.
   {%- elif reference_summary %} only a Reference summary.
   {%- else %} neither the Source nor a Reference.
   {%- endif %}

2. **Metric Selection & Execution:** Based on your scenario, run these metrics:

{% if source and reference_summary %}
   {# Full Diagnostic: Word Overlap + Semantic + Factuality + Completeness + G-Eval #}
   ```bash
   python -m src.evaluators.tool_logic run_multiple \
     --metrics "rouge, bleu, meteor, levenshtein, perplexity, chrf,bertscore, moverscore, nli, factcc, alignscore, entity_coverage, factchecker_api, semantic_coverage, bertscore_recall, bartscore,llm_faithfulness, llm_coherence, llm_relevance, llm_fluency, llm_dag, llm_prometheus" \
     --summary "{{ generated_summary }}" \
     --source "{{ source }}" \
     --reference "{{ reference_summary }}"
   ```

{% elif source %}
   {# Truth-First: Factuality + Completeness + G-Eval Faithfulness/Relevance #}
   ```bash
   python -m src.evaluators.tool_logic run_multiple \
     --metrics "nli, factcc, alignscore, entity_coverage, factchecker_api,semantic_coverage, bertscore_recall, bartscore, llm_faithfulness,llm_relevance" \
     --summary "{{ generated_summary }}" \
     --source "{{ source }}"
   ```

{% elif reference_summary %}
   {# Stylistic-Match: Word Overlap + Semantic + G-Eval Coherence #}
   ```bash
   python -m src.evaluators.tool_logic run_multiple \
     --metrics "rouge, bleu, meteor, levenshtein, perplexity, chrf,bertscore, moverscore, llm_coherence" \
     --summary "{{ generated_summary }}" \
     --reference "{{ reference_summary }}"
   ```

{% else %}
   {# Linguistic-Sanity: Perplexity + G-Eval Fluency #}
   ```bash
   python -m src.evaluators.tool_logic run_multiple \
     --metrics "perplexity, llm_fluency" \
     --summary "{{ generated_summary }}"
   ```
{% endif %}

3. **Synthesis:** Provide a final quality assessment based on the results.

Your response should be 500-700 words total with these sections:

#### 1. Scenario & Approach (50 words max)
- State the scenario (Source+Reference / Source Only / Reference Only / None)
- List the evaluation approach

#### 2. Metric Results (Core section - use table format)
| Metric Category | Metric Name | Score | Interpretation |
| :--- | :--- | :--- | :--- |
| [Category] | [Name] | [Value] | [Brief insight] |

#### 3. Key Insights (150 words max)
- 3-4 bullet points with specific, actionable insights
- Focus on what matters: critical strengths or weaknesses
- Avoid generic observations

#### 4. Overall Assessment (100 words max)
- Overall quality score (X/10) with justification
- Main strength (1 sentence)
- Main weakness (1 sentence)
- Recommendation (1 sentence)