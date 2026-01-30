#!/usr/bin/env python3
"""
Test Era 3A Logic Checkers with new API-based FactChecker.
"""

import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from dotenv import load_dotenv
load_dotenv()

from src.evaluators.era3_logic_checkers import (
    compute_nli_score,
    compute_factchecker_score,
    compute_all_era3_metrics
)

print("="*80)
print("Era 3A: Logic Checkers - Comprehensive Test")
print("="*80)

# Test cases
test_cases = [
    {
        "name": "Good Summary (Accurate)",
        "source": """
        The Great Wall of China is over 13,000 miles long and was built over many centuries.
        Construction began in the 7th century BC and continued through the Ming Dynasty.
        It was built to protect Chinese states from invasions and raids.
        """,
        "summary": """
        The Great Wall of China, spanning over 13,000 miles, was constructed across
        multiple centuries starting from the 7th century BC. Its primary purpose was
        to defend against invasions.
        """,
        "expected": "High scores (accurate summary)"
    },
    {
        "name": "Hallucination (False claim)",
        "source": """
        The Eiffel Tower was built for the 1889 World's Fair in Paris. It stands 330 meters tall
        and was designed by Gustave Eiffel. It took 2 years to complete.
        """,
        "summary": """
        The Eiffel Tower was built in 1895 for the Olympics. It stands 400 meters tall and
        was designed by Alexandre Gustave Eiffel. Construction took 5 years.
        """,
        "expected": "Low scores (multiple errors)"
    },
    {
        "name": "Contradiction",
        "source": """
        Climate change is causing global temperatures to rise. Scientists agree that
        human activities are the primary cause.
        """,
        "summary": """
        Climate change is not related to human activities. Global temperatures have been
        stable for decades according to scientific consensus.
        """,
        "expected": "Very low scores (contradicts source)"
    }
]

print("\n🧪 Running Test Cases...")
print("="*80)

for i, test_case in enumerate(test_cases, 1):
    print(f"\n{'='*80}")
    print(f"Test Case {i}: {test_case['name']}")
    print(f"{'='*80}")
    print(f"Expected: {test_case['expected']}")
    print()

    # Test 1: NLI (Local)
    print("1️⃣  NLI (DeBERTa-v3) - Local")
    print("-" * 80)
    nli_result = compute_nli_score(
        summary=test_case['summary'],
        source=test_case['source']
        )
    if 'error' in nli_result:
        print(f"   ❌ Error: {nli_result['error']}")
    else:
        print(f"   Score: {nli_result['nli_score']:.4f}")
        print(f"   Label: {nli_result['label']}")
        print(f"   Interpretation: {nli_result['interpretation']}")

    # Test 2: FactChecker (API)
    print("\n2️⃣  FactChecker (API-based)")
    print("-" * 80)
    fc_result = compute_factchecker_score(
        summary=test_case['summary'],
        source=test_case['source'],
        model_name='meta-llama/Llama-3.3-70B-Instruct',
        use_api=True
    )
    if 'error' in fc_result:
        print(f"   ❌ Error: {fc_result['error']}")
    else:
        if fc_result.get('score') is not None:
            print(f"   Score: {fc_result['score']:.3f} (Raw: {fc_result.get('raw_score', 'N/A')}/10)")
            print(f"   Claims Checked: {fc_result.get('claims_checked', 0)}")
            print(f"   Issues Found: {fc_result.get('issues_found', 0)}")
            print(f"   Interpretation: {fc_result.get('interpretation', 'N/A')}")
            print(f"   Explanation: {fc_result.get('explanation', 'N/A')[:100]}...")
        else:
            print(f"   ⚠️ No score returned")

print("\n" + "="*80)
print("Testing compute_all_era3_metrics Function")
print("="*80)

test_case = test_cases[0]  # Use first test case
print(f"\nUsing: {test_case['name']}")
print()

# Test without FactChecker
print("Without FactChecker (NLI only):")
print("-" * 80)
results = compute_all_era3_metrics(
    summary=test_case['summary'],
    source=test_case['source'],
    use_factchecker=False
)
print(f"Metrics returned: {list(results.keys())}")
for metric, result in results.items():
    if 'score' in result or 'nli_score' in result:
        score = result.get('score') or result.get('nli_score')
        print(f"  {metric}: {score:.4f}")

# Test with FactChecker
print("\nWith FactChecker (NLI + API):")
print("-" * 80)
results = compute_all_era3_metrics(
    summary=test_case['summary'],
    source=test_case['source'],
    use_factchecker=True,
    factchecker_model='meta-llama/Llama-3.3-70B-Instruct'
)
print(f"Metrics returned: {list(results.keys())}")
for metric, result in results.items():
    if 'error' in result:
        print(f"  {metric}: ERROR - {result['error']}")
    else:
        score = result.get('score') or result.get('nli_score')
        if score is not None:
            print(f"  {metric}: {score:.4f}")

print("\n" + "="*80)
print("✅ Era 3A Test Complete")
print("="*80)
print("\nSummary:")
print("  - NLI (Local): Fast, lightweight, local execution")
print("  - FactChecker (API): Thorough, LLM-powered, requires API")
print("  - Both can be used together for comprehensive checking")
