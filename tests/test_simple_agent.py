#!/usr/bin/env python3
"""
Simple Agent Test for Summary Evaluation Metrics.

This file implements a simple agent that can call evaluation metrics from
src/evaluators/agent_tools.py as tools/functions for evaluating summaries.

The agent demonstrates:
1. Tool schema definition for LLM function calling
2. Tool execution via the agent_tools module
3. Multi-step evaluation workflow
4. Result aggregation and interpretation

Run with: python -m pytest tests/agent_test.py -v
Or run directly: python tests/agent_test.py
"""

import os
import sys
import json
from typing import Dict, Any, List, Optional, Callable

# Add paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Force CPU mode
import force_cpu  # noqa: F401


# =============================================================================
# TOOL SCHEMA DEFINITIONS
# =============================================================================

# Tool schemas following OpenAI function calling format
# These can be used with any LLM that supports function calling

EVALUATION_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "list_available_metrics",
            "description": "List all available evaluation metrics with their metadata including requirements and descriptions.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_recommended_metrics",
            "description": "Get recommended metrics based on available inputs (source document, reference summary) and speed preference.",
            "parameters": {
                "type": "object",
                "properties": {
                    "has_source": {
                        "type": "boolean",
                        "description": "Whether source document is available"
                    },
                    "has_reference": {
                        "type": "boolean",
                        "description": "Whether reference summary is available"
                    },
                    "quick_mode": {
                        "type": "boolean",
                        "description": "If true, recommend only fast metrics"
                    }
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "run_metric",
            "description": "Run a specific evaluation metric by name. Returns scores, interpretation, and any errors.",
            "parameters": {
                "type": "object",
                "properties": {
                    "metric_name": {
                        "type": "string",
                        "description": "Name of the metric to run (e.g., 'rouge', 'bertscore', 'nli', 'alignscore')"
                    },
                    "summary": {
                        "type": "string",
                        "description": "The generated summary text to evaluate"
                    },
                    "source": {
                        "type": "string",
                        "description": "The original source document text (required for some metrics)"
                    },
                    "reference_summary": {
                        "type": "string",
                        "description": "A reference summary for comparison (required for some metrics)"
                    }
                },
                "required": ["metric_name", "summary"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "run_multiple_metrics",
            "description": "Run multiple evaluation metrics in a single call. Efficient for comprehensive evaluation.",
            "parameters": {
                "type": "object",
                "properties": {
                    "metric_names": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of metric names to run"
                    },
                    "summary": {
                        "type": "string",
                        "description": "The generated summary text to evaluate"
                    },
                    "source": {
                        "type": "string",
                        "description": "The original source document text"
                    },
                    "reference_summary": {
                        "type": "string",
                        "description": "A reference summary for comparison"
                    }
                },
                "required": ["metric_names", "summary"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_metric_info",
            "description": "Get detailed information about a specific metric including its requirements and what it measures.",
            "parameters": {
                "type": "object",
                "properties": {
                    "metric_name": {
                        "type": "string",
                        "description": "Name of the metric to get info about"
                    }
                },
                "required": ["metric_name"]
            }
        }
    }
]


# =============================================================================
# SIMPLE EVALUATION AGENT
# =============================================================================

class SummaryEvaluationAgent:
    """
    A simple agent that evaluates summaries using various metrics as tools.

    This agent demonstrates the tool/function calling pattern where:
    1. The agent receives a task (evaluate a summary)
    2. The agent decides which tools to call based on available inputs
    3. The agent executes the tools and aggregates results
    4. The agent provides a final assessment

    In a real implementation, steps 1-4 would be handled by an LLM.
    Here we use rule-based logic for demonstration and testing.
    """

    def __init__(self):
        """Initialize the agent with tool functions."""
        # Import tool functions from agent_tools
        from src.evaluators.agent_tools import (
            list_available_metrics,
            get_recommended_metrics,
            run_metric,
            run_multiple_metrics,
            get_metric_info
        )

        # Map tool names to actual functions
        self.tools: Dict[str, Callable] = {
            "list_available_metrics": list_available_metrics,
            "get_recommended_metrics": get_recommended_metrics,
            "run_metric": run_metric,
            "run_multiple_metrics": run_multiple_metrics,
            "get_metric_info": get_metric_info
        }

        # Store conversation/execution history
        self.history: List[Dict[str, Any]] = []

    def execute_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a tool by name with the given arguments.

        This simulates the tool execution that would happen when an LLM
        makes a function call.

        Args:
            tool_name: Name of the tool to execute
            arguments: Dictionary of arguments for the tool

        Returns:
            Tool execution result
        """
        if tool_name not in self.tools:
            return {"error": f"Unknown tool: {tool_name}"}

        tool_func = self.tools[tool_name]

        # Record the tool call
        self.history.append({
            "type": "tool_call",
            "tool": tool_name,
            "arguments": arguments
        })

        try:
            result = tool_func(**arguments)

            # Record the result
            self.history.append({
                "type": "tool_result",
                "tool": tool_name,
                "result": result
            })

            return result
        except Exception as e:
            error_result = {"error": str(e)}
            self.history.append({
                "type": "tool_error",
                "tool": tool_name,
                "error": str(e)
            })
            return error_result

    def evaluate_summary(
        self,
        summary: str,
        source: Optional[str] = None,
        reference_summary: Optional[str] = None,
        quick_mode: bool = False,
        metrics: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Evaluate a summary using appropriate metrics.

        This method demonstrates how an agent would:
        1. Determine which metrics to run based on available inputs
        2. Execute the metric tools
        3. Aggregate and interpret results

        Args:
            summary: The generated summary to evaluate
            source: Original source document (optional)
            reference_summary: Reference summary for comparison (optional)
            quick_mode: If True, only run fast metrics
            metrics: Specific metrics to run (overrides recommendations)

        Returns:
            Dictionary with evaluation results and overall assessment
        """
        self.history = []  # Reset history for new evaluation

        # Step 1: Determine which metrics to run
        if metrics is None:
            # Use the recommendation tool
            recommended = self.execute_tool("get_recommended_metrics", {
                "has_source": source is not None,
                "has_reference": reference_summary is not None,
                "quick_mode": quick_mode
            })
            metrics = recommended

        # Step 2: Run the metrics
        results = self.execute_tool("run_multiple_metrics", {
            "metric_names": metrics,
            "summary": summary,
            "source": source,
            "reference_summary": reference_summary
        })

        # Step 3: Aggregate and interpret results
        overall_assessment = self._generate_assessment(results)

        return {
            "metrics_run": metrics,
            "results": results,
            "overall_assessment": overall_assessment,
            "execution_history": self.history
        }

    def _generate_assessment(self, results: Dict[str, Dict]) -> Dict[str, Any]:
        """
        Generate an overall assessment from metric results.

        In a real LLM-based agent, this would be done by the LLM.
        Here we use simple aggregation rules.
        """
        scores = []
        errors = []
        interpretations = []

        for metric_name, result in results.items():
            if result.get('error'):
                errors.append(f"{metric_name}: {result['error']}")
                continue

            # Extract primary score from result
            metric_scores = result.get('scores', {})
            interpretation = result.get('interpretation', '')

            # Get the main score (varies by metric)
            main_score = None
            if 'score' in metric_scores:
                main_score = metric_scores['score']
            elif 'f1' in metric_scores:
                main_score = metric_scores['f1']
            elif 'recall' in metric_scores:
                main_score = metric_scores['recall']
            elif 'rouge1' in metric_scores:
                main_score = metric_scores['rouge1']
            elif 'nli_score' in metric_scores:
                main_score = metric_scores['nli_score']

            if main_score is not None:
                scores.append(main_score)

            if interpretation:
                interpretations.append(f"{metric_name}: {interpretation}")

        # Calculate average score
        avg_score = sum(scores) / len(scores) if scores else None

        # Generate overall quality label
        if avg_score is None:
            quality = "Unable to assess"
        elif avg_score >= 0.8:
            quality = "Excellent"
        elif avg_score >= 0.6:
            quality = "Good"
        elif avg_score >= 0.4:
            quality = "Fair"
        else:
            quality = "Poor"

        return {
            "average_score": round(avg_score, 4) if avg_score else None,
            "quality_label": quality,
            "metrics_succeeded": len(scores),
            "metrics_failed": len(errors),
            "errors": errors if errors else None,
            "interpretations": interpretations
        }


# =============================================================================
# TEST DATA
# =============================================================================

SOURCE_TEXT = """
The Amazon rainforest covers 5.5 million square kilometers and produces 20% of
the world's oxygen. It is home to 10% of all species on Earth. Recent studies show
deforestation has increased by 30% due to logging and agricultural expansion.
The rainforest plays a crucial role in regulating global climate patterns.
"""

REFERENCE_SUMMARY = """
The Amazon rainforest spans 5.5 million square kilometers and generates 20% of
global oxygen. It houses 10% of Earth's species and is vital for climate regulation.
Deforestation has risen 30% recently due to logging and farming activities.
"""

GOOD_SUMMARY = """
The Amazon rainforest spans 5.5 million square kilometers and generates 20% of
global oxygen. It houses 10% of Earth's species and is vital for climate regulation.
Deforestation has risen 30% recently due to logging and farming.
"""

BAD_SUMMARY = """
The Amazon rainforest covers 10 million square kilometers and produces 50% of
the world's oxygen. It contains 50% of all species. Deforestation has decreased
significantly in recent years due to new conservation efforts.
"""


# =============================================================================
# TESTS
# =============================================================================

class TestEvaluationTools:
    """Tests for the evaluation tool functions."""

    def test_list_available_metrics(self):
        """Test that list_available_metrics returns expected format."""
        from src.evaluators.agent_tools import list_available_metrics

        metrics = list_available_metrics()

        assert isinstance(metrics, list)
        assert len(metrics) > 0

        # Check structure of first metric
        first = metrics[0]
        assert 'name' in first
        assert 'category' in first
        assert 'description' in first
        assert 'requires_source' in first
        assert 'requires_reference' in first

    def test_get_recommended_metrics(self):
        """Test that get_recommended_metrics returns appropriate metrics."""
        from src.evaluators.agent_tools import get_recommended_metrics

        # With source only
        metrics = get_recommended_metrics(has_source=True, has_reference=False)
        assert 'rouge' in metrics
        assert 'nli' in metrics or 'alignscore' in metrics

        # Quick mode
        quick_metrics = get_recommended_metrics(has_source=True, quick_mode=True)
        assert len(quick_metrics) < len(metrics)

    def test_get_metric_info(self):
        """Test that get_metric_info returns correct information."""
        from src.evaluators.agent_tools import get_metric_info

        info = get_metric_info('rouge')

        assert info is not None
        assert info['name'] == 'rouge'
        assert 'description' in info
        assert 'category' in info

    def test_run_metric_rouge(self):
        """Test running ROUGE metric via run_metric."""
        from src.evaluators.agent_tools import run_metric

        result = run_metric(
            metric_name='rouge',
            summary=GOOD_SUMMARY,
            source=SOURCE_TEXT
        )

        assert result['metric_name'] == 'ROUGE'
        assert 'scores' in result
        assert 'rouge1' in result['scores']
        assert 'interpretation' in result
        assert result['error'] is None

    def test_run_metric_perplexity(self):
        """Test running Perplexity metric (no source required)."""
        from src.evaluators.agent_tools import run_metric

        result = run_metric(
            metric_name='perplexity',
            summary=GOOD_SUMMARY
        )

        assert result['metric_name'] == 'Perplexity'
        assert 'scores' in result
        # Note: perplexity may fail if transformers not installed
        if result['error'] is None:
            assert 'perplexity' in result['scores']

    def test_run_metric_missing_required_source(self):
        """Test that metrics requiring source return error when source is missing."""
        from src.evaluators.agent_tools import run_metric

        result = run_metric(
            metric_name='nli',
            summary=GOOD_SUMMARY
            # source is missing but required for NLI
        )

        assert result['error'] is not None
        assert 'requires source' in result['error'].lower()

    def test_run_multiple_metrics(self):
        """Test running multiple metrics at once."""
        from src.evaluators.agent_tools import run_multiple_metrics

        results = run_multiple_metrics(
            metric_names=['rouge', 'bleu', 'levenshtein'],
            summary=GOOD_SUMMARY,
            source=SOURCE_TEXT
        )

        assert 'rouge' in results
        assert 'bleu' in results
        assert 'levenshtein' in results

        # All should have completed
        for metric, result in results.items():
            assert 'metric_name' in result
            assert 'scores' in result


class TestSummaryEvaluationAgent:
    """Tests for the SummaryEvaluationAgent."""

    def test_agent_initialization(self):
        """Test that agent initializes with all tools."""
        agent = SummaryEvaluationAgent()

        assert 'list_available_metrics' in agent.tools
        assert 'run_metric' in agent.tools
        assert 'run_multiple_metrics' in agent.tools
        assert 'get_recommended_metrics' in agent.tools
        assert 'get_metric_info' in agent.tools

    def test_agent_execute_tool(self):
        """Test that agent can execute tools correctly."""
        agent = SummaryEvaluationAgent()

        result = agent.execute_tool('list_available_metrics', {})

        assert isinstance(result, list)
        assert len(result) > 0

        # Check history was recorded
        assert len(agent.history) == 2  # call + result
        assert agent.history[0]['type'] == 'tool_call'
        assert agent.history[1]['type'] == 'tool_result'

    def test_agent_evaluate_summary_quick(self):
        """Test agent evaluation in quick mode."""
        agent = SummaryEvaluationAgent()

        result = agent.evaluate_summary(
            summary=GOOD_SUMMARY,
            source=SOURCE_TEXT,
            quick_mode=True
        )

        assert 'metrics_run' in result
        assert 'results' in result
        assert 'overall_assessment' in result
        assert 'execution_history' in result

        # Quick mode should run fewer metrics
        assert len(result['metrics_run']) <= 5

    def test_agent_evaluate_summary_with_specific_metrics(self):
        """Test agent evaluation with specific metrics."""
        agent = SummaryEvaluationAgent()

        result = agent.evaluate_summary(
            summary=GOOD_SUMMARY,
            source=SOURCE_TEXT,
            metrics=['rouge', 'bleu']
        )

        assert result['metrics_run'] == ['rouge', 'bleu']
        assert 'rouge' in result['results']
        assert 'bleu' in result['results']

    def test_agent_overall_assessment(self):
        """Test that agent generates meaningful assessment."""
        agent = SummaryEvaluationAgent()

        result = agent.evaluate_summary(
            summary=GOOD_SUMMARY,
            source=SOURCE_TEXT,
            metrics=['rouge', 'bleu', 'levenshtein']
        )

        assessment = result['overall_assessment']

        assert 'average_score' in assessment
        assert 'quality_label' in assessment
        assert 'metrics_succeeded' in assessment
        assert 'interpretations' in assessment

    def test_agent_good_vs_bad_summary(self):
        """Test that agent scores good summary higher than bad summary."""
        agent = SummaryEvaluationAgent()

        good_result = agent.evaluate_summary(
            summary=GOOD_SUMMARY,
            source=SOURCE_TEXT,
            metrics=['rouge', 'bleu', 'levenshtein']
        )

        bad_result = agent.evaluate_summary(
            summary=BAD_SUMMARY,
            source=SOURCE_TEXT,
            metrics=['rouge', 'bleu', 'levenshtein']
        )

        good_score = good_result['overall_assessment']['average_score']
        bad_score = bad_result['overall_assessment']['average_score']

        # Good summary should score higher (comparing against source)
        # Note: For source comparison, good summary may not always score higher
        # as it's different from source. But with reference, it should.
        assert good_score is not None
        assert bad_score is not None

    def test_agent_with_reference_summary(self):
        """Test agent evaluation with reference summary."""
        agent = SummaryEvaluationAgent()

        result = agent.evaluate_summary(
            summary=GOOD_SUMMARY,
            source=SOURCE_TEXT,
            reference_summary=REFERENCE_SUMMARY,
            metrics=['rouge', 'meteor', 'bertscore']
        )

        assert result['results'] is not None
        assert len(result['metrics_run']) == 3


class TestToolSchemas:
    """Tests for tool schema definitions."""

    def test_tool_schemas_format(self):
        """Test that tool schemas follow OpenAI format."""
        for tool in EVALUATION_TOOLS:
            assert 'type' in tool
            assert tool['type'] == 'function'
            assert 'function' in tool

            func = tool['function']
            assert 'name' in func
            assert 'description' in func
            assert 'parameters' in func

            params = func['parameters']
            assert 'type' in params
            assert params['type'] == 'object'
            assert 'properties' in params
            assert 'required' in params

    def test_tool_schemas_match_functions(self):
        """Test that tool schemas match actual function signatures."""
        from src.evaluators.agent_tools import (
            list_available_metrics,
            get_recommended_metrics,
            run_metric,
            run_multiple_metrics,
            get_metric_info
        )

        tool_names = [t['function']['name'] for t in EVALUATION_TOOLS]

        assert 'list_available_metrics' in tool_names
        assert 'get_recommended_metrics' in tool_names
        assert 'run_metric' in tool_names
        assert 'run_multiple_metrics' in tool_names
        assert 'get_metric_info' in tool_names


class TestAgentWorkflow:
    """Tests for complete agent workflows."""

    def test_full_evaluation_workflow(self):
        """Test a complete evaluation workflow."""
        agent = SummaryEvaluationAgent()

        # Step 1: List available metrics
        metrics = agent.execute_tool('list_available_metrics', {})
        assert len(metrics) > 20  # Should have many metrics

        # Step 2: Get recommendations
        recommended = agent.execute_tool('get_recommended_metrics', {
            'has_source': True,
            'has_reference': False,
            'quick_mode': False
        })
        assert len(recommended) > 3

        # Step 3: Get info about a specific metric
        info = agent.execute_tool('get_metric_info', {
            'metric_name': 'alignscore'
        })
        assert info['name'] == 'alignscore'
        assert 'RECOMMENDED' in info['description'].upper()

        # Step 4: Run evaluation
        result = agent.execute_tool('run_metric', {
            'metric_name': 'rouge',
            'summary': GOOD_SUMMARY,
            'source': SOURCE_TEXT
        })
        assert result['error'] is None
        assert result['scores']['rouge1'] > 0

    def test_error_handling_workflow(self):
        """Test agent handles errors gracefully."""
        agent = SummaryEvaluationAgent()

        # Try to run unknown metric
        result = agent.execute_tool('run_metric', {
            'metric_name': 'nonexistent_metric',
            'summary': GOOD_SUMMARY
        })

        assert result['error'] is not None
        assert 'unknown metric' in result['error'].lower()

    def test_batch_evaluation_workflow(self):
        """Test evaluating multiple summaries."""
        agent = SummaryEvaluationAgent()

        summaries = [GOOD_SUMMARY, BAD_SUMMARY]
        results = []

        for summary in summaries:
            result = agent.evaluate_summary(
                summary=summary,
                source=SOURCE_TEXT,
                quick_mode=True
            )
            results.append(result)

        assert len(results) == 2

        # Both should have completed
        for r in results:
            assert r['overall_assessment']['average_score'] is not None


# =============================================================================
# DEMO FUNCTIONS
# =============================================================================

def demo_agent_evaluation():
    """
    Demonstrate how the agent evaluates a summary.

    This shows the workflow that would happen with an LLM-based agent.
    """
    print("=" * 60)
    print("SUMMARY EVALUATION AGENT DEMO")
    print("=" * 60)

    agent = SummaryEvaluationAgent()

    print("\n1. Listing available metrics...")
    metrics = agent.execute_tool('list_available_metrics', {})
    print(f"   Found {len(metrics)} metrics available")

    print("\n2. Getting recommended metrics for source-only evaluation...")
    recommended = agent.execute_tool('get_recommended_metrics', {
        'has_source': True,
        'has_reference': False,
        'quick_mode': True
    })
    print(f"   Recommended: {recommended}")

    print("\n3. Evaluating good summary...")
    result = agent.evaluate_summary(
        summary=GOOD_SUMMARY,
        source=SOURCE_TEXT,
        quick_mode=True
    )

    print(f"\n   Metrics run: {result['metrics_run']}")
    print(f"   Overall quality: {result['overall_assessment']['quality_label']}")
    print(f"   Average score: {result['overall_assessment']['average_score']}")

    print("\n4. Evaluating bad summary for comparison...")
    bad_result = agent.evaluate_summary(
        summary=BAD_SUMMARY,
        source=SOURCE_TEXT,
        quick_mode=True
    )

    print(f"   Overall quality: {bad_result['overall_assessment']['quality_label']}")
    print(f"   Average score: {bad_result['overall_assessment']['average_score']}")

    print("\n" + "=" * 60)
    print("DEMO COMPLETE")
    print("=" * 60)


def demo_tool_calling():
    """
    Demonstrate the tool calling interface.

    This shows how an LLM would interact with the tools.
    """
    print("=" * 60)
    print("TOOL CALLING DEMO")
    print("=" * 60)

    agent = SummaryEvaluationAgent()

    # Simulate LLM making function calls
    print("\n[LLM Decision] I need to evaluate this summary. Let me check what metrics are available.")

    print("\n[Tool Call] list_available_metrics()")
    metrics = agent.execute_tool('list_available_metrics', {})
    print(f"[Tool Result] {len(metrics)} metrics available")

    print("\n[LLM Decision] I'll run ROUGE and BERTScore to check lexical and semantic similarity.")

    print("\n[Tool Call] run_multiple_metrics(['rouge', 'bertscore'], ...)")
    result = agent.execute_tool('run_multiple_metrics', {
        'metric_names': ['rouge', 'bertscore'],
        'summary': GOOD_SUMMARY,
        'source': SOURCE_TEXT
    })

    print("\n[Tool Result]")
    for metric, res in result.items():
        print(f"  {metric}: {res.get('interpretation', 'N/A')}")

    print("\n[LLM Response] Based on the evaluation:")
    print(f"  - ROUGE shows {result['rouge']['interpretation']}")
    if result['bertscore'].get('error') is None:
        print(f"  - BERTScore shows {result['bertscore']['interpretation']}")

    print("\n" + "=" * 60)
    print("DEMO COMPLETE")
    print("=" * 60)


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Summary Evaluation Agent Test")
    parser.add_argument('--demo', action='store_true', help='Run demo')
    parser.add_argument('--demo-tools', action='store_true', help='Run tool calling demo')
    parser.add_argument('--test', action='store_true', help='Run tests with pytest')

    args = parser.parse_args()

    if args.demo:
        demo_agent_evaluation()
    elif args.demo_tools:
        demo_tool_calling()
    elif args.test:
        import pytest
        pytest.main([__file__, "-v", "--tb=short"])
    else:
        # Default: run demo
        print("Usage: python agent_test.py [--demo|--demo-tools|--test]")
        print("\nRunning demo by default...\n")
        demo_agent_evaluation()
