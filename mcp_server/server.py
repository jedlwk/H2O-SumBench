import os
import sys

# Add paths for imports (must be before local imports)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.evaluators.tool_logic import (
    list_available_metrics,
    get_recommended_metrics,
    run_metric,
    run_multiple_metrics,
    get_metric_info,
)

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("SumOmniEval MCP Server")


@mcp.tool()
def list_metrics():
    """List all available evaluation metrics."""
    return list_available_metrics()


@mcp.tool()
def recommend_metrics(has_source: bool = True, has_reference: bool = False, quick: bool = False):
    """Get recommended metrics based on available inputs."""
    return get_recommended_metrics(has_source, has_reference, quick)


@mcp.tool()
def run_single_metric(metric_name: str, summary: str, source: str = None, reference: str = None):
    """Run a single evaluation metric."""
    return run_metric(metric_name, summary, source, reference)


@mcp.tool()
def run_multiple(metrics: list, summary: str, source: str = None, reference: str = None):
    """Run multiple evaluation metrics at once."""
    return run_multiple_metrics(metrics, summary, source, reference)


@mcp.tool()
def get_info(metric_name: str):
    """Get detailed information about a specific metric."""
    return get_metric_info(metric_name)


def main():
    # Initialize and run the server
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
