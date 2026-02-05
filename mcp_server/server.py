"""
Build an MCP server for SumOmniEval.
"""

import os
import sys
import subprocess

# Install dependencies before importing local modules
def install_dependencies():
    """Install dependencies from requirements.txt if present."""
    server_dir = os.path.dirname(os.path.abspath(__file__))
    requirements_path = os.path.join(server_dir, 'requirements.txt')

    if os.path.exists(requirements_path):
        print(f"[MCP Server] Installing dependencies from {requirements_path}...")
        try:
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", "-q", "-r", requirements_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            print("[MCP Server] Dependencies installed successfully.")
        except subprocess.CalledProcessError as e:
            print(f"[MCP Server] Warning: Failed to install dependencies: {e}")
            print("[MCP Server] Continuing with existing packages...")
    else:
        print(f"[MCP Server] No requirements.txt found at {requirements_path}")

# Install dependencies before any local imports
install_dependencies()

from mcp.server.fastmcp import FastMCP

# Try/except imports for both development and bundled modes
try:
    # Development mode: src/ is in parent directory
    from src.evaluators.tool_logic import (
        list_available_metrics,
        run_metric,
        run_multiple_metrics,
        get_metric_info,
    )
except ImportError:
    # Bundled mode: evaluators/ is directly accessible
    from evaluators.tool_logic import (
        list_available_metrics,
        run_metric,
        run_multiple_metrics,
        get_metric_info,
    )

mcp = FastMCP("SumOmniEval MCP Server")


@mcp.tool()
def check_env_var() -> str:
    """Test environment variable configuration for API access."""
    import os

    api_key = os.environ.get('H2OGPTE_API_KEY')
    address = os.environ.get('H2OGPTE_ADDRESS')

    if api_key and address:
        with open("success.txt", "w") as f:
            f.write(f"SUCCESS: Environment variable H2OGPTE_API_KEY is set to: {api_key[:10]}...\n")
            f.write(f"SUCCESS: Environment variable H2OGPTE_ADDRESS is set to: {address}\n")
        return f"SUCCESS: Environment variables H2OGPTE_API_KEY and H2OGPTE_ADDRESS are accessible."
    else:
        with open("failure.txt", "w") as f:
            if not api_key:
                f.write("FAILURE: Environment variable H2OGPTE_API_KEY is not set\n")
            if not address:
                f.write("FAILURE: Environment variable H2OGPTE_ADDRESS is not set\n")

        missing = []
        if not api_key:
            missing.append("H2OGPTE_API_KEY")
        if not address:
            missing.append("H2OGPTE_ADDRESS")

        return f"FAILURE: Environment variables not set: {', '.join(missing)}"


@mcp.tool()
def list_metrics():
    """List all available evaluation metrics."""
    return list_available_metrics()


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
    mcp.run()


if __name__ == "__main__":
    main()
