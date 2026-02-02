"""
Agent Evaluation Page - Run H2OGPTE agent-based summary evaluation.
"""

import os
import sys

# Add paths for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'agents'))

import streamlit as st
from dotenv import load_dotenv

st.set_page_config(
    page_title="Agent Evaluation",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 Agent Evaluation")
st.markdown("Let an AI agent evaluate summaries using the SumOmniEval metrics.")

# Check for H2OGPTE credentials
load_dotenv()
H2OGPTE_API_KEY = os.getenv('H2OGPTE_API_KEY')
H2OGPTE_ADDRESS = os.getenv('H2OGPTE_ADDRESS')

if not H2OGPTE_API_KEY or not H2OGPTE_ADDRESS:
    st.error("⚠️ H2OGPTE credentials not found. Please configure `.env` file with:")
    st.code("H2OGPTE_API_KEY=your_key_here\nH2OGPTE_ADDRESS=https://your-instance.h2ogpte.com")
    st.stop()

st.success("✅ H2OGPTE credentials found")

# Import agent modules (after path setup)
try:
    from agents.shared_utils import load_summaries
    from agents.h2o.orchestrator import create_client, setup_collection, run_evaluation
except ImportError as e:
    st.error(f"Failed to import agent modules: {e}")
    st.stop()

# Sidebar configuration
st.sidebar.header("Configuration")

# Agent type selection
agent_type = st.sidebar.radio(
    "Agent Type",
    options=["agent", "agent_with_mcp"],
    format_func=lambda x: "Code Execution" if x == "agent" else "MCP Server",
    help="Code Execution: Agent runs tool_logic.py directly\nMCP Server: Agent uses MCP protocol"
)

if agent_type == "agent_with_mcp":
    st.sidebar.warning("⚠️ MCP mode requires the bundle to be created first. Use the MCP Dashboard page.")

# Load samples
try:
    samples = load_summaries()
    sample_keys = list(samples.keys())
except Exception as e:
    st.error(f"Failed to load samples: {e}")
    st.stop()

# Sample selection
sample_key = st.sidebar.selectbox(
    "Select Sample",
    options=sample_keys,
    format_func=lambda x: samples[x].get('name', x)
)

# Main content
col1, col2 = st.columns(2)

with col1:
    st.subheader("Generated Summary")
    generated_summary = st.text_area(
        "Generated Summary",
        value=samples[sample_key].get('generated_summary', ''),
        height=200,
        label_visibility="collapsed"
    )

with col2:
    st.subheader("Reference Summary")
    reference_summary = st.text_area(
        "Reference Summary",
        value=samples[sample_key].get('reference_summary', ''),
        height=200,
        label_visibility="collapsed"
    )

# Source document (if available)
source = samples[sample_key].get('source')
if source:
    with st.expander("Source Document"):
        st.text_area("Source", value=source, height=150, disabled=True, label_visibility="collapsed")

st.divider()

# Run evaluation
if st.button("🚀 Run Agent Evaluation", type="primary", use_container_width=True):
    if not generated_summary.strip():
        st.error("Please provide a generated summary to evaluate.")
    else:
        with st.spinner("Connecting to H2OGPTE and running agent evaluation..."):
            try:
                # Create client
                status = st.empty()
                status.info("Creating H2OGPTE client...")
                client = create_client()

                # Setup collection
                status.info("Setting up collection and tools...")
                setup_collection(client, agent_type)

                # Run evaluation
                status.info("Running agent evaluation (this may take a few minutes)...")
                response = run_evaluation(
                    client=client,
                    generated_summary=generated_summary,
                    reference_summary=reference_summary if reference_summary.strip() else None,
                    source=source
                )

                status.empty()

                # Display results
                st.subheader("Agent Response")
                st.markdown(response)

            except Exception as e:
                st.error(f"Evaluation failed: {e}")

# Info section
with st.expander("ℹ️ How it works"):
    st.markdown("""
    **Code Execution Mode (`agent`):**
    - Uploads `tool_logic.py` to H2OGPTE
    - Agent executes Python code to run metrics
    - Direct access to all 28 evaluation metrics

    **MCP Server Mode (`agent_with_mcp`):**
    - Uses Model Context Protocol for structured tool access
    - Requires MCP bundle to be created first
    - More structured API for tool invocation

    **Available Metrics:**
    - Word Overlap: ROUGE, BLEU, METEOR, chrF++, Levenshtein
    - Semantic: BERTScore, MoverScore
    - Factuality: NLI, AlignScore, FactCC, Entity Coverage
    - Completeness: Semantic Coverage, BERTScore Recall
    - Fluency: Perplexity
    """)
