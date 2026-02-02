"""
MCP Dashboard Page - Build bundles and test MCP tools.
"""

import os
import sys

# Add paths for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import streamlit as st

st.set_page_config(
    page_title="MCP Dashboard",
    page_icon="🔌",
    layout="wide"
)

st.title("🔌 MCP Server Dashboard")
st.markdown("Build MCP bundles and test evaluation tools.")

# Import tool functions
try:
    from src.evaluators.tool_logic import (
        list_available_metrics,
        get_recommended_metrics,
        run_metric,
        get_metric_info
    )
except ImportError as e:
    st.error(f"Failed to import tool_logic: {e}")
    st.stop()

# Bundle Section
st.header("📦 Bundle Management")

col1, col2 = st.columns([2, 1])

with col1:
    bundle_path = os.path.join(os.path.dirname(__file__), '..', '..', 'mcp_server', 'sum_omni_eval_mcp.zip')
    if os.path.exists(bundle_path):
        size_mb = os.path.getsize(bundle_path) / (1024 * 1024)
        st.success(f"✅ Bundle exists: `sum_omni_eval_mcp.zip` ({size_mb:.2f} MB)")
    else:
        st.warning("⚠️ No bundle found. Create one to use MCP mode.")

with col2:
    if st.button("🔨 Build Bundle", use_container_width=True):
        with st.spinner("Building MCP bundle..."):
            try:
                from mcp_server.bundle import build_mcp_zip
                zip_path = build_mcp_zip()
                st.success(f"✅ Bundle created!")
                st.rerun()
            except Exception as e:
                st.error(f"Build failed: {e}")

st.divider()

# Tool Testing Section
st.header("🧪 Tool Testing")

tab1, tab2, tab3 = st.tabs(["List Metrics", "Run Metric", "Get Recommendations"])

with tab1:
    st.subheader("Available Metrics")
    if st.button("📋 List All Metrics", key="list_btn"):
        metrics = list_available_metrics()

        # Group by category
        by_category = {}
        for m in metrics:
            cat = m.get('category', 'Other')
            if cat not in by_category:
                by_category[cat] = []
            by_category[cat].append(m)

        for category, metric_list in by_category.items():
            with st.expander(f"**{category}** ({len(metric_list)} metrics)", expanded=True):
                for m in metric_list:
                    req = []
                    if m.get('requires_source'):
                        req.append("source")
                    if m.get('requires_reference'):
                        req.append("reference")
                    req_str = f" (requires: {', '.join(req)})" if req else ""
                    st.markdown(f"- **{m['name']}**: {m['description']}{req_str}")

with tab2:
    st.subheader("Run a Single Metric")

    metrics = list_available_metrics()
    metric_names = [m['name'] for m in metrics]

    selected_metric = st.selectbox("Select Metric", metric_names)

    # Get metric info for requirements
    metric_info = get_metric_info(selected_metric)

    if metric_info:
        st.caption(f"**{metric_info['description']}**")
        if metric_info.get('requires_source'):
            st.caption("⚠️ Requires source document")
        if metric_info.get('requires_reference'):
            st.caption("⚠️ Requires reference summary")

    summary_input = st.text_area("Summary to Evaluate", height=100, key="summary_input")

    col1, col2 = st.columns(2)
    with col1:
        source_input = st.text_area("Source Document (optional)", height=100, key="source_input")
    with col2:
        reference_input = st.text_area("Reference Summary (optional)", height=100, key="reference_input")

    if st.button("▶️ Run Metric", type="primary", key="run_btn"):
        if not summary_input.strip():
            st.error("Please provide a summary to evaluate.")
        else:
            with st.spinner(f"Running {selected_metric}..."):
                result = run_metric(
                    metric_name=selected_metric,
                    summary=summary_input,
                    source=source_input if source_input.strip() else None,
                    reference_summary=reference_input if reference_input.strip() else None
                )

                if 'error' in result:
                    st.error(f"Error: {result['error']}")
                else:
                    st.subheader("Results")

                    # Display scores
                    if result.get('scores'):
                        st.json(result['scores'])

                    # Display interpretation
                    if result.get('interpretation'):
                        st.info(f"**Interpretation:** {result['interpretation']}")

with tab3:
    st.subheader("Get Recommended Metrics")

    col1, col2, col3 = st.columns(3)
    with col1:
        has_source = st.checkbox("Has Source Document", value=True)
    with col2:
        has_reference = st.checkbox("Has Reference Summary", value=False)
    with col3:
        quick_mode = st.checkbox("Quick Mode (fast metrics only)", value=False)

    if st.button("💡 Get Recommendations", key="recommend_btn"):
        recommended = get_recommended_metrics(
            has_source=has_source,
            has_reference=has_reference,
            quick_mode=quick_mode
        )

        st.subheader("Recommended Metrics")
        for metric in recommended:
            info = get_metric_info(metric)
            if info:
                st.markdown(f"- **{metric}**: {info['description']}")
            else:
                st.markdown(f"- **{metric}**")

st.divider()

# Info section
with st.expander("ℹ️ About MCP Server"):
    st.markdown("""
    **Model Context Protocol (MCP)** provides a standardized way for AI agents to interact with tools.

    **MCP Tools Available:**
    - `list_metrics()` - List all available evaluation metrics
    - `recommend_metrics(has_source, has_reference, quick)` - Get recommended metrics
    - `run_single_metric(metric_name, summary, source, reference)` - Run one metric
    - `run_multiple(metrics, summary, source, reference)` - Run multiple metrics
    - `get_info(metric_name)` - Get detailed metric information

    **Usage:**
    1. Build the MCP bundle using the button above
    2. Use the Agent Evaluation page with "MCP Server" mode
    3. Or integrate the MCP server in your own applications
    """)
