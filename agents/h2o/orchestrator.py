#!/usr/bin/env python3
"""
Full H2OGPTE Agent Test with All Evaluation Metrics.

This test demonstrates using H2OGPTE agent with all summary evaluation metrics
from SumOmniEval. The agent decides which metrics to use based on:
1. What the user wants to evaluate (factuality, fluency, completeness, etc.)
2. Whether a reference summary is available
"""

import os
import sys
import time

# Add paths for imports (must be before local imports)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from dotenv import load_dotenv
from h2ogpte import H2OGPTE
from shared_utils import load_prompt, load_summaries, render_dynamic_prompt

BASE_DIR = os.path.dirname(__file__)
TOOL_FILE = os.path.join(BASE_DIR, '..', '..', 'src', 'evaluators', 'tool_logic.py')
SERVER_FILE = os.path.join(BASE_DIR, '..', '..', 'mcp_server', 'sum_omni_eval_mcp.zip')

def create_client() -> H2OGPTE:
    """Create and return H2OGPTE client."""
    load_dotenv()

    api_key = os.getenv('H2OGPTE_API_KEY')
    address = os.getenv('H2OGPTE_ADDRESS')

    if not api_key or not address:
        raise ValueError("H2OGPTE_API_KEY and H2OGPTE_ADDRESS must be set in .env file")

    print(f"Connecting to {address}...")
    client = H2OGPTE(address=address, api_key=api_key)
    print("Client created successfully")
    return client


def setup_collection(client: H2OGPTE, agent_type: str) -> str:
    """Create collection and ingest the evaluation tool."""
    collection_id = client.create_collection(
        name='SumOmniEval Full Agent',
        description='H2OGPTE Agent with All Summary Evaluation Metrics',
    )
    print(f"Collection created: {collection_id}")

    # Select file based on agent type
    if agent_type == "agent":
        tool_path = TOOL_FILE
        tool_filename = 'tool_logic.py'
    else:  # agent_with_mcp
        tool_path = SERVER_FILE
        tool_filename = 'sum_omni_eval_mcp.zip'

    # Upload the evaluation tool file
    with open(tool_path, 'rb') as f:
        upload_id = client.upload(tool_filename, f)

    ingest_job = client.ingest_uploads(
        collection_id=collection_id,
        upload_ids=[upload_id],
        ingest_mode='agent_only'
    )

    print(f"Waiting for {tool_filename} ingestion...")
    while True:
        job_status = client.get_job(ingest_job.id)
        if job_status.completed:
            print("Ingestion complete")
            break
        if job_status.failed:
            raise RuntimeError(f"Ingestion failed: {job_status.errors}")
        time.sleep(2)

    # Register the custom tool
    if agent_type == "agent":
        client.add_custom_agent_tool(
            tool_type='general_code',
            tool_args={
                'tool_name': 'tool_logic',
                'enable_by_default': False,
                'tool_usage_mode': 'runner'
            },
            custom_tool_path=TOOL_FILE
        )
    else:  # agent_with_mcp
        client.add_custom_agent_tool(
            tool_type='local_mcp',
            tool_args={
                'tool_name': 'sum_omni_eval_mcp',
                'enable_by_default': False,
                'tool_usage_mode': 'runner'
            },
            custom_tool_path=SERVER_FILE
        )

    return collection_id


def run_evaluation(collection_id: str, client: H2OGPTE, agent_type: str, generated_summary: str, reference_summary: str = None, source: str = None) -> str:
    """Run the agent evaluation on the given summaries."""
    chat_session_id = client.create_chat_session(collection_id)
    print(f"Chat session created: {chat_session_id}")

    # Select tool based on agent type
    tool_name = ""
    if agent_type == "agent":
        tool_name = "tool_logic"
    else:
        tool_name = "sum_omni_eval_mcp"

    # Select agent type
    agent_type_str = "auto"
    if agent_type == "agent":
        agent_type_str = "general"
    else:
        agent_type_str = "mcp_tool_runner"

    # Warmup: ensure MCP server is fully initialized before running evaluation
    if agent_type == "agent_with_mcp":
        max_retries = 3
        retry_delay = 5  # seconds

        for attempt in range(1, max_retries + 1):
            print(f"Warming up MCP server (attempt {attempt}/{max_retries})...")
            with client.connect(chat_session_id) as session:
                warmup_reply = session.query(
                    message="Call check_env_var to verify the MCP server is ready. Only respond with SUCCESS or FAILURE.",
                    llm_args=dict(
                        use_agent=True,
                        agent_type=agent_type_str,
                        agent_tools=[tool_name]
                    )
                )

            response_text = warmup_reply.content.upper()
            if "SUCCESS" in response_text:
                print("MCP server ready - environment variables configured")
                break
            else:
                print(f"MCP server not ready: {warmup_reply.content[:100]}...")
                if attempt < max_retries:
                    print(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                else:
                    raise RuntimeError(
                        f"MCP server failed to initialize after {max_retries} attempts. "
                        "Environment variables may not be configured correctly."
                    )

    # Load prompts
    system_prompt = load_prompt('system.md')

    # Render dynamic prompt
    user_prompt = render_dynamic_prompt(
        'user.md',
        generated_summary=generated_summary,
        reference_summary=reference_summary,
        source=source
    )

    print(f"Running agent query with tool: {tool_name} and agent type: {agent_type_str}")
    with client.connect(chat_session_id) as session:
        reply = session.query(
            message=user_prompt,
            system_prompt=system_prompt,
            llm_args=dict(
                use_agent=True,
                agent_type=agent_type_str,
                agent_tools=[tool_name]
            )
        )

    return reply.content


def main(sample_idx: str = "0", agent_type: str = "agent"):
    """Main entry point."""
    # Load sample data
    sample = load_summaries(sample_idx=int(sample_idx))
    sample_id = sample.get('id', sample_idx)
    print(f"Loaded sample: {sample_id}")

    # Create client and setup
    client = create_client()
    collection_id = setup_collection(client, agent_type)

    # Run evaluation
    response = run_evaluation(
        collection_id=collection_id,
        client=client,
        agent_type=agent_type,
        generated_summary=sample['summary'],
        reference_summary=sample.get('reference_summary'),
        source=sample.get('source')
    )

    print("\nAgent Response:")
    print(response)

    return response


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run H2OGPTE agent evaluation")
    parser.add_argument("--agent-type", choices=["agent", "agent_with_mcp"], default="agent",
                        help="Type of agent to use for evaluation")
    parser.add_argument("--sample-idx", default="0",
                        help="Sample index for JSON. Default: 0")
    parser.add_argument("--list", action="store_true", help="List available samples")

    args = parser.parse_args()

    if args.list:
        samples = load_summaries()
        print("Available samples:")
        for idx, sample in enumerate(samples):
            sample_id = sample.get('id', f'sample_{idx}')
            print(f"  - {idx}: {sample_id}")
        print(f"\nTotal samples: {len(samples)}")
    else:
        main(args.sample_idx, args.agent_type)
