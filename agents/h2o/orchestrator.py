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
from shared_utils import load_prompt, load_summaries

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
            tool_args={'tool_name': 'tool_logic'},
            custom_tool_path=TOOL_FILE
        )
    else:  # agent_with_mcp
        client.add_custom_agent_tool(
            tool_type='local_mcp',
            tool_args={'tool_name': 'sum_omni_eval_mcp'},
            custom_tool_path=SERVER_FILE
        )

    return collection_id


def run_evaluation(client: H2OGPTE, generated_summary: str, reference_summary: str = None, source: str = None) -> str:
    """Run the agent evaluation on the given summaries."""
    chat_session_id = client.create_chat_session()
    print(f"Chat session created: {chat_session_id}")

    # Load prompts
    system_prompt = load_prompt('system.md')
    user_prompt_template = load_prompt('user.md')

    user_prompt = user_prompt_template.format(
        generated_summary=generated_summary,
        reference_summary=reference_summary or "Not provided",
        source=source or "Not provided",
    )

    print("Running agent query...")
    with client.connect(chat_session_id) as session:
        reply = session.query(
            user_prompt,
            llm_args=dict(
                use_agent=True,
                agent_type="auto",
                agent_code_writer_system_message=system_prompt
            )
        )

    return reply.content


def main(sample_key: str = "grenfell_tower", agent_type: str = "agent"):
    """Main entry point."""
    # Load sample data
    sample = load_summaries(sample_key)
    print(f"Loaded sample: {sample.get('name', sample_key)}")

    # Create client and setup
    client = create_client()
    setup_collection(client, agent_type)

    # Run evaluation
    response = run_evaluation(
        client=client,
        generated_summary=sample['generated_summary'],
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
    parser.add_argument("--sample", default="grenfell_tower", help="Sample key from data/sample_summaries.json")
    parser.add_argument("--list", action="store_true", help="List available samples")

    args = parser.parse_args()

    if args.list:
        samples = load_summaries()
        print("Available samples:")
        for key, value in samples.items():
            print(f"  - {key}: {value.get('name', 'Unnamed')}")
    else:
        main(args.sample, args.agent_type)
