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
TOOL_FILENAME = 'tool_logic.py'
TOOL_FILE = os.path.join(BASE_DIR, '..', '..', 'src', 'evaluators', TOOL_FILENAME)
SERVER_FILENAME = 'sum_omni_eval_mcp.zip'
SERVER_FILE = os.path.join(BASE_DIR, '..', '..', 'mcp_server', SERVER_FILENAME)

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


def _setup_agent_keys(client: H2OGPTE) -> None:
    """Ensure agent keys for MCP env vars exist, reusing or creating as needed."""
    required_keys = {
        "H2OGPTE_API_KEY": os.getenv('H2OGPTE_API_KEY'),
        "H2OGPTE_ADDRESS": os.getenv('H2OGPTE_ADDRESS'),
    }

    # Check for existing keys
    existing = {k['name']: k['id'] for k in client.get_agent_keys()
                if k['name'] in required_keys}

    # Create missing keys
    for name, value in required_keys.items():
        if name not in existing:
            result = client.add_agent_key([
                {"name": name, "value": value, "key_type": "private",
                 "description": f"{name} for MCP server"}
            ])
            existing[name] = result[0]["agent_key_id"]
            print(f"  Created agent key: {name}")
        else:
            print(f"  Reusing agent key: {name}")

    # Associate keys with the MCP tool
    client.assign_agent_key_for_tool([{
        "tool_dict": {
            "tool": SERVER_FILENAME,
            "keys": [{"name": name, "key_id": kid}
                     for name, kid in existing.items()]
        }
    }])
    print("Agent keys associated with MCP tool")


def setup_collection(client: H2OGPTE, agent_type: str) -> str:
    """Create collection and ingest the evaluation tool."""
    # Create collection based on agent type
    if agent_type == "agent":
        collection_id = client.create_collection(
        name='SumOmniEval Agent Only',
        description='H2OGPTE Agent: Evaluate summaries using SumOmniEval metrics through tool-calling.',
    )
    else:  # agent_with_mcp
        collection_id = client.create_collection(
        name='SumOmniEval Agent with MCP',
        description='H2OGPTE Agent: Evaluate summaries using SumOmniEval metrics through local MCP server.',
    )
    print(f"Collection created: {collection_id}")

    # Select file based on agent type
    if agent_type == "agent":
        tool_path = TOOL_FILE
        tool_filename = TOOL_FILENAME
    else:  # agent_with_mcp
        tool_path = SERVER_FILE
        tool_filename = SERVER_FILENAME

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
                'tool_name': TOOL_FILENAME,
                'enable_by_default': False,
                'tool_usage_mode': 'runner'
            },
            custom_tool_path=TOOL_FILE
        )
    else:  # agent_with_mcp
        client.add_custom_agent_tool(
            tool_type='local_mcp',
            tool_args={
                'tool_name': SERVER_FILENAME,
                'enable_by_default': False,
                'tool_usage_mode': 'runner'
            },
            custom_tool_path=SERVER_FILE
        )

        # Ensure agent keys exist and associate them with the MCP tool
        # so environment variables are injected into the MCP server process
        _setup_agent_keys(client)

    return collection_id


def run_evaluation(collection_id: str, client: H2OGPTE, agent_type: str, generated_summary: str, reference_summary: str = None, source: str = None) -> str:
    """Run the agent evaluation on the given summaries."""
    chat_session_id = client.create_chat_session(collection_id)
    print(f"Chat session created: {chat_session_id}")

    # Select tool based on agent type
    tool_name = ""
    if agent_type == "agent":
        tool_name = TOOL_FILENAME
    else:  # agent_with_mcp
        tool_name = SERVER_FILENAME

    # Select agent type
    agent_type_str = "auto"
    if agent_type == "agent":
        agent_type_str = "general"
    else:  # agent_with_mcp
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

    # Load system prompt based on agent type
    system_prompt = load_prompt('system_base.md')
    if agent_type == "agent":
        system_prompt += "\n" + load_prompt('system_agent.md')
    else:  # agent_with_mcp
        system_prompt += "\n" + load_prompt('system_mcp.md')

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


def main(sample_idx: str = "0", agent_type: str = "agent", data_file: str = None):
    """Main entry point."""
    # Load sample data
    sample = load_summaries(sample_idx=int(sample_idx), data_file=data_file)
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
    parser.add_argument("--data-file", default=None,
                        help="Path to custom data file. Default: data/processed/cnn_dm_sample_with_gen_sum.json")
    parser.add_argument("--list", action="store_true", help="List available samples")

    args = parser.parse_args()

    if args.list:
        samples = load_summaries(data_file=args.data_file)
        data_file_display = args.data_file if args.data_file else "data/processed/cnn_dm_sample_with_gen_sum.json"
        print(f"Available samples from: {data_file_display}")
        for idx, sample in enumerate(samples):
            sample_id = sample.get('id', f'sample_{idx}')
            print(f"  - {idx}: {sample_id}")
        print(f"\nTotal samples: {len(samples)}")
    else:
        main(args.sample_idx, args.agent_type, args.data_file)
