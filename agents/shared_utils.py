"""
Shared utilities for agent orchestration.
"""

import os
import json

# Base paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
DATA_FILE = os.path.join(PROJECT_ROOT, 'data', 'sample_summaries.json')
PROMPTS_DIR = os.path.join(BASE_DIR, 'prompts')


def load_prompt(prompt_name: str) -> str:
    """
    Load a prompt from the prompts directory.

    Args:
        prompt_name: Name of the prompt file (e.g., 'system.md', 'user.md')

    Returns:
        The prompt content as a string.
    """
    prompt_path = os.path.join(PROMPTS_DIR, prompt_name)

    if not os.path.exists(prompt_path):
        raise FileNotFoundError(f"Prompt file not found: '{prompt_path}'")

    with open(prompt_path, 'r', encoding='utf-8') as f:
        return f.read()


def load_summaries(sample_key: str = None) -> dict:
    """
    Load summaries from the sample summaries JSON file.

    Args:
        sample_key: Optional key to load a specific sample.
                   If None, returns all samples.

    Returns:
        Dictionary containing the sample(s).
    """
    if not os.path.exists(DATA_FILE):
        raise FileNotFoundError(f"Data file not found: '{DATA_FILE}'")

    with open(DATA_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if sample_key:
        if sample_key not in data:
            raise KeyError(f"Sample '{sample_key}' not found. Available: {list(data.keys())}")
        return data[sample_key]

    return data
