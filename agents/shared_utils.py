"""
Shared utilities for agent orchestration.
"""

import os
import json
from jinja2 import Template, Environment, FileSystemLoader

# Base paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
DATA_FILE = os.path.join(PROJECT_ROOT, 'data', 'processed', 'cnn_dm_sample_with_gen_sum.json')
PROMPTS_DIR = os.path.join(BASE_DIR, 'prompts')


def load_prompt(prompt_name: str) -> str:
    """
    Load a simple prompt from the prompts directory.

    Args:
        prompt_name: Name of the prompt file (e.g., 'system.md')

    Returns:
        The prompt content as a string.
    """
    prompt_path = os.path.join(PROMPTS_DIR, prompt_name)

    if not os.path.exists(prompt_path):
        raise FileNotFoundError(f"Prompt file not found: '{prompt_path}'")

    with open(prompt_path, 'r', encoding='utf-8') as f:
        return f.read()
    

def render_dynamic_prompt(template_name: str, **kwargs) -> str:
    """
    Helper to render Jinja2 templates from a directory.
    
    Args:
        template_name: Name of the template file (e.g., 'user.md')
        **kwargs: Variables to pass into the template.
    
    Returns:
        Rendered template as a string.
    """
    env = Environment(loader=FileSystemLoader(PROMPTS_DIR)) 
    template = env.get_template(template_name)
    return template.render(**kwargs)


def load_summaries(sample_idx: int = None) -> dict:
    """
    Load summaries from the sample summaries JSON file.

    Args:
        sample_idx: Optional index to load a specific sample.
                   If None, returns all samples.

    Returns:
        Dictionary containing the sample(s).
    """
    if not os.path.exists(DATA_FILE):
        raise FileNotFoundError(f"Data file not found: '{DATA_FILE}'")

    with open(DATA_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if sample_idx is not None:
        if sample_idx < 0 or sample_idx >= len(data):
            raise IndexError(f"Sample index {sample_idx} out of range. Valid: 0-{len(data)-1}")
        return data[sample_idx]

    return data
