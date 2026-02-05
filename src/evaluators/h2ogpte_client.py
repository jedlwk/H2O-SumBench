#!/usr/bin/env python3
"""
Shared H2OGPTE client module for all evaluators.

Reads credentials from environment variables (H2OGPTE_API_KEY, H2OGPTE_ADDRESS).
For MCP servers, these are passed via envs.json configuration.
"""

import os
from typing import Optional, Tuple

# Module-level client cache
_client = None


def get_credentials() -> Tuple[Optional[str], Optional[str]]:
    """
    Get API credentials from environment variables.

    Returns:
        Tuple of (api_key, address)
    """
    return os.environ.get('H2OGPTE_API_KEY'), os.environ.get('H2OGPTE_ADDRESS')


def get_h2ogpte_client():
    """
    Get or create the H2OGPTE client.

    Returns:
        H2OGPTE client instance

    Raises:
        ValueError: If API credentials are not available
    """
    global _client

    api_key, address = get_credentials()

    if not api_key or not address:
        raise ValueError(
            "H2OGPTE credentials not configured. "
            "Set H2OGPTE_API_KEY and H2OGPTE_ADDRESS environment variables."
        )

    if _client is None:
        from h2ogpte import H2OGPTE
        _client = H2OGPTE(address=address, api_key=api_key)

    return _client


def is_configured() -> bool:
    """Check if credentials are available."""
    api_key, address = get_credentials()
    return bool(api_key and address)


def reset_client() -> None:
    """Reset the cached client."""
    global _client
    _client = None
