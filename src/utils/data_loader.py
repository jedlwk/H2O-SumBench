"""Data loading utilities for sample data."""

import pandas as pd
import json
import os
from typing import Dict, Optional

# Load sample data as default
DEFAULT_DATA = 'cnn_dm_sample_with_gen_sum.json'


def load_sample_data(data_path: Optional[str] = None) -> pd.DataFrame:
    """
    Load sample data from CSV or JSON file.

    Args:
        data_path: Path to the CSV or JSON file. If None, uses default location.

    Returns:
        DataFrame with 'report' and 'summary' columns.

    Raises:
        FileNotFoundError: If the data file doesn't exist.
        ValueError: If the file doesn't have required columns or invalid format.
    """
    if data_path is None:
        # Default path relative to project root
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(current_dir))
        data_path = os.path.join(project_root, 'data', 'processed', DEFAULT_DATA)

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Sample data file not found at: {data_path}")

    # Determine file type by extension
    file_ext = os.path.splitext(data_path)[1].lower()

    try:
        if file_ext == '.json':
            # Load JSON file
            with open(data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Convert to DataFrame
            if isinstance(data, list):
                df = pd.DataFrame(data)
            else:
                df = pd.DataFrame([data])

            # Map common column names to expected format
            # Support both 'source' and 'report' for source text
            if 'source' in df.columns and 'report' not in df.columns:
                df['report'] = df['source']

        elif file_ext == '.csv':
            df = pd.read_csv(data_path)
        else:
            raise ValueError(f"Unsupported file format: {file_ext}. Only .csv and .json are supported.")

    except json.JSONDecodeError as e:
        raise ValueError(f"Error parsing JSON file: {e}")
    except Exception as e:
        raise ValueError(f"Error reading file: {e}")

    # Validate required columns
    required_columns = ['report', 'summary']
    if not all(col in df.columns for col in required_columns):
        raise ValueError(
            f"File must contain columns: {required_columns}. "
            f"Found: {list(df.columns)}"
        )

    return df


def get_sample_by_index(index: int, data_path: Optional[str] = None) -> Dict[str, str]:
    """
    Get a specific sample by index.

    Args:
        index: Zero-based index of the sample to retrieve.
        data_path: Path to the CSV file. If None, uses default location.

    Returns:
        Dictionary with 'source' and 'summary' keys.

    Raises:
        IndexError: If index is out of bounds.
    """
    df = load_sample_data(data_path)

    if index < 0 or index >= len(df):
        raise IndexError(
            f"Index {index} out of bounds. "
            f"Valid range: 0 to {len(df) - 1}"
        )

    row = df.iloc[index]

    result = {
        'source': row['report'],
        'summary': row['summary']
    }

    # Include reference if available (support both column names)
    if 'reference_summary' in df.columns:
        result['reference'] = row['reference_summary']
    elif 'reference' in df.columns:
        result['reference'] = row['reference']

    return result


def get_sample_titles(data_path: Optional[str] = None, max_length: int = 100) -> list:
    """
    Get abbreviated titles for each sample (first N chars of source).

    Args:
        data_path: Path to the CSV file. If None, uses default location.
        max_length: Maximum length of each title.

    Returns:
        List of abbreviated titles.
    """
    df = load_sample_data(data_path)

    titles = []
    for idx, row in df.iterrows():
        source = row['report']
        # Extract first sentence or first max_length characters
        first_part = source[:max_length].strip()
        if len(source) > max_length:
            first_part += "..."
        titles.append(f"Sample {idx + 1}: {first_part}")

    return titles
