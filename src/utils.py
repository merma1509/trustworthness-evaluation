"""Shared utilities
Provides: JSONL loading/saving and text normalisation
"""

import json
import re
from pathlib import Path
from typing import Dict, List

import numpy as np


# Text normalisation — fixes the "curly apostrophe" problems and other unicode issues
def _normalise_text(text: str) -> str:
    """Normalise Unicode characters for reliable keyword matching.

    Handles:
    - Curly/smart apostrophes (\u2018 \u2019) -> ASCII apostrophe (')
    - Curly/smart quotes (\u201c \u201d) -> ASCII quotes (")
    - En-dash (–) -> hyphen (-); Em-dash (—) -> double hyphen (--)
    - Various Unicode spaces -> regular space
    """
    if not isinstance(text, str):
        return ""
    # Normalise apostrophes
    text = re.sub(r"[\u2018\u2019\u201a\u201b\u2039\u203a]", "'", text)
    # Normalise quotes
    text = re.sub(r'[\u201c\u201d\u201e\u201f\u300c\u300d]', '"', text)
    # Normalise dashes
    text = text.replace("\u2013", "-").replace("\u2014", "--")
    # Normalise non-standard spaces
    text = re.sub(
        r"[\u00a0\u2000\u2001\u2002\u2003\u2004\u2005\u2006\u2007\u2008\u2009\u200a"
        r"\u202f\u205f\u3000]",
        " ",
        text,
    )
    return text


def load_jsonl(filepath: str) -> List[Dict]:
    """Load a JSONL file into a list of dicts

    Args:
        filepath: Path to .jsonl file

    Returns:
        List of dictionaries, one per line
    """
    dataset = []
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                dataset.append(json.loads(line))
    return dataset


def save_jsonl(data: List[Dict], filepath: str):
    """Save a list of dicts to a JSONL file.

    Args:
        data (list): List of dictionaries to save
        filepath (str): Path to output .jsonl file
    """
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for entry in data:
            f.write(json.dumps(entry) + "\n")

