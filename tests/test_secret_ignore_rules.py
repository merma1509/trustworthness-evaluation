"""Secret-material ignore rules must hold at any depth.

``experiment-seal-content-aware`` seals into ``experiment/sealed/<experiment-id>/``
and writes the keeper passphrase there:

    experiment/sealed/<id>/.sealed_keeper.json

The ignore rule was ``experiment/.sealed_keeper.*``, which is anchored to the
top level only. The keeper in the new subdirectory was therefore *trackable* --
``git add`` would have committed the only plaintext copy of the passphrase that
decrypts the sealed labels. Caught before committing, but the rule had to be
widened, not just worked around.

The companion rule (labels directory) matters for the same reason: ``*.enc``
must stay out of git, while the plaintext ``auto_labels.jsonl`` is committed so
gold can be rebuilt without the passphrase.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


def _is_ignored(rel_path: str) -> bool:
    """True when git would NOT track the path (i.e. it is ignored)."""
    result = subprocess.run(
        ["git", "check-ignore", "-q", rel_path],
        cwd=ROOT, capture_output=True,
    )
    if result.returncode not in (0, 1):
        pytest.skip("git not available")
    return result.returncode == 0


@pytest.mark.parametrize(
    "path",
    [
        "experiment/.sealed_keeper.json",
        # The regression: a keeper inside a re-do experiment subdirectory.
        "experiment/sealed/some-experiment-id/.sealed_keeper.json",
        "experiment/sealed/some-experiment-id/labels/sealed_auto_labels.jsonl.enc",
        "experiment/sealed/labels/sealed_auto_labels.jsonl.enc",
    ],
)
def test_secret_material_is_ignored(path: str):
    assert _is_ignored(path), f"{path} would be committed — secret material leaked"


def test_plaintext_auto_labels_stay_tracked():
    """Gold must be rebuildable from a clone without the passphrase."""
    assert not _is_ignored("experiment/sealed/labels/auto_labels.jsonl")
    assert not _is_ignored(
        "experiment/sealed/some-experiment-id/labels/auto_labels.jsonl"
    )
