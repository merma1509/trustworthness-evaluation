"""Re-sealing must never destroy an experiment it cannot recreate.

Sealing writes two artifacts to **fixed** filenames, independent of
``--experiment-id``:

    experiment/sealed/manifests/sealing_manifest.json
    experiment/sealed/labels/sealed_auto_labels.jsonl.enc

Templates, by contrast, are named per experiment. That asymmetry is a trap: a
second seal into the same directory leaves the old ``*_template.jsonl`` files in
place while overwriting the manifest and the encrypted labels. The result *looks*
like the old experiment still exists — but its labels can no longer be verified
or decrypted, and the newer manifest makes ``find_latest_experiment_id`` point
elsewhere. Nothing in the script refused this before.

The keeper compounds it: ``experiment/.sealed_keeper.json`` holds the only copy
of the passphrase and is gitignored, so overwriting it orphans the existing
``.enc`` permanently.

Pinned here:

1. sealing over an existing manifest / ``.enc`` is refused;
2. overwriting an existing keeper is refused;
3. a fresh directory still seals normally (the guard must not block real work);
4. ``--force`` remains the explicit escape hatch.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "results" / "raw_outputs"


def _seal(experiment_dir: Path, keeper: Path, *extra: str,
          experiment_id: str = "guard-test") -> subprocess.CompletedProcess:
    return subprocess.run(
        [
            sys.executable, "scripts/seal_experiment.py",
            "--raw", str(RAW),
            "--experiment-dir", str(experiment_dir),
            "--experiment-id", experiment_id,
            "--seed", "42",
            "--content-aware",
            "--generate-passphrase",
            "--keeper", str(keeper),
            *extra,
        ],
        cwd=ROOT, capture_output=True, text=True,
    )


@pytest.fixture()
def sealed_once(tmp_path: Path) -> Path:
    """A directory that already holds a sealed experiment."""
    exp_dir = tmp_path / "sealed"
    result = _seal(exp_dir, tmp_path / "keeper.json")
    assert result.returncode == 0, result.stdout + result.stderr
    return exp_dir


def test_fresh_directory_seals(sealed_once: Path):
    """The guard must not block first-time sealing."""
    assert (sealed_once / "manifests" / "sealing_manifest.json").exists()
    assert (sealed_once / "labels" / "sealed_auto_labels.jsonl.enc").exists()


def test_resealing_same_directory_is_refused(sealed_once: Path, tmp_path: Path):
    result = _seal(sealed_once, tmp_path / "keeper2.json", experiment_id="second")

    assert result.returncode == 1
    assert "REFUSING TO OVERWRITE" in result.stdout


def test_existing_artifacts_survive_the_refused_reseal(
    sealed_once: Path, tmp_path: Path
):
    """A refused run must leave the original bytes untouched."""
    manifest = sealed_once / "manifests" / "sealing_manifest.json"
    before = json.loads(manifest.read_text())

    _seal(sealed_once, tmp_path / "keeper2.json", experiment_id="second")

    assert json.loads(manifest.read_text()) == before, "manifest was modified"


def test_overwriting_an_existing_keeper_is_refused(tmp_path: Path):
    """A new passphrase must not orphan an existing .enc."""
    keeper = tmp_path / "keeper.json"
    keeper.write_text(json.dumps({"passphrase": "existing-secret"}))

    result = _seal(tmp_path / "sealed", keeper)

    assert result.returncode == 1
    assert "keeper" in result.stdout.lower()
    # The original passphrase must be intact.
    assert json.loads(keeper.read_text())["passphrase"] == "existing-secret"


def test_force_allows_deliberate_overwrite(sealed_once: Path, tmp_path: Path):
    result = _seal(sealed_once, tmp_path / "keeper2.json", "--force",
                   experiment_id="forced")

    assert result.returncode == 0, result.stdout + result.stderr
    manifest = json.loads(
        (sealed_once / "manifests" / "sealing_manifest.json").read_text()
    )
    assert manifest["experiment_id"] == "forced"
