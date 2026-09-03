#!/usr/bin/env python3
"""onboard_raters.py — Generate rater declaration forms + annotation templates

Each rater (RATER_A, RATER_B, ADJUDICATOR) gets:
    * a declaration FORM template (to be signed before annotation), and
    * a per-rater, per-split annotated template that they MAIL back

The annotation template is created by copying the sealed template and adding the
exact fields the rater may fill (the label-constrained; schema)

Usage:
    python3 scripts/onboard_raters.py \
        --experiment-dir experiment/sealed \
        --out-experiment experiment \
        --raters RATER_A RATER_B ADJUDICATOR \
        [--split calibration heldout]

Output:
    experiment/rater_declarations/{RATER}_declaration_form.md   (to sign)
    experiment/annotations/{experiment}_{RATER}_{split}.jsonl  (blank template)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.labels import label_set_for

# Which split(s) to prepare per rater.
DEFAULT_SPLITS = ["calibration", "heldout"]

RATER_POOL = ["RATER_A", "RATER_B", "ADJUDICATOR"]


def _find_experiment_id(base_dir: Path) -> str:
    """Locate the experiment id from the most recent sealing manifest."""
    manifests = sorted(base_dir.glob("manifests/*.json"))
    if not manifests:
        raise FileNotFoundError(f"No sealing manifest under {base_dir / 'manifests'}")
    m = json.loads(manifests[-1].read_text())
    return m["experiment_id"]


def _find_split_template(templates_dir: Path, experiment_id: str, split: str) -> Path:
    candidates = list(templates_dir.glob(f"{experiment_id}_{split}_template.jsonl"))
    if not candidates:
        raise FileNotFoundError(
            f"No sealed {split} template for {experiment_id} under {templates_dir}"
        )
    return candidates[0]


def declaration_form(rater_id: str, experiment_id: str, rubric_path: Path) -> str:
    """Render the rater agreement form"""
    return f"""# RATER DECLARATION FORM — {rater_id}

**Experiment:** {experiment_id}

I, ____________________________, agree to the following:

1. **Independence**: I have NOT seen the model responses I will annotate before
   this session, nor any automatic scores or expected labels associated with them.

2. **No AI Assistance**: I have NOT used any LLM, AI assistant, or external tool
   to assist with my annotations. If I do, I will immediately declare it.

3. **Rubric Adherence**: I will follow the rubric at
   `docs/MANUAL_AUDIT_RUBRIC_v3.md` (version: rubric-v3). I will NOT discuss
   specific cases with other raters during annotation.

4. **Sealed Template**: I will annotate only the responses in my assigned sealed
   template file, without cross-referencing other files.

5. **Difficult Cases**: I understand that I should NOT skip or discard difficult
   cases. If I am uncertain, I will mark `rater_uncertainty: uncertain` and note
   my reasoning.

**Label sets you may assign (per dimension):**
- **safety**       : {', '.join(label_set_for('safety'))}
- **truthfulness** : {', '.join(label_set_for('truthfulness'))}
- **consistency**  : {', '.join(label_set_for('consistency'))}

Rater ID: {rater_id}
Date: __________
Signature: __________
"""


def _annotatable_template(sealed_rec: dict, experiment_id: str, rater_id: str,
                          split: str) -> dict:
    """Add the fillable annotation fields to a sealed template record

    The sealed template already carries the correct structure; the rater fills:
      * the dimension's label (ONE value),
      * `rater_uncertainty`,
      * `ai_assistance_used` (must stay false to be valid),
      * optional `notes`.

    Metadata is stamped so an ingestion side can detect which
    template_version/rubric the record was sealed under and verify nothing was
    tampered with.
    """
    rec = json.loads(json.dumps(sealed_rec))  # deep copy
    rec["annotation_id"] = ""
    rec["rater_id"] = rater_id
    rec["experiment_id"] = experiment_id
    rec["_split"] = split

    # Ensure all label fields are present and empty for the rater
    rec["annotations"].setdefault("notes", "")
    rec["annotations"]["label_source"] = "rater_judgment"
    return rec


def _emit_declaration(decl_dir: Path, rater_id: str, experiment_id: str,
                      rubric_path: Path) -> Path:
    decl_dir.mkdir(parents=True, exist_ok=True)
    p = decl_dir / f"{rater_id}_declaration_form.md"
    p.write_text(declaration_form(rater_id, experiment_id, rubric_path))
    return p


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-dir", default="experiment/sealed")
    parser.add_argument("--out-experiment", default="experiment")
    parser.add_argument("--raters", nargs="+", default=RATER_POOL)
    parser.add_argument("--splits", nargs="+", default=DEFAULT_SPLITS)
    args = parser.parse_args()

    base_dir = Path(args.experiment_dir)
    out_dir = Path(args.out_experiment)
    if not base_dir.exists():
        print(f"  Sealed experiment dir not found: {base_dir}")
        return 1

    experiment_id = _find_experiment_id(base_dir)
    templates_dir = base_dir / "templates"
    rubric_path = Path("docs/MANUAL_AUDIT_RUBRIC_v3.md")

    decl_dir = out_dir / "rater_declarations"
    annotations_dir = out_dir / "annotations"
    decl_dir.mkdir(parents=True, exist_ok=True)
    annotations_dir.mkdir(parents=True, exist_ok=True)

    # ── Prepare one annotation file per rater per split ──────────────────────
    prepared = []
    for split in args.splits:
        template_path = _find_split_template(templates_dir, experiment_id, split)
        sealed_recs = [
            json.loads(line) for line in template_path.open() if line.strip()
        ]
        for rater in args.raters:
            out_path = annotations_dir / f"{experiment_id}_{rater}_{split}.jsonl"
            with out_path.open("w") as f:
                for rec in sealed_recs:
                    f.write(
                        json.dumps(
                            _annotatable_template(rec, experiment_id, rater, split),
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
            prepared.append((rater, split, out_path, len(sealed_recs)))

    # ── Emit declaration forms ────────────────────────────────────────────────
    decl_forms = {}
    for rater in args.raters:
        p = _emit_declaration(decl_dir, rater, experiment_id, rubric_path)
        decl_forms[rater] = str(p.relative_to(out_dir))

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"  Experiment: {experiment_id}")
    for rater, split, path, n in prepared:
        print(f"  {rater:<10} {split:<12} {n:>3} records -> {path.relative_to(out_dir)}")
    for rater, rel in decl_forms.items():
        print(f"  Declaration form {rater:<12} -> {rel}")

    print("\n  NEXT: raters sign their declaration, then fill their annotation file")
    print("  Labels are schema-validated on ingestion — no free-text labels accepted")
    return 0


if __name__ == "__main__":
    sys.exit(main())
