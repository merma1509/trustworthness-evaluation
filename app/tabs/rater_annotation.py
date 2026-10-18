"""Tab: Rater Annotation — interactive annotation of sealed templates.

Lets a rater (RATER_A / RATER_B / ADJUDICATOR) fill the sealed templates from
the Streamlit UI instead of editing raw JSONL files. Each decision is saved
immediately to ``experiment/annotations/{experiment_id}_{rater}_{split}.jsonl``
in the exact schema required by ``scripts/ingest_annotations.py`` and
``scripts/resolve_disagreements.py``.

Design constraints (to keep the blinded protocol intact):
  * the UI shows ONLY the dimension, prompt text and response(s) — it never
    renders ``auto_label``, ``model_id``, ``attack_type``, ``prompt_id`` or any
    similarity/score field (locked out by the sealed template schema in any
    case),
  * the label input is constrained to the finite set for the record's dimension
    (``src/labels.VALID_LABELS``) — a free-text label is impossible,
  * ``flags.ai_assistance_used`` is never offered to the rater (it must remain
    ``false``), matching the binding declaration forms,
  * every record stays in the file (no-skip): only ``rater_uncertainty =
    uncertain`` or ``response_invalid = true`` can mark a hard case without
    deleting it.

The tab reads/writes per-rater annotation files only. It does NOT open the
sealed auto labels and does NOT show other raters' answers.
"""

import json
import re
import sys
from datetime import date, datetime
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.labels import VALID_LABELS  # noqa: E402

# Where the sealed experiment templates and per-rater annotations live.
# Choose the latest sealed experiment automatically (same logic as Makefile).
ANNOTATIONS_DIR = Path("experiment/annotations")
SEALED_DIR = Path("experiment/sealed")
DECLARATIONS_DIR = Path("experiment/rater_declarations")

# Which field carries the label per dimension (mirrors src/labels + scripts).
DIMENSION_LABEL_KEY = {
    "safety": "safety_label",
    "truthfulness": "truthfulness_label",
    "consistency": "consistency_label",
}

RATERS = ("RATER_A", "RATER_B", "ADJUDICATOR")
SPLITS = ("calibration", "heldout")
UNCERTAINTY_OPTIONS = ("confident", "uncertain", "guess")


# ─────────────────────────── file helpers ───────────────────────────────────
def _latest_experiment_id() -> Optional[str]:
    """Pick the most recent sealed experiment id from the sealing manifest."""
    manifests = sorted(SEALED_DIR.glob("manifests/*.json"))
    if not manifests:
        # Fall back to whatever annotation templates exist.
        files = list(ANNOTATIONS_DIR.glob("*_RATER_A_calibration.jsonl"))
        if not files:
            return None
        names = sorted(p.name for p in files)
        return names[-1].split("_RATER_A_calibration.jsonl")[0]
    m = json.loads(manifests[-1].read_text())
    return m.get("experiment_id")


def _annotation_path(experiment_id: str, rater: str, split: str) -> Path:
    return ANNOTATIONS_DIR / f"{experiment_id}_{rater}_{split}.jsonl"


def _load_records(experiment_id: str, rater: str, split: str) -> List[dict]:
    """Load the per-rater annotation file (created by onboard_raters / sealed)."""
    path = _annotation_path(experiment_id, rater, split)
    if not path.exists():
        return []
    return [json.loads(line) for line in path.open() if line.strip()]


def _dimension_of(rec: dict) -> str:
    pr = rec.get("prompt_record") or {}
    return (pr.get("dimension") or "").strip()


def _label_of(rec: dict) -> str:
    dim = _dimension_of(rec)
    key = DIMENSION_LABEL_KEY.get(dim)
    if not key:
        return ""
    return ((rec.get("annotations") or {}).get(key) or "").strip()


def _declaration_path(experiment_id: str, rater: str) -> Path:
    """Return the declaration-form file for a rater in the current experiment.

    ``onboard_raters.py`` writes a plain ``{rater}_declaration_form.md`` (one
    sheet per rater, reused across experiments). If a
    ``{experiment_id}_{rater}_declaration_form.md`` file exists instead, prefer
    it so per-experiment declarations are honoured.
    """
    candidates = [
        DECLARATIONS_DIR / f"{experiment_id}_{rater}_declaration_form.md",
        DECLARATIONS_DIR / f"{rater}_declaration_form.md",
    ]
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


def _load_declaration(experiment_id: str, rater: str) -> str:
    """Read the declaration-form markdown for ``rater`` ('' if missing)."""
    path = _declaration_path(experiment_id, rater)
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8")


def _declaration_signed(md_text: str) -> bool:
    r"""Heuristic: is the declaration form filled in (signed) rather than blank?

    ``onboard_raters.py`` emits a blank template whose blank fields are rendered
    as escaped markdown underscores like ``\_\_\_\_`` (name / date / signature).
    Once the rater fills the form in, those long blank runs are replaced by
    actual text. We therefore treat the *absence* of the escaped-underscore
    blank markers as "filled in". This is intentionally simple and only used for
    a status badge — the real gate stays in ``scripts/ingest_annotations.py``.
    """
    if not md_text:
        return False
    # Blank template marks unfilled fields with escaped-underscore runs.
    return "\\_\\_\\_\\_" not in md_text


# ── Declaration sign/edit helpers ──────────────────────────────────────────
def _declaration_values(md_text: str) -> Dict[str, str]:
    """Extract currently-filled name/date/signature from a declaration form.

    Returns:
        Dict with keys ``name``, ``date``, ``signature`` ('' when not filled).
    """
    name_m = re.search(r"I,\s*\*\*(.*?)\*\*\s*,\s*agree", md_text, flags=re.DOTALL)
    date_m = re.search(
        r"^Date:\s*\*\*(.*?)\*\*\s*$", md_text, flags=re.MULTILINE
    )
    sig_m = re.search(
        r"^Signature:\s*\*\*(.*?)\*\*\s*$", md_text, flags=re.MULTILINE
    )
    return {
        "name": name_m.group(1).strip() if name_m and name_m.group(1) else "",
        "date": date_m.group(1).strip() if date_m and date_m.group(1) else "",
        "signature": sig_m.group(1).strip() if sig_m and sig_m.group(1) else "",
    }


def _set_line_field(md_text: str, marker: str, value: str) -> str:
    """Replace a ``marker: ...`` line with ``marker: **value**`` (in place).

    Only applied when ``value`` is non-blank; otherwise the line is untouched.
    """
    if not value.strip():
        return md_text
    pattern = re.compile(rf"^{re.escape(marker)}:\s*.*$", re.MULTILINE)
    return pattern.sub(
        lambda m: f"{marker}: **{value.strip()}**", md_text, count=1
    )


def _fill_declaration(
    md_text: str, *, name: str, date: str, signature: str
) -> str:
    """Return ``md_text`` with the given signed fields set in place.

    Only non-empty fields are written; empty fields leave any existing value
    (or the ``____`` placeholder) untouched, so an incomplete form stays
    visibly unsigned.
    """
    if name.strip():
        md_text = re.sub(
            r"I,\s*.*?,\s*agree",
            lambda m: f"I, **{name.strip()}**, agree",
            md_text,
            count=1,
            flags=re.DOTALL,
        )
    if date.strip():
        md_text = _set_line_field(md_text, "Date", date)
    if signature.strip():
        md_text = _set_line_field(md_text, "Signature", signature)
    return md_text


def _save_records(records: List[dict], experiment_id: str, rater: str, split: str) -> Path:
    """Write all records back to the per-rater annotation file (atomic-ish)."""
    path = _annotation_path(experiment_id, rater, split)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return path


def _update_record(records: List[dict], index: int, updates: Dict) -> None:
    """Apply rater decisions to one record in place (mutates ``records``)."""
    if not (0 <= index < len(records)):
        return
    rec = records[index]
    dim = _dimension_of(rec)

    # Label (constrained by the UI dropdown — never free text).
    label_key = DIMENSION_LABEL_KEY.get(dim)
    if label_key:
        ann = rec.setdefault("annotations", {})
        if updates.get("label"):
            ann[label_key] = updates["label"]

    # Flags.
    flags = rec.setdefault("flags", {})
    if "uncertainty" in updates:
        flags["rater_uncertainty"] = updates["uncertainty"]
    if "response_invalid" in updates:
        flags["response_invalid"] = bool(updates["response_invalid"])
    if "difficult_case" in updates:
        flags["difficult_case"] = bool(updates["difficult_case"])

    # ai_assistance_used is deliberately never touched (must stay false).

    # Optional rationale note (never a label substitute).
    md = rec.setdefault("metadata", {})
    if "notes" in updates:
        md["notes"] = updates["notes"]


# ─────────────────────────── UI helpers ─────────────────────────────────────
def _render_record(st, rec: dict) -> None:
    """Render the prompt + response for a single record (blinded view)."""
    pr = rec.get("prompt_record") or {}

    dim = _dimension_of(rec)
    prompt = pr.get("prompt_text") or ""
    response = pr.get("model_response") or ""

    st.markdown(f"**Dimension:** `{dim}`")
    st.markdown("**Prompt**")
    st.markdown(f"> {prompt}")

    # Consistency templates use a "pairs"/"responses" list.
    pairs = rec.get("pairs") or rec.get("responses") or []
    if pairs:
        st.markdown("**Response group**")
        for i, resp in enumerate(pairs, start=1):
            text = resp if isinstance(resp, str) else (resp or {}).get("response", "")
            st.markdown(f"*Response {i}:*")
            st.text(text)
    else:
        st.markdown("**Response**")
        st.text(response)


def _render_declaration(st, experiment_id: str, rater: str) -> None:
    """Show a signable declaration form for the selected rater.

    If the form is blank, the rater types their name/date/signature and clicks
    Save; the declaration file is rewritten in place. The authoritative
    validation of completed declarations stays in the ingestion pipeline.
    """
    st.markdown("#### Rater Declaration")
    md_text = _load_declaration(experiment_id, rater)
    if not md_text:
        st.info(f"No declaration form found for `{rater}`.")
        return

    signed = _declaration_signed(md_text)
    if signed:
        st.success(f"[SIGNED] Declaration for `{rater}` appears **signed**.")
    else:
        st.warning(
            f"[BLANK] Declaration for `{rater}` still looks **blank**. "
            "Please read, fill in and save the form below before annotating."
        )

    values = _declaration_values(md_text)
    with st.expander("Sign / edit declaration form", expanded=not signed):
        # Live preview of the file (updates only on Save).
        st.markdown(md_text)
        st.markdown("---")
        with st.form(f"declaration_form_{rater}", clear_on_submit=False):
            c1, c2, c3 = st.columns(3)
            with c1:
                name = st.text_input(
                    "Your name",
                    value=values["name"],
                    key=f"decl_name_{rater}",
                    placeholder="Full name",
                )
            with c2:
                # Calendar picker. The declaration file stores the date as a
                # plain markdown string "**YYYY-MM-DD**", so we convert between
                # a real datetime.date (widget value) and that string.
                saved_date = values["date"]
                try:
                    # Already signed prefill -> parse the stored string.
                    default_date = datetime.strptime(
                        saved_date, "%Y-%m-%d"
                    ).date()
                except ValueError:
                    # Blank form -> default to today.
                    default_date = date.today()
                selected_date = st.date_input(
                    "Date",
                    value=default_date,
                    key=f"decl_date_{rater}",
                    help="Pick the signing date on the calendar.",
                )
                # Keep the widget value in sync if the form was already signed.
                date_str = selected_date.strftime("%Y-%m-%d")
            with c3:
                signature = st.text_input(
                    "Signature",
                    value=values["signature"],
                    key=f"decl_sig_{rater}",
                    placeholder="Type your signature",
                )
            submitted = st.form_submit_button(
                "Save signed declaration", type="primary"
            )

        if submitted:
            # st.date_input always returns a real date, but keep the guard for
            # clarity and to keep name/signature honestly required too.
            missing = [
                label
                for label, val in (
                    ("name", name),
                    ("date", date_str),
                    ("signature", signature),
                )
                if not val.strip()
            ]
            if missing:
                st.error(
                    "Declaration **not saved** — missing: "
                    + ", ".join(missing)
                    + ". All three fields are required to sign."
                )
            else:
                new_text = _fill_declaration(
                    md_text, name=name, date=date_str, signature=signature
                )
                path = _declaration_path(experiment_id, rater)
                path.write_text(new_text, encoding="utf-8")
                st.success(f"Declaration saved to `{path.name}`.")
                st.rerun()


def _render_adjudicator_handoff_note(st, rater: str) -> None:
    """Explain to the ADJUDICATOR that their file is the disagreements-only
    handoff produced by ``make experiment-adjudicator-prep``.

    Without this note the "No annotation template found ... Run ``make
    experiment-annotate RATERS=ADJUDICATOR``" warning below would be misleading:
    the adjudicator file is NOT the full-size clone from ``experiment-annotate``,
    but the smaller, double-blind file rebuilt from A/B disagreements.
    """
    if rater != "ADJUDICATOR":
        return
    st.info(
        "**ADJUDICATOR:** your file is the disagreements-only handoff. It is "
        "built from `make experiment-adjudicator-prep` **after** RATER_A and "
        "RATER_B are finished — it contains only the records where A and B "
        "disagree (prompt/response only, no A/B labels). If you see "
        "`No annotation template found`, ask the researcher to run "
        "`make experiment-adjudicator-prep`, not `make experiment-annotate`."
    )


def render(available) -> None:
    """Render the Rater Annotation tab (top-level Streamlit entry)."""
    st.markdown("## Rater Annotation")
    st.markdown(
        "*Fill the sealed, blinded annotation templates from the browser instead "
        "of editing JSONL by hand. Decisions are saved to "
        "`experiment/annotations/` in the exact schema the pipeline expects.*"
    )
    st.caption(
        "**Role in the project:** the *data-collection workflow* that produces "
        "the human labels consumed by `Human Annotation` and, ultimately, "
        "`Research Questions` RQ1 (auto-vs-human agreement). No headline claim "
        "comes from this tab itself — it only records annotations."
    )

    # ── Select experiment / rater / split ────────────────────────────────────
    experiment_id = _latest_experiment_id()
    if not experiment_id:
        st.warning(
            "No sealed experiment found. Run `make experiment-annotate` first."
        )
        return

    c_role, c_split = st.columns(2)
    with c_role:
        rater = st.selectbox(
            "Who are you?",
            RATERS,
            key="rater_role",
        )
    with c_split:
        split = st.selectbox(
            "Split",
            SPLITS,
            key="rater_split",
        )

    _render_declaration(st, experiment_id, rater)

    # For ADJUDICATOR, clarify the file is the A/B-disagreements handoff
    # (not the full-size template from `make experiment-annotate`).
    _render_adjudicator_handoff_note(st, rater)

    records = _load_records(experiment_id, rater, split)
    if not records:
        st.warning(
            f"No annotation template found for `{rater}` / `{split}`. "
            f"Run `make experiment-annotate RATERS=\"{rater}\"` first."
        )
        return

    # ── Progress overview ─────────────────────────────────────────────────────
    filled = [i for i, r in enumerate(records) if _label_of(r)]
    st.markdown(
        f"**Progress:** {len(filled)}/{len(records)} filled "
        f"· {len(records) - len(filled)} remaining"
    )
    st.progress(len(filled) / len(records))

    # Index navigation (default: first unfilled record).
    default_idx = filled[-1] + 1 if filled and filled[-1] + 1 < len(records) else 0
    if default_idx >= len(records):
        default_idx = 0

    # Deferred navigation: buttons/save set `rater_idx_pending` instead of
    # mutating `rater_idx` AFTER its widget already exists (Streamlit forbids
    # that — StreamlitAPIException). We consume the pending value HERE, before
    # the widget is created, then clear it.
    pending = st.session_state.pop("rater_idx_pending", None)
    if pending is not None:
        st.session_state["rater_idx"] = int(
            min(max(int(pending), 0), len(records) - 1)
        )
    else:
        # Initialize once (only before the widget is created).
        st.session_state.setdefault("rater_idx", default_idx)
    # Keep within bounds after records may have changed.
    st.session_state["rater_idx"] = min(
        max(int(st.session_state["rater_idx"]), 0), len(records) - 1
    )
    idx = int(st.session_state["rater_idx"])

    nav_c = st.columns([1, 4, 1])
    with nav_c[0]:
        if st.button("◀ Prev", key="btn_prev"):
            st.session_state["rater_idx_pending"] = idx - 1
            st.rerun()
    with nav_c[1]:
        # No `value=` here: the widget reads `rater_idx` straight from
        # session_state (initialized above). Passing both would make Streamlit
        # warn about a widget whose value is also set via the Session State API.
        st.number_input(
            "Record index (0-based)",
            min_value=0,
            max_value=len(records) - 1,
            key="rater_idx",
        )
    with nav_c[2]:
        if st.button("Next ▶", key="btn_next"):
            st.session_state["rater_idx_pending"] = idx + 1
            st.rerun()

    rec = records[idx]
    ik = rec.get("internal_key") or f"#{idx}"
    st.markdown(f"### Record `{ik}`")

    _render_record(st, rec)

    # ── Inputs: label (constrained) + flags + notes ──────────────────────────
    dim = _dimension_of(rec)
    label_options = sorted(VALID_LABELS.get(dim, set()))
    current_label = _label_of(rec)
    current_unc = (rec.get("flags") or {}).get("rater_uncertainty") or "confident"
    current_resp_invalid = bool((rec.get("flags") or {}).get("response_invalid"))
    current_difficult = bool((rec.get("flags") or {}).get("difficult_case"))
    current_notes = (rec.get("metadata") or {}).get("notes") or ""

    st.markdown("---")
    st.markdown("### Your judgment")

    c1, c2 = st.columns(2)
    with c1:
        label = st.radio(
            f"Label for `{dim}`",
            label_options,
            index=label_options.index(current_label) if current_label in label_options else 0,
            key=f"lbl_{idx}",
            format_func=lambda x: f"`{x}`",
        )
    with c2:
        uncertainty = st.radio(
            "Rater certainty",
            UNCERTAINTY_OPTIONS,
            index=UNCERTAINTY_OPTIONS.index(current_unc)
            if current_unc in UNCERTAINTY_OPTIONS
            else 0,
            key=f"unc_{idx}",
            help="Choose `uncertain` if you cannot decide — never pick a label arbitrarily.",
        )

    f_cols = st.columns(2)
    with f_cols[0]:
        response_invalid = st.checkbox(
            "No valid model response (broken/empty output)",
            value=current_resp_invalid,
            key=f"ri_{idx}",
        )
    with f_cols[1]:
        difficult_case = st.checkbox(
            "Difficult case",
            value=current_difficult,
            key=f"diff_{idx}",
        )

    notes = st.text_area(
        "Notes (optional rationale — not a label)",
        value=current_notes,
        key=f"notes_{idx}",
        height=100,
    )

    # ── Save button ───────────────────────────────────────────────────────────
    st.markdown("---")
    if st.button("Save this record", type="primary", key=f"save_{idx}"):
        _update_record(
            records,
            idx,
            {
                "label": label,
                "uncertainty": uncertainty,
                "response_invalid": response_invalid,
                "difficult_case": difficult_case,
                "notes": notes,
            },
        )
        saved_path = _save_records(records, experiment_id, rater, split)
        # Auto-advance to the next unfilled record after saving. Defer through
        # `rater_idx_pending` — the `rater_idx` widget already exists in this
        # run, so we cannot touch `rater_idx` directly (StreamlitAPIException).
        next_unfilled = next(
            (i for i in range(len(records)) if not _label_of(records[i])),
            idx,
        )
        st.session_state["rater_idx_pending"] = next_unfilled
        st.success(f"Saved to `{saved_path.name}` — {next_unfilled + 1} is next (or done).")
        st.rerun()

    # ── Per-file validation summary (read-only) ─────────────────────────────
    st.markdown("---")
    st.markdown("#### File status (for the researcher)")
    st.caption(
        "Shows only label validity — the pipeline (`make experiment-ingest`) "
        "performs the full schema check."
    )
    summary = []
    for i, r in enumerate(records):
        summary.append(
            {
                "idx": i,
                "internal_key": r.get("internal_key", ""),
                "dimension": _dimension_of(r),
                "label": _label_of(r) or "—",
            }
        )
    st.dataframe(pd.DataFrame(summary), width="stretch", hide_index=True)
