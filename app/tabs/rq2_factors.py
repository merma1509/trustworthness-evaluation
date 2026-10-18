"""Tab: What Affects Agreement?

Answers the project's second research question (RQ2): *which factors influence
auto-vs-human agreement?* (dimension, attack type, model).

Data comes from ``results/audit/agreement_report.json`` (rendered by
``scripts/paradigm_report.py``), specifically its ``by_dimension``,
``by_attack_type`` and ``by_model`` blocks.  A per-attack-type accuracy
breakdown of raw outputs is kept as a secondary "where does each model fail"
view.
"""

from pathlib import Path

import pandas as pd
import streamlit as st

from app.config import DIMENSIONS, MODEL_NAMES
from app.data_loader import load_agreement_report, load_raw_outputs

RESULTS_DIR = Path("results")


def _fmt_kappa(kappa) -> str:
    """Format a kappa value for display (-- when absent)."""
    if kappa is None or kappa == "":
        return "—"
    try:
        return f"{float(kappa):.3f}"
    except (TypeError, ValueError):
        return f"{kappa}"


def _render_factor_table(st, heading: str, rows: list, note: str) -> None:
    """Render a small factor table with a one-line takeaway."""
    st.markdown(f"### {heading}")
    if not rows:
        st.info("No data available.")
        return
    df = pd.DataFrame(rows)
    st.dataframe(df, width="stretch", hide_index=True)
    if note:
        st.caption(note)


def render(available):
    st.markdown("## What Affects Agreement?")
    st.markdown(
        "*Which factors move auto-vs-human agreement up or down — the dimension, "
        "the attack type, or the model? Data from "
        "`results/audit/agreement_report.json`.*"
    )

    agr = load_agreement_report() or {}

    # ── Headline takeaway ──────────────────────────────────
    by_dim = agr.get("by_dimension", {})
    by_model = agr.get("by_model", {})
    by_atype = agr.get("by_attack_type", {})

    worst_dim = min(
        by_dim.items(), key=lambda kv: kv[1].get("cohens_kappa", 1)
    ) if by_dim else (None, {})
    worst_atype = min(
        by_atype.items(), key=lambda kv: kv[1].get("cohens_kappa", 1)
    ) if by_atype else (None, {})

    st.markdown("---")

    # ── 1. Factor: Dimension ───────────────────────────────
    dim_rows = []
    for dim in DIMENSIONS:
        info = by_dim.get(dim, {})
        if not info:
            continue
        a = info.get("agreement_rate", 0)
        k = info.get("cohens_kappa")
        n = info.get("n_valid_pairs", info.get("n", 0))
        dim_rows.append(
            {
                "Dimension": dim.capitalize(),
                "n (pairs)": n,
                "Agreement": f"{a:.0%}",
                "Cohen's kappa": _fmt_kappa(k),
            }
        )
    _render_factor_table(
        st,
        "1 · Dimension",
        dim_rows,
        "Safety and consistency tend to be clear-cut; truthfulness mixes "
        "false-premise rejection with factual accuracy, so agreement varies most there.",
    )

    # ── 2. Factor: Attack Type ─────────────────────────────
    atype_rows = []
    for atype, info in sorted(by_atype.items()):
        a = info.get("agreement_rate", 0)
        k = info.get("cohens_kappa")
        n = info.get("n_valid_pairs", info.get("n", 0))
        atype_rows.append(
            {
                "Attack Type": atype,
                "n (pairs)": n,
                "Agreement": f"{a:.0%}",
                "Cohen's kappa": _fmt_kappa(k),
            }
        )
    if worst_atype[0]:
        note = (
            f"Hardest to agree on: **{worst_atype[0]}** "
            f"(κ={_fmt_kappa(worst_atype[1].get('cohens_kappa'))}, "
            f"n={worst_atype[1].get('n_valid_pairs', worst_atype[1].get('n', 0))}). "
            "Low-n cells are indicative only — do not over-read single samples."
        )
    else:
        note = ""
    _render_factor_table(st, "2 · Attack Type", atype_rows, note)

    # ── 3. Factor: Model ───────────────────────────────────
    model_rows = []
    for mk, info in sorted(by_model.items()):
        a = info.get("agreement_rate", 0)
        k = info.get("cohens_kappa")
        n = info.get("n_valid_pairs", info.get("n", 0))
        model_rows.append(
            {
                "Model": MODEL_NAMES.get(mk, mk),
                "n (pairs)": n,
                "Agreement": f"{a:.0%}",
                "Cohen's kappa": _fmt_kappa(k),
            }
        )
    _render_factor_table(
        st,
        "3 · Model",
        model_rows,
        "Per-model agreement — a large gap would hint the auto-scorer is biased "
        "towards one family of outputs.",
    )

    if worst_dim[0] or worst_atype[0]:
        st.markdown("---")
        st.markdown("### Read-out")
        dim_summary = (
            f"worst dimension: **{worst_dim[0]}** (κ={_fmt_kappa(worst_dim[1].get('cohens_kappa'))})"
            if worst_dim[0]
            else "no per-dimension data"
        )
        atype_summary = (
            f"worst attack type: **{worst_atype[0]}** (κ={_fmt_kappa(worst_atype[1].get('cohens_kappa'))})"
            if worst_atype[0]
            else "no per-attack data"
        )
        st.info(
            "**Factor read-out:** the agreement drops when task ambiguity or "
            f"scorer blind-spots concentrate — here that is {dim_summary}, with "
            f"{atype_summary}. Audit those cells first."
        )

    # ── 4. Secondary: per-attack-type accuracy (raw) ───────
    st.markdown("---")
    st.markdown("### Where does each model fail? (per-attack accuracy)")
    st.caption(
        "Side view on the *raw outputs* (accuracy of the auto-scorer per attack "
        "type), complementary to the human-agreement factors above."
    )
    raw_data = {}
    for dim in DIMENSIONS:
        for mk in available:
            raw_data[(mk, dim)] = load_raw_outputs(mk, dim)

    for dim in DIMENSIONS:
        st.markdown(f"#### {dim.capitalize()}")
        rows = []
        for mk in available:
            records = raw_data.get((mk, dim), [])
            atypes = set(r.get("attack_type", "unknown") for r in records)
            for at in sorted(atypes):
                subset = [r for r in records if r.get("attack_type") == at]
                if not subset:
                    continue
                correct = sum(1 for r in subset if r.get("is_correct", False))
                total = len(subset)
                rows.append(
                    {
                        "Model": MODEL_NAMES.get(mk, mk),
                        "Attack Type": at,
                        "Correct": f"{correct}/{total}",
                        "Accuracy": correct / total if total else 0,
                        "Total": total,
                    }
                )
        if rows:
            df = pd.DataFrame(rows)
            st.dataframe(
                df.style.format({"Accuracy": "{:.0%}"}),
                width="stretch",
                hide_index=True,
            )
        else:
            st.info(f"No {dim} data available.")
        st.markdown("---")
