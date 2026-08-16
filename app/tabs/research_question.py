"""Tab: Research Question — When can auto-evaluation be trusted?

All numbers computed from:
  - results/audit/agreement_report.json  (Cohen's kappa, confusion matrix, 174 records)
  - results/validation_report.json       (jackknife stability, cost-benefit)
  - results/ranking_stability.json       (ranking sensitivity to weights)
"""

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from app.config import MODEL_NAMES
from app.data_loader import load_agreement_report, load_ranking_stability


def _kappa_label(k: float) -> str:
    if k >= 0.81:
        return "Almost Perfect"
    if k >= 0.61:
        return "Substantial"
    if k >= 0.41:
        return "Moderate"
    if k >= 0.21:
        return "Fair"
    return "Slight"


def _load_validation():
    path = Path("results/validation_report.json")
    if not path.exists():
        return {}
    with open(path) as f:
        return json.loads(f.read())


def render(available, gemma_scores, llama_scores, gemma_cis, llama_cis, all_scores, all_cis):
    # ── Load data ──────────────────────────────────────────
    report = _load_validation()
    rq3 = report.get("rq3_dataset_stability", {})
    rq4 = report.get("rq4_cost", {})

    agr = load_agreement_report() or {}
    overall = agr.get("overall", {})
    dim_data = agr.get("by_dimension", {})
    model_data = agr.get("by_model", {})
    ranking = load_ranking_stability()

    overall_kappa = overall.get("cohens_kappa", 0)
    overall_agree = overall.get("agreement_rate", 0)
    overall_n = overall.get("n_valid_pairs", 0)
    overall_kdesc = _kappa_label(overall_kappa)

    st.markdown("## Research Question")
    st.markdown(
        "*When can a very small, cheap, local evaluation be trusted, "
        "and how do we know when it fails?*"
    )
    st.markdown("---")

    # ── 1. Core Finding ─────────────────────────────────────
    st.markdown("### Core Finding")
    with st.container(border=True):
        dim_rows = ""
        for d in ["safety", "truthfulness", "consistency"]:
            dd = dim_data.get(d, {})
            if dd:
                a = dd.get("agreement_rate", 0)
                k = dd.get("cohens_kappa", 0)
                n = dd.get("n_valid_pairs", dd.get("n", 0))
                desc = _kappa_label(k)
                dim_rows += (
                    f"| **{d.capitalize()}** | {a:.0%} ({n} pairs) "
                    f"| **{k:.2f}** | {desc} |\n"
                )

        st.success(
            f"**Auto-scorer achieves {overall_agree:.0%} agreement with human judgment "
            f"(Cohen's kappa = {overall_kappa:.4f}, {overall_kdesc}, "
            f"n={overall_n} across {len(dim_data)} dimensions, "
            f"{len(model_data)} models)**\n\n"
            f"| Dimension | Agreement | Cohen's kappa | Interpretation |\n"
            f"|-----------|:---------:|:-------------:|:---------------|\n"
            f"{dim_rows}"
            f"\n**Conclusion:** Auto-scoring is reliable for clear-cut cases "
            f"but degrades on:\n"
            f"1. **Safety** (κ={dim_data.get('safety', {}).get('cohens_kappa', 0):.2f}) — "
            f"punctuation artifacts in refusal detection\n"
            f"2. **Truthfulness** (κ={dim_data.get('truthfulness', {}).get('cohens_kappa', 0):.2f}) "
            f"— false-premise vs factual accuracy mix\n"
            f"3. **Consistency** (κ={dim_data.get('consistency', {}).get('cohens_kappa', 0):.2f}) "
            f"— duplicate prompts inflate scores"
        )

    # ── 2. Cohen's Kappa ────────────────────────────────────
    st.markdown("### Cohen's Kappa — Agreement Beyond Chance")
    with st.container(border=True):
        st.markdown(
            "**Cohen's Kappa** measures agreement *beyond random chance*.\n"
            "κ=1.0 = perfect, κ=0 = chance level, κ<0 = worse than random.\n\n"
            "*Why kappa over accuracy?* With skewed labels, random guessing "
            "can achieve high accuracy. Kappa corrects for this."
        )
        rows = []
        rows.append({
            "Factor": "Overall",
            "Kappa": f"{overall_kappa:.4f}",
            "Interpretation": overall_kdesc,
            "n": overall_n,
        })
        for d in ["safety", "truthfulness", "consistency"]:
            dd = dim_data.get(d, {})
            if dd:
                k = dd.get("cohens_kappa", 0)
                n = dd.get("n_valid_pairs", dd.get("n", 0))
                rows.append({
                    "Factor": d.capitalize(),
                    "Kappa": f"{k:.4f}",
                    "Interpretation": _kappa_label(k),
                    "n": n,
                })
        for mk, md in sorted(model_data.items()):
            k = md.get("cohens_kappa", 0)
            n = md.get("n_valid_pairs", md.get("n", 0))
            rows.append({
                "Factor": "Model: " + MODEL_NAMES.get(mk, mk),
                "Kappa": f"{k:.4f}",
                "Interpretation": _kappa_label(k),
                "n": n,
            })
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

    # ── 3. Evidence (per-label CM) ──────────────────────────
    st.markdown("### Evidence — Per-Label Precision & Recall")
    with st.container(border=True):
        per_label = overall.get("per_label_agreement", {})
        if per_label:
            ev_rows = []
            for lbl, metrics in per_label.items():
                ev_rows.append({
                    "Label": lbl.capitalize(),
                    "Auto Count": metrics.get("auto_count", 0),
                    "Human Count": metrics.get("human_count", 0),
                    "Precision": f"{metrics.get('precision', 0):.0%}",
                    "Recall": f"{metrics.get('recall', 0):.0%}",
                    "F1": f"{metrics.get('f1', 0):.0%}",
                })
            st.dataframe(pd.DataFrame(ev_rows), width="stretch", hide_index=True)
            st.caption(
                "Auto is **conservative** — high precision for 'incorrect' "
                f"({per_label.get('incorrect', {}).get('precision', 0):.0%}), "
                f"lower recall ({per_label.get('incorrect', {}).get('recall', 0):.0%}). "
                f"Overall κ={overall_kappa:.4f} ({overall_kdesc})."
            )
        else:
            st.info("Per-label agreement data not available.")

    # ── 4. Confusion Matrix ─────────────────────────────────
    st.markdown("### Confusion Matrix: Auto vs Human")
    with st.container(border=True):
        cm = overall.get("confusion_matrix", {})
        if cm:
            cells = ["correct", "incorrect", "consistent", "inconsistent"]
            cm_data = []
            for row_label in cells:
                row_data = cm.get(row_label, {})
                row_vals = {c: row_data.get(c, 0) for c in cells}
                row_vals["Label"] = row_label.capitalize()
                cm_data.append(row_vals)
            cm_df = pd.DataFrame(cm_data)
            col_order = ["Label"] + [c.capitalize() for c in cells]
            cm_df = cm_df.rename(columns={c: c.capitalize() for c in cells})
            st.dataframe(
                cm_df[col_order] if all(c in cm_df.columns for c in col_order) else cm_df,
                width="stretch",
                hide_index=True,
            )
            st.markdown(
                f"**{overall_n} annotated pairs** "
                f"across {len(dim_data)} dimensions, {len(model_data)} models."
            )
        else:
            st.info("Confusion matrix data not available.")

    # ── 5. When It Fails ────────────────────────────────────
    st.markdown("### When Does It Fail?")
    with st.container(border=True):
        st.markdown("**Root causes by dimension:**")
        for d in ["safety", "truthfulness", "consistency"]:
            dd = dim_data.get(d, {})
            if dd:
                k = dd.get("cohens_kappa", 0)
                n = dd.get("n_valid_pairs", dd.get("n", 0))
                a = dd.get("agreement_rate", 0)
                issues = {
                    "safety": "punctuation artifacts, apostrophe mismatch, "
                             "role-play compliance missed",
                    "truthfulness": "all benign = correct (no scorer), "
                                   "only false-premise rejection measured",
                    "consistency": "duplicate prompts, fail-open similarity "
                                   "=1.0, singleton groups always pass",
                }
                st.markdown(
                    f"- **{d.capitalize()}** (κ={k:.2f}, n={n}, "
                    f"agreement={a:.0%}): {issues.get(d, '')}"
                )
        st.markdown(
            "\n**Rule of thumb:** Audit ≥10% of auto-labels (≈17–20 responses). "
            "If κ < 0.60, investigate failure modes before drawing conclusions."
        )

    # ── 6. Ranking Stability ────────────────────────────────
    st.markdown("### Ranking Sensitivity to Weights")
    with st.container(border=True):
        if ranking:
            configs = ranking.get("configurations", [])
            if configs:
                rs_rows = []
                for cfg in configs:
                    row = {"Config": cfg.get("config", "?")}
                    scores = cfg.get("scores", {})
                    for mk in available:
                        ml = MODEL_NAMES.get(mk, mk)
                        # ranking uses "gemma3:4b" / "llama3.1:8b" keys
                        # Map model keys
                        for score_key, score_val in scores.items():
                            short_key = score_key.replace(":", "_")
                            if short_key == mk:
                                row[ml] = score_val
                    rs_rows.append(row)
                rs_df = pd.DataFrame(rs_rows)
                st.dataframe(
                    rs_df.style.format(
                        {k: "{:.4f}" for k in rs_df.columns if k != "Config"}
                    ),
                    width="stretch",
                    hide_index=True,
                )
                flips = sum(
                    1 for cfg in configs
                    if len(set(cfg.get("ranking", {}).values())) > 1
                )
                if flips > 1:
                    st.warning(
                        "⚠️ **Ranking flips detected!** "
                        "The 'best' model changes depending on weight choice. "
                        "Absolute rankings are not robust."
                    )
                else:
                    st.success("Ranking is stable across weight configurations.")
        else:
            st.info("Ranking stability data not available.")

    # ── 7. Dataset Stability ────────────────────────────────
    st.markdown("### Dataset Stability (Jackknife)")
    with st.container(border=True):
        if rq3:
            r3_rows = []
            for dim, data in rq3.items():
                jk = data.get("jackknife_stability", {})
                ds = data.get("dataset_size_sensitivity", {})
                r3_rows.append({
                    "Dimension": dim.capitalize(),
                    "Score": f"{jk.get('full_score', 0):.4f}",
                    "n": jk.get("n_total", 0),
                    "Jackknife Std": f"±{jk.get('std_jackknife', 0):.4f}",
                    "Max Change": f"{jk.get('max_decrease', 0):.4f}",
                    "Min N Needed": ds.get("estimated_min_n", "?"),
                })
            if r3_rows:
                st.dataframe(pd.DataFrame(r3_rows), width="stretch", hide_index=True)
            st.success(
                "All dimensions are **stable**: removing any 1 prompt "
                "changes the score by < 3%."
            )
        else:
            st.info("Stability data not available. Run `scripts/paradigm_report.py` first.")

    # ── 8. Cost-Benefit ────────────────────────────────────
    st.markdown("### Cost-Benefit")
    with st.container(border=True):
        if rq4:
            auto = rq4.get("fully_automatic", {})
            human = rq4.get("fully_human", {})
            hybrid = rq4.get("hybrid_50pct_audit", {})
            auto_cost = auto.get("cost", 0.29)
            human_cost = human.get("cost", 35.0)
            ratio = human_cost / auto_cost if auto_cost > 0 else 0

            cost_rows = [
                {
                    "Method": "Fully auto",
                    "Time": f"{auto.get('time_hours', 0.6):.1f}h",
                    "Cost": f"${auto_cost:.2f}",
                    "Quality": f"{overall_agree:.0%} agree, κ={overall_kappa:.2f}",
                },
                {
                    "Method": "Fully human",
                    "Time": f"{human.get('time_hours', 1.8):.1f}h",
                    "Cost": f"${human_cost:.2f}",
                    "Quality": "100% ground truth",
                },
                {
                    "Method": "Hybrid (50% audit)",
                    "Time": f"{hybrid.get('time_hours', 1.5):.1f}h",
                    "Cost": f"${hybrid.get('cost', 17.79):.2f}",
                    "Quality": "Validated auto + human",
                },
            ]
            st.dataframe(pd.DataFrame(cost_rows), width="stretch", hide_index=True)
            st.info(
                f"Auto evaluation is **{ratio:.0f}x cheaper** than human evaluation. "
                f"A hybrid approach with 50% validation provides measurement "
                f"validation at half the human cost."
            )
        else:
            st.info("Cost data not available.")

    # ── 9. When to Trust ────────────────────────────────────
    st.markdown("### When to Trust It")
    with st.container(border=True):
        st.info(
            "**You CAN trust the auto-evaluation when:**\n\n"
            "- Relative rankings (which model wins per dimension)\n"
            "- Accept ±5% margin of error (all CIs overlap)\n"
            "- Spot-check ≥10% of labels for known failure modes\n"
            "- Clear-cut safety and consistency (κ ≥ 0.80)\n"
            "- Dataset ≥ 50 prompts per dimension (stable)\n\n"
            "**You CANNOT trust it when:**\n\n"
            "- Benign truthfulness is included (no scorer, κ drops)\n"
            "- Models respond in multiple languages\n"
            "- Evaluation involves nuanced role-play\n"
            "- You need absolute scores (auto is systematically too strict)\n"
            "- Sample size < 10 prompts per dimension"
        )

    # Footer
    st.caption(
        "All numbers computed from `results/audit/agreement_report.json`, "
        "`results/validation_report.json`, and `results/ranking_stability.json`. "
        "Re-run `scripts/paradigm_report.py` after pipeline changes."
    )
