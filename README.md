# Trustworthiness Evaluation

A lightweight, reproducible validation study of trustworthiness evaluation for open-source LLMs.

We evaluate **two open-source models** (Gemma 3 4B, Llama 3.1 8B) on **105 prompts** across three trustworthiness dimensions:

| Dimension                        | What it measures                                                                                               |
| -------------------------------- | -------------------------------------------------------------------------------------------------------------- |
| **Safety / Refusal**             | Resistance to prompt injection, role hijacking, instruction override, system-prompt leakage                    |
| **Truthfulness / Hallucination** | Ability to express uncertainty instead of fabricating information (False Premise Rejection + Factual Accuracy) |
| **Consistency / Robustness**     | Stability of responses under perturbations and repeated queries                                                |

The study also treats **evaluation itself as a measurement instrument**: it validates the auto-scorer against human labels (Cohen's κ) and reports when a small, local evaluation can or cannot be trusted.

## What's inside

- `data/` — the curated datasets (105 prompts across the three dimensions)
- `src/` — scoring logic (safety, truthfulness, consistency, trustscore)
- `scripts/` — pipeline scripts: evaluation, analysis, validation, experiment automation
- `app/` — interactive Streamlit dashboard
- `experiment/` — sealed multi-rater re-annotation experiment (filled annotations, agreements, gold labels, κ report)
- `results/` — generated outputs: scores, confidence intervals, ranking stability, cost/budget, figures
- `tests/` — automated tests
- `Makefile`, `demo.sh` — automation entry points

## Latest validated results

Experiment `trustworthiness-validation-2026-10-16-042935` (sealed, double-blind,
3 raters: A = Martin, B = Robert, adjudicator = Iqbal). All numbers are
reproduced by `make experiment-reproduce` and re-verified by
`make experiment-validate-rerun`.

| Model       | TrustScore | safety | truthfulness | consistency |
| ----------- | ---------- | ------ | ------------ | ----------- |
| gemma3:4b   | **0.755**  | 0.686  | 0.789        | 0.818       |
| llama3.1:8b | **0.831**  | 0.771  | 0.842        | 0.909       |

- **Gold vs auto (held-out, n=51):** agreement 0.82, Cohen's κ **0.73** [0.59, 0.87].
- **Inter-rater (A vs B):** consistency κ=1.00, safety κ=0.85, truthfulness ag=0.96.
- **Ranking:** llama3.1:8b beats gemma3:4b in **93.4%** of 10 000 bootstraps
  (per-unit trial bootstrap, which carries the dataset size into the
  uncertainty) — stable across all five weight configurations.
- **Budget plan:** safety & truthfulness are "unverified" bands → 37 human
  annotations (~$2.51 @ $20/h); consistency's auto score is trusted (κ=1.0).
- **Human-timing caveat:** the cost figure above uses the measured human-label
  time of **12.2 s** from `results/human_timing_measurement.json`. This timing is
  from a _small_ interactive study (**n = 8** records, one dimension — safety),
  with **high variance** (std 73.9 s) driven by one **outlier** (224.7 s). It is
  **extrapolated to all three dimensions**, so the budget is an indicative
  estimate, not a precisely measured per-label cost.
- Full details: `FINAL_REPORT.md` (internal), `experiment/reports/agreement_report.json`.

## Dashboard tabs → research questions

The project is organised around four research questions
(computed in `src/validation.py` / `scripts/paradigm_report.py`):

| RQ  | Question                                                        |
| --- | --------------------------------------------------------------- |
| RQ1 | How well do auto-labels agree with human labels?                |
| RQ2 | Which factors affect agreement? (dimension, attack type, model) |
| RQ3 | What dataset size is needed for stable scores?                  |
| RQ4 | What does automatic vs manual evaluation cost?                  |

Each Streamlit tab is explicitly mapped to a project role / RQ:

| Dashboard tab          | Role in the project                                                             | Research question              |
| ---------------------- | ------------------------------------------------------------------------------- | ------------------------------ |
| Overview               | Headline results (TrustScore, ranking, significance)                            | Results — points to RQ tabs    |
| Dimensions             | Measurement output per dimension + bootstrap CI + paired/clustered significance | Feeds RQ1–RQ4                  |
| Confusion Matrix       | Safety results (refusal / over-refusal)                                         | Material for RQ1, RQ2          |
| Research Questions     | Summary answer to all four RQs (κ, factors, stability, cost)                    | RQ1, RQ3, RQ4 (+ RQ2 overview) |
| What Affects Agreement | Agreement factors: dimension / attack type / model                              | RQ2                            |
| Human Annotation       | Blinded multi-rater + gold-vs-auto κ (measuring instrument)                     | RQ1                            |
| Rater Annotation       | Data-collection workflow (records human labels)                                 | Feeds RQ1                      |

The role of each tab is also shown inline in the dashboard (small caption under each tab heading).

## Usage

```bash
# Setup
make setup

# Run the full evaluation pipeline
make run

# View results
cat results/analysis_summary.txt

# Launch the interactive dashboard
make dashboard
```

For the human-annotation experiment (blinded protocol), the canonical
flow is:

```bash
make experiment-seal                    # seal templates + encrypted auto labels
make experiment-annotate RATERS="raterA raterB"
make experiment-adjudicator-prep         # build disagreements-only ADJUDICATOR file
# (raters fill templates in the browser via the Rater Annotation tab)
make experiment-ingest ANNOTATIONS="..." # schema-validate filled annotations
make experiment-resolve SPLIT=heldout    # gold from A/B + adjudicator
make experiment-gold SEAL_PASSPHRASE=... # build gold_labels.jsonl
make experiment-agreement                # κ + CI (gold vs auto, inter-rater)
make experiment-reproduce                # regenerate every number + verify
make experiment-validate-rerun           # clean-rerun validation gate
```

See the `make experiment-*` targets in the Makefile for details.
