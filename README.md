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
- `experiment/` — sealed multi-rater re-annotation experiment (templates, annotations, reports)
- `results/` — generated outputs: scores, confidence intervals, validation report
- `tests/` — automated tests
- `Makefile`, `demo.sh` — automation entry points

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

For the human-annotation experiment (Part 1), see the `make experiment-*` targets in the Makefile.
