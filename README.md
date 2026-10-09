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

- `data/` — the datasets (105 curated prompts + original seeds)
- `src/` — scoring logic (safety, truthfulness, consistency, trustscore)
- `scripts/` — pipeline scripts: evaluation, analysis, validation, experiment automation
- `app/` — interactive Streamlit dashboard
- `experiment/` — sealed multi-rater re-annotation experiment (templates, annotations, reports)
- `results/` — generated outputs: scores, confidence intervals, validation report
- `tests/` — automated tests
- `Makefile`, `demo.sh` — automation entry points

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
