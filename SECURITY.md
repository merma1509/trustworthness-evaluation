# Security Policy

This project evaluates trustworthiness of open-source LLMs. It processes and
stores **human-annotation data** (blinded rater templates, gold labels,
sealed/encrypted auto labels) and runs a small local inference pipeline.
We take the confidentiality of annotation data and the integrity of the
evaluation seriously.

---

## Supported versions

| Version | Branch          | Supported          |
| ------- | --------------- | ------------------ |
| Latest  | `main`          | :white_check_mark: |
| dev     | `develop`       | :white_check_mark: (active development) |
| staging | `staging`       | :white_check_mark: (QA only)            |
| Older   | other branches  | :x:                 |

Security fixes are backported to the latest released (tagged) version on
`main`. `develop` always carries the newest fixes.

---

## Reporting a vulnerability

**Please do not open a public issue for a security vulnerability.**

Report it privately instead:

- **Email:** open a GitHub **private security advisory** via
  *Security → Report a vulnerability*, or
- **Ping a maintainer** privately on the development channel.

Please include:

1. A short description of the issue and its impact.
2. The affected file(s) / function(s) / version.
3. A minimal reproduction (steps or a small snippet).
4. Any suggested fix, if you have one.

We aim to acknowledge reports within **3 business days** and to respond with a
plan (fix / mitigations) within **10 business days**.

---

## What to report

- Leakage or exposure of **sealed/encrypted annotation data**, plaintext
  auto labels (`experiment/sealed/labels/auto_labels.jsonl`, `.enc` files),
  rater templates, or the seal passphrase.
- Integrity issues: tampering with gold labels, agreement reports, sealed
  manifests, or any **immutable reproducibility artifact**.
- Blind-protocol violations: anything that could reveal
  `experiment/blinded/ground_truth_blinded.json` or unblind a rater.
- Prompt-injection / data-exfiltration risks in the evaluation pipeline.
- Supply-chain issues (dependencies) affecting the evaluation.
- Any unexpected data destruction or unauthorized modification of results.

---

## Data handling rules (reporters & contributors)

- The seal passphrase must **never** be committed. It lives only in local
  keeper state / environment (see `.gitignore`).
- Blinded ground truth is researcher-only and must never reach annotators or
  the committed annotation files.
- Plaintext committed auto labels are a deliberate, single, tracked
  exception (for a secret-free clean re-run) — keep them consistent with the
  encrypted sealed copies.
- When reporting, do **not** paste raw annotation records or passphrase
  material into public channels.

---

## Security best practices for contributors

- Use a private GitHub repository (or protected branches) for this
  experiment; `main`, `staging`, and `develop` should be protected.
- Require pull-request reviews before merging changes that touch sealed
  data, scoring, or reproducibility artifacts.
- Do not add secrets to `pyproject.toml`, `.env`, logs, or commit messages.
- Verify the seal/blinding integrity gates before and after your changes:
  `make experiment-validate-rerun` includes seal integrity + blinding checks.

---

## Security-relevant checks

The pipeline itself ships with integrity gates that double as security
controls:

```bash
make verify-artifacts              # verify_immutable — no artifact drift
make experiment-validate-rerun     # seal integrity + blinding + clean re-run
```

If any of these fail, treat it as a potential integrity incident and report it.
