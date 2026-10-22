# Contributing to Trustworthiness Evaluation

Thanks for your interest in contributing! This project is a reproducible, held-out validation study of trustworthiness evaluation for open-source LLMs. Please follow the workflow below so every change stays reviewable, reproducible, and safe to ship.

---

## 1. Branching model (GitFlow-lite)

The repository uses three long-lived branches with a clear promotion path:

```bash
main      (protected) — production. Only receives merged,
                        tested releases via pull request from staging.
staging   (protected) — testing / QA. Integration branch where changes
                        from develop are published and validated before
                        they reach main.
develop   (protected) — development / draft. Default base branch for
                        feature and fix branches.
```

- **Feature / fix branches** branch off `develop` and are merged back via a
  pull request.
- **Hotfixes** for issues in production branch off `main`, are reviewed
  against `staging` (or `main` for truly critical fixes), and are then
  merged back into both `main` and `develop`.

Branch naming convention:

| Type    | Prefix     | Example                        |
| ------- | ---------- | ------------------------------ |
| Feature | `feature/` | `feature/add-rq5-cost-tab`     |
| Fix     | `fix/`     | `fix/truthfulness-scorer-rtti` |
| Docs    | `docs/`    | `docs/contributing-guidelines` |
| Chore   | `chore/`   | `chore/update-lockfile`        |
| Hotfix  | `hotfix/`  | `hotfix/seal-manifest-path`    |

---

## 2. Development setup

```bash
make setup          # uv sync --extra dev (installs everything from pyproject.toml + uv.lock)
make test           # run the full pytest suite (198 tests)
make lint           # if applicable
```

We use **`uv`** (see `pyproject.toml` + `uv.lock`). Do **not** add a
`requirements.txt`. If you add a dependency, use:

```bash
uv add <package>            # runtime dependency
uv add --dev <package>      # dev-only dependency
```

---

## 3. Making a change (step by step)

1. **Branch** from the latest `develop`:

   ```bash
   git checkout develop
   git pull
   git checkout -b feature/your-short-description
   ```

2. **Make focused commits** with clear, imperative messages.
   See [Commit conventions](#4-commit-conventions).

3. **Run the checks locally** before pushing:

   ```bash
   make test                          # whole test suite passes
   python3 scripts/verify_immutable.py --manifest results/manifest.json --base-dir .
   ```

4. **Push and open a pull request** targeting **`develop`**:

   ```bash
   git push -u origin feature/your-short-description
   ```

   Use the repository's pull request template
   (`.github/PULL_REQUEST_TEMPLATE.md`). Keep PRs small and focused; a PR that
   is easy to review is easy to merge.

5. **Pass CI / checks.** The PR must be reviewable and green before merge:
   - all tests pass,
   - no generated artifacts drift (`verify_immutable` passes),
   - at least one approving review (more for core changes).

---

## 4. Commit conventions

- Use the imperative mood: `Fix ...`, `Add ...`, `Update ...`.
- Prefix commits by scope when useful: `fix(script): ...`, `docs: ...`,
  `chore: ...`.
- Keep each commit logically self-contained; don't bundle unrelated changes.
- Prefer small commits: they make review and `git bisect` much easier.

Examples:

```text
fix(scoring): misclassify reject-then-invent as fabrication
docs: clarify human-timing caveat in README
chore: prune unused Makefile targets
```

---

## 5. Reproducibility & verification (non-negotiable)

This project treats **evaluation reproducibility** as a first-class
requirement. Before a change that touches scoring, results, or budgets is
approved, it must hold:

- `make test` — full suite (198 tests) passes.
- `make experiment-validate-rerun` — clean re-run from immutable inputs
  regenerates every reported number (no secrets needed).
- `make verify-artifacts` — `verify_immutable` reports **no modifications**.

If your change alters any reported number, **update the affected artifacts**
(`results/*.json`, figures, manifests) in the same PR, and explain in the PR
description **why** the number changed.

Products must never over-claim measurement:

- mark `MEASURED` / `ASSUMED` / `MODEL` inputs honestly,
- surface caveats (e.g. the small single-dimension human-timing study),
- never present a model estimate as a measured result.

---

## 6. Merge / release flow

1. Merge feature/fix PRs into **`develop`** (squash merge recommended).
2. Promote `develop` → **`staging`** with a pull request.
   Run the QA checklist on `staging` (tests + validate-rerun + verify-artifacts).
3. When a release is ready, open a PR **`staging` → `main`**.
   Only maintainers merge to `main`; `main` is protected.
4. Tag releases on `main` (e.g. `v2.0.0`).

---

## 7. Reporting bugs / requesting features

Open a GitHub issue with:

- a clear title,
- steps to reproduce (for bugs),
- expected vs actual behavior,
- environment (commit hash / branch, OS), if relevant.

---

## 8. Code of conduct

Be respectful, constructive, and precise. This is a research codebase with careful reproducibility commitments; review questions are expected and encouraged. Personal attacks are not tolerated.
