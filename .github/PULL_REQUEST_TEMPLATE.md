## What does this PR do?

<!-- Brief, focused description of the change. Link the issue it resolves. -->

Closes #<!-- issue number -->

## Type of change

<!-- Check all that apply -->

- [ ] Bug fix
- [ ] New feature
- [ ] Documentation update
- [ ] Refactor / code hygiene
- [ ] Dependency update
- [ ] Reproducibility / results change

## Testing & verification

<!-- Replace the placeholders with actual output. Do not mark unchecked items as done. -->

- [ ] `make test` passes (full pytest suite)
- [ ] `make experiment-validate-rerun` passes (clean re-run, no secrets needed)
- [ ] `make verify-artifacts` passes (no immutable-artifact drift)
- [ ] No generated artifacts (`results/*`, figures, manifests) changed unexpectedly
- [ ] If numbers/budgets changed: updated artifacts in the same PR and explained **why** below

## Did any reported number change?

<!-- REQUIRED if this PR touches scoring, cost, or budgets. Reproducibility is
     a first-class requirement: any numeric change must be justified. -->

- [ ] No — numbers are unchanged
- [ ] Yes — see "Number changes" below

### Number changes (if any)

| Metric / artifact | Before | After | Reason |
| ----------------- | ------ | ----- | ------ |
| <!-- e.g. ranking stability → 93.4% --> | | | |

## Measurement honesty

- [ ] Inputs are explicitly labelled `MEASURED` / `ASSUMED` / `MODEL` as appropriate
- [ ] Caveats (e.g. small-sample timing, apples-to-oranges cost) are surfaced, not hidden

## Checklist before merge

- [ ] At least one approving review
- [ ] Branch is up to date with the target base branch
- [ ] Commit messages follow the project conventions (imperative, scoped)
- [ ] No secrets / passphrase / blinded ground-truth material in the PR

---

## Additional context

<!-- Anything else the reviewer should know. -->
