# 🌿 Git Commit Strategy (Clean History)

A clean, atomic, **conventional-commit** history is what makes a repo look
professional. Commits should be **small, logical, and self-contained** — one
intent per commit — on feature branches merged via PR.

## Workflow
1. `main` is always green (tests pass).
2. Work on `feat/<topic>` branches; open PRs; squash-merge.
3. Use **Conventional Commits**: `type(scope): subject`
   - `feat` · `fix` · `docs` · `test` · `refactor` · `chore` · `perf`
4. Write the **imperative** subject ("add feature", not "added").
5. Reference the issue/phase in the body when relevant.

## Recommended commit sequence (mirrors the 17 phases)

```
chore: scaffold project structure, config, requirements, gitignore      # Phase 1-2
feat(data): synthetic credit dataset generator + data quality report    # Phase 3
feat(eda): distribution, correlation, outlier, skew analysis            # Phase 4
feat(preprocessing): leakage-safe cleaning + sklearn pipeline           # Phase 5
feat(features): domain feature engineering (DTI, risk index, FHI)       # Phase 6
feat(features): consensus feature selection (MI/RFE/permutation)        # Phase 7
feat(training): stratified split + cross-validation protocol            # Phase 8
feat(training): 10-model comparison harness with leakage-safe CV        # Phase 9
feat(evaluation): full metric suite + ROC/PR/calibration/lift/gain      # Phase 10
feat(training): randomized/grid hyperparameter search + early stopping  # Phase 11
feat(evaluation): SHAP/LIME/PDP explainability + reason codes           # Phase 12
feat(inference): joblib serialization, versioning, prediction API       # Phase 13
feat(inference): validated scoring service (schema, exceptions, logs)   # Phase 14
test: pytest suite (unit, contract, edge, performance)                  # Phase 15
docs: README, model card, architecture diagram, portfolio kit           # Phase 16-17
```

## Reproduce this history
Run `scripts/seed_git.sh` to initialize a repo and replay a clean, logically
grouped commit history (scaffold → data → modeling → evaluation → deployment →
tests → docs).

## .gitignore discipline
- **Never commit** raw datasets, model binaries, logs, or notebooks with outputs.
- Keep folder skeletons via `.gitkeep`.
- We strip notebook outputs with `nbstripout` before committing.
