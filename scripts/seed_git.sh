#!/usr/bin/env bash
# ---------------------------------------------------------------
# Seed a clean, logically-grouped git history for the portfolio repo.
# Usage:  bash scripts/seed_git.sh
# ---------------------------------------------------------------
set -euo pipefail

cd "$(git rev-parse --show-toplevel 2>/dev/null || echo .)"

if [ ! -d .git ]; then
  git init -q
  git branch -M main
fi

# Configure identity if unset (placeholder — replace with yours).
git config user.email >/dev/null 2>&1 || git config user.email "you@example.com"
git config user.name  >/dev/null 2>&1 || git config user.name  "Your Name"

commit() {  # commit "<message>" <path1> <path2> ...
  local msg="$1"; shift
  git add -- "$@" 2>/dev/null || true
  git commit -q -m "$msg" || echo "skip (nothing to commit): $msg"
}

commit "chore: scaffold project structure, config, requirements, gitignore" \
  .gitignore LICENSE pyproject.toml Makefile requirements.txt requirements-dev.txt \
  config/ src/__init__.py src/config src/utils README.md

commit "feat(data): synthetic credit dataset generator + data quality report" \
  src/preprocessing/data_generator.py src/preprocessing/data_inspection.py \
  src/preprocessing/cleaner.py reports/data_quality_report.md

commit "feat(eda): distribution, correlation, outlier, skew analysis" \
  src/visualization reports/eda_report.md "reports/figures/0*.png"

commit "feat(preprocessing): leakage-safe cleaning + sklearn pipeline" \
  src/preprocessing/pipeline.py

commit "feat(features): domain feature engineering + consensus selection" \
  src/feature_engineering reports/feature_selection_report.md artifacts/selected_features.json

commit "feat(training): split, 10-model comparison, hyperparameter tuning" \
  src/training reports/model_comparison.md reports/tuning_report.md \
  reports/data_split_report.md artifacts/best_params.json

commit "feat(evaluation): metric suite + SHAP/LIME/PDP explainability" \
  src/evaluation reports/evaluation_report.md reports/explainability_report.md

commit "feat(inference): serialization, versioning, scoring service" \
  src/inference reports/model_finalization_report.md "models/*.meta.json"

commit "test: pytest suite (unit, contract, edge, performance)" \
  tests

commit "docs: README, model card, architecture, portfolio kit" \
  README.md MODEL_CARD.md PROJECT_PLANNING.md architecture_diagram.svg \
  PORTFOLIO.md COMMITS.md DELIVERABLES.md scripts/seed_git.sh

echo
echo "✅ Clean history created. Review with:  git log --oneline"
