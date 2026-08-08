# ✅ Final Deliverables Index

Every requested deliverable and where to find it.

| Deliverable | Location |
|---|---|
| Production-quality Jupyter notebook | `notebooks/01_data_loading_and_inspection.ipynb` |
| Modular Python package | `src/` (preprocessing · feature_engineering · training · evaluation · inference · visualization · config · utils) |
| `requirements.txt` / `requirements-dev.txt` | repo root |
| `README.md` | repo root |
| `.gitignore` · `LICENSE` · `pyproject.toml` · `Makefile` | repo root |
| Trained model | `models/credit_scoring_logreg_v1.joblib` (+ `.pkl`) |
| Saved preprocessing pipeline | bundled inside the `.joblib` (end-to-end artifact) |
| Evaluation report | `reports/evaluation_report.md` (+ `evaluation_metrics.csv`) |
| Visualizations | `reports/figures/` (EDA, ROC/PR, confusion, calibration, lift/gain, threshold-cost) |
| Feature importance report | `reports/feature_selection_report.md`, `reports/explainability_report.md` |
| Confusion matrix | `reports/figures/10_confusion_*.png` |
| ROC curve | `reports/figures/10_roc_curves.png` |
| SHAP analysis | `reports/figures/12_shap_summary.png` (+ PDP, LIME, coefficients) |
| Business report | `reports/model_finalization_report.md`, `MODEL_CARD.md` |
| Technical report | `reports/` (DQ, EDA, split, model comparison, tuning, explainability) |
| Interview questions | `PORTFOLIO.md` |
| Resume bullet points | `PORTFOLIO.md` |
| LinkedIn project description | `PORTFOLIO.md` |
| Project planning | `PROJECT_PLANNING.md` |
| Architecture diagram | `architecture_diagram.svg` |
| Model card | `MODEL_CARD.md` |
| Commit strategy | `COMMITS.md` (+ `scripts/seed_git.sh`) |

## Key numbers at a glance
- **Final model:** tuned Logistic Regression → **Test ROC-AUC 0.944 · KS 0.748 · cost 0.215/applicant**
- **Approve/reject** at a cost-optimal threshold (~68% approval, ~2% approved bad rate)
- **10 algorithms compared**, **26 features selected** (from 45), **31 tests passing**
