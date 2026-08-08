# 🏦 Credit Scoring System — Production-Ready ML

> A modular, reproducible, fairness-aware machine-learning pipeline that predicts
> customer **creditworthiness** the way a real risk-analytics team would build it —
> engineered for interpretability, cost-optimal decisions, and auditability.

[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Code style: Ruff](https://img.shields.io/badge/code%20style-ruff-261230.svg)](https://docs.astral.sh/ruff/)
[![Tests](https://img.shields.io/badge/tests-31%20passing-brightgreen.svg)](#-testing)
[![Model](https://img.shields.io/badge/test%20AUC-0.944-success.svg)](#-results)

---

## 📑 Table of Contents
1. [Overview](#-overview)
2. [Business & ML Framing](#-business--ml-framing)
3. [Workflow / Architecture](#-workflow--architecture)
4. [Results](#-results)
5. [Installation Guide](#-installation-guide)
6. [Quickstart / Reproducibility](#-quickstart--reproducibility)
7. [Project Structure](#-project-structure)
8. [Dataset Description](#-dataset-description)
9. [Model Summary](#-model-summary)
10. [Explainability](#-explainability)
11. [Testing](#-testing)
12. [Limitations](#-limitations)
13. [Future Improvements](#-future-improvements)
14. [License](#-license)

---

## 🔭 Overview

This project builds an **end-to-end credit-scoring system** that classifies loan
applicants as **creditworthy (1)** or **defaulter (0)** and outputs a calibrated
probability, an approve/reject decision at a **cost-optimal threshold**, a **risk
tier**, and per-applicant **reason codes**. It is built with industry-grade
software engineering: a modular `src/` package, YAML-driven configuration,
leakage-safe pipelines, structured logging, a tested inference service, and
versioned artifacts.

**Final model:** tuned **Logistic Regression** (best ROC-AUC, lowest business cost,
fully interpretable, regulator-ready). Test-set **AUC = 0.944**, **KS = 0.748**.

---

## 🎯 Business & ML Framing

- **Problem:** Decide whether an applicant will repay. Wrong decisions are
  costly in both directions — approving a defaulter (loss given default) and
  rejecting a good customer (lost revenue).
- **ML goal:** Supervised **binary classification** predicting `P(creditworthy)`,
  optimized for **rank-ordering** (KS/AUC) under a **cost-asymmetric** objective.
- **Cost model:** approving a defaulter is **5×** costlier than rejecting a good
  customer → the decision threshold is tuned to **minimize business cost**, not
  accuracy.
- **Convention:** positive class = `creditworthy` (approve). Therefore
  *FP = approving a defaulter* (expensive) and *FN = rejecting a good customer*.
- **Fairness:** protected attributes (`gender`, `marital_status`) are **excluded
  from training** and used only for fairness auditing.

---

## 🧭 Workflow / Architecture

The system is a **leakage-safe, 12-stage pipeline** with cross-cutting concerns
(config, logging, tests, artifacts, reports):

```
RAW DATA → Cleaning → Feature Engineering → Preprocessing → Feature Selection
   → [Stratified Split: 64/16/20] → Model Comparison (10 algos) → Evaluation
   → Hyperparameter Tuning → Explainability (SHAP/LIME) → Serialize → Inference Service
   → Monitoring/Retraining (future)

Cross-cutting: config.yaml · structured logs · pytest · reports/figures · artifacts/
```

A full layered architecture diagram is in [`architecture_diagram.svg`](architecture_diagram.svg).

**Leakage-prevention protocol (enforced everywhere):**
1. Split first, before any fitting. 2. Preprocessing fit on **train only**.
3. Selection fit on **train only**. 4. Tuning via CV folds from train only.
5. Test set touched **once**, at finalization.

---

## 📊 Results

**Model comparison (validation, 10 algorithms, cost-optimal thresholds):**

| Model | ROC-AUC | KS | Cost/app |
|---|---|---|---|
| 🏆 **Logistic Regression (tuned)** | **0.943** | **0.748** | **0.215** |
| Gradient Boosting (tuned) | 0.935 | 0.717 | 0.242 |
| Random Forest | 0.933 | 0.716 | 0.244 |
| SVM | 0.933 | 0.729 | 0.233 |
| XGBoost / LightGBM / CatBoost | 0.92–0.94 | 0.68–0.70 | 0.24–0.28 |
| KNN / Naive Bayes | 0.93 | 0.71 | 0.25–0.26 |
| Decision Tree | 0.842 | 0.588 | 0.352 |

**Final model (Logistic Regression, tuned) — honest test-set performance:**

| Set | ROC-AUC | KS | Brier | Cost/app | θ |
|---|---|---|---|---|---|
| Validation | 0.9425 | 0.7310 | 0.0907 | 0.2300 | 0.64 |
| **TEST** | **0.9438** | **0.7483** | **0.0998** | **0.2150** | 0.52 |

> Test ≥ validation → **no overfitting**. At the operating point, the model
> approves ~68% of applicants while keeping the **approved-portfolio bad rate ≈ 2.2%**.

**Key drivers (SHAP / permutation):** engineered `risk_index` is the #1 feature
(even above `credit_score`), followed by `credit_score`, `interest_rate`, and
`payment_consistency` — confirming the value of domain feature engineering.

Figures (ROC/PR overlays, confusion matrix, calibration, lift, gain, SHAP, PDP)
are in [`reports/figures/`](reports/figures/).

---

## 🛠 Installation Guide

**Prerequisites:** Python ≥ 3.10.

```bash
# 1. Clone
git clone <your-repo-url>
cd credit-scoring-system

# 2. Create & activate a virtual environment
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# 3. Install dependencies (core only)
pip install -r requirements.txt
#    ...or core + dev tools (tests, lint, notebooks)
pip install -r requirements-dev.txt
```

> Note: `imbalanced-learn`, `xgboost`, `lightgbm`, `catboost`, `shap`, and `lime`
> are required for the full modeling/explainability workflow.

---

## 🚀 Quickstart / Reproducibility

Each phase is a runnable module (config-driven, fully seeded):

```bash
# Data
python -m src.preprocessing.data_generator      # generate synthetic data → data/raw/
python -m src.preprocessing.cleaner             # clean → data/processed/
python -m src.training.data_splitting           # stratified 64/16/20 split

# Modeling
python -m src.feature_engineering.feature_selection   # consensus feature selection
python -m src.training.train                     # compare 10 models (CV)
python -m src.training.tune                      # Randomized/Grid search + early stopping
python -m src.evaluation.evaluate                # full metric suite + charts
python -m src.evaluation.explainability          # SHAP / LIME / PDP

# Ship
python -m src.inference.finalize                 # save artifact + first test eval
python -m src.inference.service                  # inference service demo
python -m pytest tests/ -q                       # run the test suite
```

Or via the Makefile: `make install`, `make test`, `make lint`.

---

## 📁 Project Structure

```
credit-scoring-system/
├── data/                  raw/ (immutable) · processed/ (clean + splits) · external/
├── models/                serialized artifact (.joblib) + .meta.json + .pkl
├── notebooks/             EDA / storytelling only (logic lives in src/)
├── src/                   THE importable package
│   ├── preprocessing/         generator · cleaner · inspection · pipeline
│   ├── feature_engineering/   domain features · selection
│   ├── training/              models · splitting · tuning
│   ├── evaluation/            metrics · plots · explainability
│   ├── inference/             serialize · predict · service
│   ├── visualization/         EDA
│   ├── config/                YAML loader
│   └── utils/                 logger
├── config/                config.yaml        # single source of truth
├── artifacts/             selected_features.json · best_params.json
├── reports/               figures/ + *.md reports (DQ, EDA, eval, tuning, explainability)
├── logs/                  per-run structured logs
├── tests/                 pytest suite (unit · contract · edge · performance)
├── requirements.txt · requirements-dev.txt
├── pyproject.toml · Makefile · .gitignore · LICENSE
├── PROJECT_PLANNING.md · architecture_diagram.svg · MODEL_CARD.md
└── README.md
```

---

## 🗃 Dataset Description

A **realistic synthetic** financial dataset (**10,000 applicants**, 24 raw features
+ target) generated with a known, noisy **data-generating process** so the problem
is learnable (AUC ≈ 0.94) but not trivially separable. Deliberate data-quality
issues (missing values, duplicates, outliers) are injected so cleaning is meaningful.

| Group | Features |
|---|---|
| Demographic | age, dependents, education, employment_status, employment_years |
| Financial | monthly_income, monthly_expenses, savings_balance, total_assets, monthly_debt_payment |
| Credit history | num_open_accounts, num_credit_inquiries_6m, num_late_payments_12m, num_previous_defaults, months_since_last_delinquency, credit_utilization_ratio, credit_score |
| Loan | loan_amount, loan_term_months, interest_rate, loan_purpose |
| Target | `creditworthy` (1=repays, 0=default @90+ DPD) |

**Target balance:** 82.2% creditworthy / **17.8% defaulter** (imbalanced).

---

## 🧠 Model Summary

- **Algorithm:** tuned **Logistic Regression** (`C≈0.216`, L2, `lbfgs`,
  `class_weight='balanced'`).
- **Input:** 21 raw features → feature-engineered (11 domain features) →
  preprocessed → **26 selected** consensus features.
- **Why LogReg:** highest AUC + lowest business cost, **best interpretability**
  (coefficients = reason codes), instant train/serve, and **regulator-friendly**.
  (Boosting models are tuned challengers; ensemble-able.)
- **Calibration:** Brier ≈ 0.10 (improvable via isotonic/Platt if needed).
- **Decision:** threshold = 0.64 (validation cost-optimal; 0.52 on test).

Full details in [`MODEL_CARD.md`](MODEL_CARD.md).

---

## 🔍 Explainability

- **Global:** permutation importance + SHAP (`risk_index`, `credit_score`,
  `interest_rate` dominate).
- **Local:** LIME + LogReg coefficient×value → per-applicant **adverse-action
  reason codes** (e.g., "rejected due to low payment_consistency, low
  credit_score, high utilization").
- **Partial Dependence:** non-linear risk thresholds (e.g., default risk jumps
  past ~0.6 utilization).
- ⚠️ Honest finding: raw LogReg coefficients show **multicollinearity artifacts**
  (e.g., income sign flips) → rely on SHAP/permutation for importance; VIF-prune
  for production reason codes.

---

## 🧪 Testing

A **31-test pytest suite** (unit · contract · edge · performance), passing in ~1.6s:

```bash
make test      # or: python -m pytest tests/ -q
```

Covers metric math (AUC 1.0/0.0, cost), reproducibility, immutability, the full
service validation contract, empty/all-NaN/extreme inputs, and a batch-throughput
SLA. The suite **caught a real empty-input bug** that was then fixed.

---

## ⚠️ Limitations

- **Synthetic data:** the target is a (noisy) linear logit, which favors LogReg;
  real bureau data typically lets tuned boosting pull ahead by 1–3 AUC points.
- **No time dimension:** no true temporal split or concept-drift/PSI monitoring yet.
- **Calibration:** LogReg Brier ≈ 0.10 — acceptable, not best-in-class.
- **Multicollinearity** affects raw coefficient interpretation.
- **Fairness:** audited at a basic level (protected attrs excluded); full
  disparate-impact/Adverse-Impact-Ratio tracking across segments is future work.
- **Single model** deployed (no champion/champion ensemble).

---

## 🚧 Future Improvements

- **Champion–challenger** ensemble (LogReg + XGBoost) with stacked probabilities.
- **Probability calibration** (isotonic/Platt) for tighter risk pricing.
- **Temporal validation** + **PSI / drift monitoring** and automated retraining.
- **VIF-pruned scorecard** variant for clean reason codes.
- **Fairness deep-dive:** AIR, equal-opportunity difference, proxy audits.
- **REST/gRPC serving** wrapper around `CreditScoringService` + Docker + CI/CD.
- **Real data integration** + feature store.

---

## 📸 Gallery

| | |
|---|---|
| **Architecture** | **ROC curves (all models)** |
| ![architecture](architecture_diagram.svg) | ![roc](reports/figures/10_roc_curves.png) |
| **SHAP summary** | **Confusion matrix (LogReg)** |
| ![shap](reports/figures/12_shap_summary.png) | ![confusion](reports/figures/10_confusion_logistic_regression.png) |

*(More in `reports/figures/`: PR curves, calibration, lift, gain, threshold-cost,
PDP, LIME, EDA, correlation heatmap.)*

---

## 🧳 Portfolio & Repo Kit

- **[`PORTFOLIO.md`](PORTFOLIO.md)** — interview Q&A, resume bullets, LinkedIn description.
- **[`DELIVERABLES.md`](DELIVERABLES.md)** — every deliverable mapped to its file.
- **[`COMMITS.md`](COMMITS.md)** — clean, conventional-commit history strategy.
- **[`scripts/seed_git.sh`](scripts/seed_git.sh)** — replay a clean grouped git history.

---

## 📄 License

[MIT](LICENSE) — © 2026 Credit Scoring System.
