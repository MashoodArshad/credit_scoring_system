# 📌 Portfolio Kit — Credit Scoring System

Everything you need to talk about, show, and pitch this project: interview Q&A,
resume bullets, and a ready-to-paste LinkedIn description.

---

## 🎤 Interview Questions (with answer cues)

### Project-specific / ML depth
1. **Why did Logistic Regression beat XGBoost here?**
   Strong feature engineering made the signal nearly linearly separable, and the
   synthetic target was generated as a *linear logit*, which LogReg matches
   exactly. LogReg also won on interpretability + calibration. *Honest caveat:*
   on real bureau data, tuned boosting usually edges ahead by 1–3 AUC points.
2. **How did you prevent data leakage?**
   Split *first*; preprocessing & feature selection fit on **train only** inside
   sklearn `Pipeline` (refit per CV fold); tuning uses CV folds from train; the
   test set is evaluated **once**, at finalization.
3. **What is the KS statistic, and why prefer it over accuracy?**
   KS = max separation between the good/bad score CDFs — a **rank-ordering**
   metric that's standard in credit risk. Accuracy is misleading under the 18%
   class imbalance.
4. **How did you handle the 17.8% class imbalance?**
   Stratified splits + `class_weight='balanced'` / `scale_pos_weight`; evaluated
   with AUC/KS/F1 (not accuracy); tuned the threshold on a cost objective.
5. **How did you choose the decision threshold?**
   Cost-minimization: search over thresholds to minimize `5·FP + 1·FN`. The
   default 0.5 is suboptimal when approving a defaulter is far costlier.
6. **Explain your feature engineering — why does `risk_index` matter?**
   It's a domain composite of burden + behavior, **deliberately independent of
   `credit_score`**, so it adds orthogonal signal. It had the **highest mutual
   information** of any feature.
7. **What's the FP vs FN cost in your model?**
   FP = approving a defaulter (cost 5×, loss given default); FN = rejecting a good
   customer (cost 1×, lost margin). Positive class = creditworthy.
8. **You found a multicollinearity issue with LogReg coefficients — explain.**
   Correlated features (income/​expenses/​savings/​ratios) flip coefficient signs
   (e.g., income appeared negative). Fix: rely on SHAP/permutation for importance,
   and VIF-prune for clean production reason codes.
9. **How would you deploy & monitor this?**
   Serialize the full pipeline; serve via `CreditScoringService` (validation +
   logging) behind REST/gRPC in Docker; monitor **PSI/drift** + fairness; retrain
   on a schedule.
10. **SHAP vs LIME vs permutation importance — when?**
    SHAP = rigorous global+local (gold standard); LIME = fast local surrogate for
    per-applicant reasons; permutation = cheap model-agnostic global ranking.
11. **Why joblib over pickle?**
    joblib is optimized for numpy/sklearn and compresses arrays; pickle is a
    security risk (arbitrary code execution). *Gotcha:* lambdas aren't picklable —
    I hit that with a `FunctionTransformer` and fixed it with a named function.
12. **How is the pipeline structured to avoid train/serve skew?**
    One end-to-end artifact (`FeatureEngineer → Preprocess → Selector → Model`);
    inference consumes **raw** data, so train and serve transform identically.

### Software engineering
13. **Why config-driven design?**
    Reproducibility, separation of concerns (code = *how*, config = *what*), and
    environment portability.
14. **Describe your testing strategy.**
    Test pyramid: unit (pure logic), contract (service↔artifact), edge cases,
    performance. AAA pattern, session fixtures, deterministic seeds. The suite
    **caught a real empty-input bug** I then fixed.
15. **How do you guarantee reproducibility?**
    Pinned seeds threaded through every step + versioned JSON metadata (env,
    threshold, metrics, features) + reload-verified predictions.

### Behavioral
16. **A challenge you overcame.** → Debugging the `get_feature_names_out` mismatch
    in a Pipeline/ColumnTransformer composition (root-caused to a missing
    `feature_names_in_`).
17. **A tradeoff you made.** → Chose interpretability + lower cost (LogReg) over a
    marginal AUC gain from boosting — the right call for regulated credit.

---

## 📄 Resume Bullet Points

Pick 3–5. All are **quantified + action-oriented (STAR/XYZ style)**.

- **Engineered a production-grade credit-scoring ML pipeline** (10k applicants)
  achieving **0.94 ROC-AUC / 0.75 KS**; tuned a cost-optimal threshold that cut
  the approved-portfolio bad rate to **~2%** at **68% approval**.
- **Designed a leakage-safe sklearn pipeline** (feature engineering →
  preprocessing → selection → model) and **compared 10 algorithms** with
  stratified 5-fold CV, selecting the best on AUC **and** business cost.
- **Created 11 domain features** (DTI, financial-health index, risk index);
  the engineered **risk_index became the #1 predictor** by mutual information,
  outperforming the raw credit score.
- **Built model explainability** (SHAP, LIME, PDP, permutation importance) with
  per-applicant **adverse-action reason codes** for regulatory compliance.
- **Shipped a versioned model artifact + validated inference service** (input
  schema, exception handling, structured logging) backed by a **31-test pytest
  suite** that caught real edge-case bugs.
- **Reduced data leakage & skew risk** by fitting preprocessing/selection inside
  CV folds and persisting a single end-to-end pipeline for train/serve parity.

---

## 💼 LinkedIn Project Description

> Built an end-to-end, production-ready **credit-scoring system** using machine
> learning — predicting applicant creditworthiness with **0.94 ROC-AUC** and a
> **KS of 0.75**, while keeping the model interpretable, fair, and auditable.
>
> 🔧 **What I did:**
> - Generated & cleaned a realistic 10k-row financial dataset and ran full EDA.
> - Engineered 11 domain features (debt-to-income, financial-health index, risk
>   index) — my engineered `risk_index` became the single strongest predictor.
> - Compared 10 algorithms under a leakage-safe pipeline and tuned the winner
>   (Logistic Regression) with Randomized/Grid search + early stopping.
> - Selected the decision threshold by **business-cost minimization** (a false
>   approval costs 5× a false rejection).
> - Added explainability (SHAP/LIME/PDP) with per-applicant reason codes.
> - Packaged everything as a versioned artifact + a validated inference service,
>   covered by a 31-test pytest suite.
>
> 🧠 **Tech:** Python, scikit-learn, XGBoost/LightGBM/CatBoost, SHAP, pandas,
> NumPy, joblib, pytest, YAML-driven config. Practices: leakage-safe pipelines,
> stratified CV, model cards, structured logging, modular `src/` package.
>
> 📈 **Result:** test AUC 0.94 / KS 0.75, ~2% bad rate at 68% approval — and a
> clean, recruiter-ready GitHub repo documenting every decision and limitation.

*(Hashtags: #MachineLearning #CreditRisk #DataScience #MLOps #Python #Portfolio)*

---

## 📂 Where to find everything
- **Results & reports:** `reports/` (evaluation, tuning, explainability, DQ, EDA)
- **Figures (for screenshots):** `reports/figures/`
- **Model artifact:** `models/credit_scoring_logreg_v1.joblib`
- **Code:** `src/` (preprocessing · feature_engineering · training · evaluation · inference)
- **Tests:** `tests/`
- **Docs:** `README.md`, `MODEL_CARD.md`, `PROJECT_PLANNING.md`,
  `architecture_diagram.svg`, `COMMITS.md`
