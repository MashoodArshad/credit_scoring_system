# Credit Scoring System — Project Planning Document (Phase 1)

> **Status:** Planning · **Owner:** ML Team · **Document type:** Business + ML specification

---

## 1. Business Problem

Financial institutions must decide, **before granting credit**, whether an applicant
will repay their obligation. Wrong decisions are expensive in *both* directions:

| Decision | Reality | Consequence |
|----------|---------|-------------|
| Approve | Customer **defaults** | Loss of principal + interest + collection cost |
| Reject  | Customer would have **repaid** | Lost revenue + churn + reputational risk |

Traditional underwriting relies on manual rules and linear scorecards (e.g., FICO-style
logistic models). These are interpretable but capture only linear relationships and miss
non-linear, interaction-driven patterns in modern data. **Machine learning can reduce
expected credit loss while keeping approval volume healthy — but only if engineered for
fairness, stability, and interpretability.**

---

## 2. Business Goal

1. **Reduce the default (Non-Performing Loan) rate** in the approved portfolio.
2. **Increase approval rate** among genuinely creditworthy applicants.
3. **Automate & accelerate** credit decisions (instant, digital-first lending).
4. **Reduce manual underwriting cost**.
5. Produce **auditable, explainable, regulator-ready** decisions.

---

## 3. Machine Learning Goal

A **supervised binary classification** problem:

> Predict the probability `P(default)` for each applicant, then convert to a
> **risk score + risk tier** at a business-chosen operating threshold.

- **Probabilistic output** → enables threshold tuning, calibration, and tiered pricing.
- **Rank-ordering power** matters more than raw accuracy (KS / Gini / AUC).

---

## 4. Success Criteria

| Category | Metric | Target (illustrative) |
|----------|--------|------------------------|
| Discrimination | AUC-ROC | ≥ 0.80 (vs. business baseline) |
| Rank-ordering | **KS statistic** (industry-standard) | ≥ 0.40 |
| Rank-ordering | Gini coefficient | ≥ 0.60 |
| Stability | Population Stability Index (PSI) | < 0.10 across time windows |
| Fairness | Adverse Impact Ratio (AIR) | ≥ 0.80 across groups |
| Business | Bad rate in approved segment | Reduce ≥ 20% vs. current policy |
| Business | Approval rate | Maintain/improve vs. current policy |

> **Why AUC alone is not enough:** AUC measures ranking, not the cost-weighted decision.
> Credit teams care about **KS**, **bad rate at a given approval rate**, and
> **expected loss**. We optimize for the business loss function, not pure accuracy.

---

## 5. Business Constraints

- **Regulatory compliance:** fair-lending / equal-credit rules (e.g., FCRA/EFTA-style,
  SBP consumer-protection rules in the PK context), Basel model-risk standards.
- **Explainability:** adverse-action **reason codes** may be legally required when
  declining an applicant.
- **Cost asymmetry:** the cost of a **False Negative** (approving a defaulter) is usually
  **10–50×** the cost of a **False Positive** (rejecting a good customer).
- **Latency:** decisioning typically required in seconds for digital lending.
- **Fairness:** no disparate impact on protected groups; no use of protected attributes
  *or strong proxies*.

---

## 6. Technical Constraints

- **Class imbalance:** defaulters are a minority (often 5–20%).
- **Concept drift:** borrower behavior shifts with macroeconomic cycles.
- **Data leakage:** using post-outcome information inflates metrics catastrophically.
- **Reproducibility:** identical inputs → identical model (seed pinning, versioning).
- **Interpretability ceiling:** some regulated deployments prefer inherently
  interpretable models (logistic/GBDT-with-reason-codes).

---

## 7. KPIs (Key Performance Indicators)

**Model KPIs:** AUC, KS, Gini, Brier score (calibration), PSI.
**Business KPIs:** Default/Bad rate, Approval rate, Expected loss, ROI of the scorecard.
**Operational KPIs:** Decision latency, uptime, monitoring coverage.
**Fairness KPIs:** AIR, demographic parity difference, equal-opportunity difference.

---

## 8. Target Variable

- **`creditworthy`** — binary:
  - `1` = creditworthy (customer **did not** default within the observation window)
  - `0` = defaulter
- **Default definition (industry standard):** customer became **90+ Days Past Due (DPD)**
  within the performance window.
- **Observation vs. performance windows:** features are measured *as-of application*
  (observation window); the label is measured *after* (performance window) to prevent leakage.

> ⚠️ We treat the **positive / "good" class as `1`** for creditworthiness. We will
> carefully state the positive class in every metric to avoid confusion between
> precision/recall for *default* vs. *good*.

---

## 9. Feature Categories

| Category | Example features | Notes |
|----------|------------------|-------|
| **Demographic / Stability** | age, employment tenure, residence tenure | Beware of fairness & proxy risk |
| **Financial Capacity** | monthly income, total debt, DTI, assets | Core affordability signals |
| **Credit History** | payment history, # delinquencies, # active accounts, # inquiries | Strongest predictors |
| **Loan-Specific** | loan amount, term, interest rate, purpose | Loan-level risk |
| **Behavioral** | savings consistency, transaction regularity | For existing customers |
| **Engineered (Phase 6)** | DTI, credit utilization, savings ratio, financial-health index | Domain-driven |

**Protected attributes (age-in-narrow-window, gender, race, religion, marital status) are
excluded from training** and used *only* for fairness auditing.

---

## 10. Potential Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| **Data leakage** | Over-optimistic, fails in prod | Point-in-time feature design; leak checklist |
| **Class imbalance** | Model ignores minority | Stratified split, class weights, proper thresholding |
| **Concept drift** | Decay in performance | PSI monitoring, scheduled retraining |
| **Unfairness / bias** | Legal & reputational harm | Fairness audit, proxy checks, AIR |
| **Poor calibration** | Wrong risk pricing | Calibration curve, Platt/isotonic |
| **Data quality** | Garbage in/out | Automated DQ report + tests |

---

## 11. Ethical Considerations

- **Bias & Fairness:** detect and limit disparate impact; audit on group-level metrics;
  reject strong proxies for protected attributes.
- **Interpretability:** provide reason codes (SHAP/feature contributions) for declines.
- **Privacy:** minimize/encrypt PII; never persist raw PII in artifacts/reports.
- **Accountability:** human-in-the-loop for high-value/borderline cases; appeals path.
- **Transparency:** ship a **Model Card** documenting intended use, limitations, metrics.

---

## 12. Project Architecture (High Level)

```
                  ┌─────────────────────────────────────────────┐
   CONFIG (YAML)  │  Centralized, version-controlled parameters  │  ◄──── single source of truth
                  └─────────────────────────────────────────────┘
   DATA SOURCES ──► Ingestion & Validation ──► Preprocessing ──► Feature Engineering
                         (DQ report)            (cleaning)        (domain features)
                                                                      │
                            ◄──── Train/Val/Test  ◄──── Leakage-safe Split ◄────┘
                                 (Stratified, seed-pinned)
                                      │
            ┌─────────────────────────┴─────────────────────────┐
   Multiple models (LogReg, RF, GBM, XGB, LGBM, SVM, KNN, NB…)   │
            └─────────────────────────┬─────────────────────────┘
                                      ▼
   Evaluation (AUC/KS/Gini/PR/Calibration/Cost) ──► Tuning ──► Explainability (SHAP/LIME)
                                      │
                                      ▼
   Model Registry / Serialization (joblib, versioned) ──► Inference Pipeline (validation+logging)
                                      │
                                      ▼
   Monitoring (PSI drift, fairness, performance) ──► Retraining loop

   CROSS-CUTTING:  Logging · Tests (pytest) · CI/CD · Artifacts · Reports/Figures
```

A full diagram is provided in `architecture_diagram.svg`.
```
