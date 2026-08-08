# Model Explainability Report (Phase 12)
_Models explained:_ tuned Logistic Regression (primary) + tuned XGBoost (challenger). Reason codes come from LogReg; SHAP/PDP from the tree model.

## 1. Reason codes — Logistic Regression coefficients

Coefficients are the change in log-odds of approval per +1 (scaled) unit. `odds_ratio = exp(coef)`. Green = increases approval odds; red = decreases.
```
                                                coefficient  odds_ratio                   effect
payment_consistency                                   0.968       2.633  increases approval odds
credit_score                                          0.835       2.305  increases approval odds
monthly_expenses                                      0.476       1.610  increases approval odds
credit_tier_Deep Subprime                             0.265       1.304  increases approval odds
employment_status_Self-Employed                       0.220       1.246  increases approval odds
financial_health_index                                0.207       1.230  increases approval odds
loan_purpose_Education                                0.198       1.219  increases approval odds
employment_years                                      0.198       1.219  increases approval odds
income_stability                                      0.197       1.218  increases approval odds
missingindicator_months_since_last_delinquency        0.144       1.155  increases approval odds
num_late_payments_12m                                 0.122       1.130  increases approval odds
savings_balance                                       0.062       1.064  increases approval odds
monthly_debt_payment                                 -0.002       0.998  decreases approval odds
employment_status_Employed                           -0.041       0.960  decreases approval odds
risk_index                                           -0.070       0.933  decreases approval odds
credit_inquiry_density                               -0.081       0.922  decreases approval odds
maxed_out_flag                                       -0.099       0.905  decreases approval odds
loan_monthly_burden                                  -0.124       0.884  decreases approval odds
employment_status_Unemployed                         -0.156       0.856  decreases approval odds
credit_tier_Prime                                    -0.209       0.812  decreases approval odds
monthly_income                                       -0.233       0.792  decreases approval odds
interest_rate                                        -0.354       0.702  decreases approval odds
num_previous_defaults                                -0.472       0.624  decreases approval odds
dti_ratio                                            -0.483       0.617  decreases approval odds
credit_utilization_ratio                             -0.576       0.562  decreases approval odds
num_credit_inquiries_6m                              -0.810       0.445  decreases approval odds
```

## 2. Top approval drivers

```
                                 coefficient  odds_ratio                   effect
payment_consistency                    0.968       2.633  increases approval odds
credit_score                           0.835       2.305  increases approval odds
monthly_expenses                       0.476       1.610  increases approval odds
credit_tier_Deep Subprime              0.265       1.304  increases approval odds
employment_status_Self-Employed        0.220       1.246  increases approval odds
financial_health_index                 0.207       1.230  increases approval odds
loan_purpose_Education                 0.198       1.219  increases approval odds
employment_years                       0.198       1.219  increases approval odds
```

## 3. Top rejection drivers

```
                              coefficient  odds_ratio                   effect
num_credit_inquiries_6m            -0.810       0.445  decreases approval odds
credit_utilization_ratio           -0.576       0.562  decreases approval odds
dti_ratio                          -0.483       0.617  decreases approval odds
num_previous_defaults              -0.472       0.624  decreases approval odds
interest_rate                      -0.354       0.702  decreases approval odds
monthly_income                     -0.233       0.792  decreases approval odds
credit_tier_Prime                  -0.209       0.812  decreases approval odds
employment_status_Unemployed       -0.156       0.856  decreases approval odds
```

## 4. Permutation importance (model-agnostic)

```
payment_consistency         0.0491
credit_score                0.0260
dti_ratio                   0.0140
credit_utilization_ratio    0.0114
monthly_expenses            0.0050
loan_monthly_burden         0.0050
num_credit_inquiries_6m     0.0048
interest_rate               0.0030
income_stability            0.0017
employment_years            0.0017
```

## 5. SHAP mean |contribution| (XGBoost)

```
risk_index                                        1.1575
credit_score                                      0.6898
interest_rate                                     0.5814
missingindicator_months_since_last_delinquency    0.3344
monthly_expenses                                  0.1959
credit_inquiry_density                            0.1691
financial_health_index                            0.1657
income_stability                                  0.1635
payment_consistency                               0.1419
employment_years                                  0.0961
```

## 6. Business narrative

- The strongest APPROVAL drivers (raise log-odds): payment_consistency, credit_score, monthly_expenses, credit_tier_Deep Subprime, employment_status_Self-Employed
- The strongest REJECTION drivers (lower log-odds): num_credit_inquiries_6m, credit_utilization_ratio, dti_ratio, num_previous_defaults, interest_rate
- These map directly to underwriting logic: capacity (income/savings), discipline (payment consistency, no delinquency), and burden (DTI, utilization).

## 7. Local explanations (LIME)

- [approved_good] P(creditworthy)=0.69 | -0.34405990229657685 payment_consistency <= -0.43; 0.09522950523363598 num_previous_defaults <= 0.00; 0.08169199098723676 0.06 < credit_score <= 0.79; -0.046728348521751485 credit_tier_Deep Subprime <= 0.00; -0.04642712193625133 employment_status_Self-Employed <= 0.00
- [rejected_bad] P(creditworthy)=0.28 | -0.3284483438734609 payment_consistency <= -0.43; -0.24725285701480126 credit_score <= -0.65; -0.09924407471108143 monthly_expenses <= -0.53; -0.07950140730152269 interest_rate > 0.64; -0.07462590078093043 loan_monthly_burden > 0.68

## 8. Figures (reports/figures/)

- `12_logreg_coefficients.png` — reason codes (signed)
- `12_permutation_importance.png` — model-agnostic ranking
- `12_shap_summary.png` — SHAP beeswarm (XGBoost)
- `12_partial_dependence.png` — PDPs for top features
- `12_lime_approved_good.png` / `12_lime_rejected_bad.png` — per-applicant explanations