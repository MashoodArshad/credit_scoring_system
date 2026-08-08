# Feature Selection Report
_Features evaluated:_ 45  
_Final selected:_ **26** (consensus: selected by >=2 of MI/RFE/Permutation, minus near-constant & redundant)

## 1. Method comparison

| Method | Type | Keeps | Note |
|---|---|---|---|
| Variance threshold (>=0.01) | Filter (unsupervised) | 44 | dropped 1 near-constant |
| Correlation (>0.9) | Filter (redundancy) | - | 3 redundant pairs found |
| Mutual Information | Filter (target-aware) | top 25 | non-linear relevance |
| RFE (LogReg) | Wrapper | top 25 | model-specific ranking |
| Permutation (RF) | Agnostic | top 25 | non-linear, held-out scored |

## 2. Consensus ranking

**All features ranked by votes then MI**

```
                                           feature  mutual_info  perm_imp  rfe_kept  votes    vif  final_keep
16                                   interest_rate       0.1682    0.0114      True      3    6.0        True
15                                    credit_score       0.1647    0.0058      True      3   12.8        True
23                             payment_consistency       0.1314    0.0051      True      3  481.9        True
14                        credit_utilization_ratio       0.0929    0.0024      True      3   11.0        True
17                                       dti_ratio       0.0865    0.0020      True      3    9.5        True
11                           num_late_payments_12m       0.0854    0.0004      True      3  326.6        True
0                                   monthly_income       0.0519    0.0005      True      3   15.8        True
12                           num_previous_defaults       0.0519    0.0005      True      3  108.6        True
1                                 monthly_expenses       0.0336    0.0006      True      3    8.0        True
8                                 employment_years       0.0318    0.0012      True      3    8.7        True
22                                      risk_index       0.2045    0.0120     False      2   31.8        True
21                          financial_health_index       0.0995    0.0026     False      2   32.0        True
41                               credit_tier_Prime       0.0783   -0.0003      True      2    inf        True
28  missingindicator_months_since_last_delinquency       0.0782    0.0006     False      2    inf        True
26                         has_delinquency_history       0.0725    0.0004     False      2    inf       False
39                       credit_tier_Deep Subprime       0.0577   -0.0002      True      2    inf        True
2                                  savings_balance       0.0427   -0.0003      True      2    7.6        True
4                             monthly_debt_payment       0.0354    0.0012     False      2    2.1        True
18                             loan_monthly_burden       0.0298   -0.0003      True      2    2.2        True
25                                  maxed_out_flag       0.0271    0.0002     False      2    1.3        True
31                    employment_status_Unemployed       0.0271   -0.0002      True      2    inf        True
20                          credit_inquiry_density       0.0254    0.0005     False      2    4.9        True
24                                income_stability       0.0245    0.0005     False      2   13.3        True
10                         num_credit_inquiries_6m       0.0166    0.0004      True      2    4.3        True
30                 employment_status_Self-Employed       0.0078    0.0003      True      2    inf        True
35                          loan_purpose_Education       0.0063    0.0004      True      2    inf        True
29                      employment_status_Employed       0.0037    0.0007      True      2    inf        True
27                   months_since_last_delinquency       0.0745    0.0001     False      1   12.3       False
42                            credit_tier_Subprime       0.0451   -0.0006     False      1    inf       False
5                                      loan_amount       0.0076    0.0001      True      1    2.0       False
7                                       dependents       0.0060    0.0002     False      1    1.0       False
13                                loan_term_months       0.0005   -0.0005      True      1    1.5       False
44                                       education       0.0005    0.0005     False      1    1.0       False
9                                num_open_accounts       0.0000    0.0006     False      1    2.8       False
19                               savings_to_income       0.0000   -0.0007      True      1   23.6       False
33                           loan_purpose_Business       0.0000   -0.0002      True      1    inf       False
38                           loan_purpose_Personal       0.0000   -0.0001      True      1    inf       False
43                             credit_tier_missing       0.0000   -0.0000      True      1    inf       False
3                                     total_assets       0.0210   -0.0001     False      0    1.5       False
40                          credit_tier_Near-Prime       0.0025   -0.0003     False      0    inf       False
34                 loan_purpose_Debt Consolidation       0.0020   -0.0005     False      0    inf       False
6                                              age       0.0000   -0.0003     False      0    2.4       False
32                               loan_purpose_Auto       0.0000   -0.0003     False      0    inf       False
36                               loan_purpose_Home       0.0000    0.0002     False      0    inf       False
37                            loan_purpose_Medical       0.0000   -0.0001     False      0    inf       False
```

## 3. Multicollinearity (VIF > 10, caution for linear models)

```
                                                       vif
loan_purpose_Auto                                      inf
loan_purpose_Business                                  inf
loan_purpose_Debt Consolidation                        inf
loan_purpose_Education                                 inf
loan_purpose_Home                                      inf
loan_purpose_Medical                                   inf
loan_purpose_Personal                                  inf
credit_tier_Deep Subprime                              inf
credit_tier_Near-Prime                                 inf
credit_tier_Prime                                      inf
credit_tier_Subprime                                   inf
missingindicator_months_since_last_delinquency         inf
has_delinquency_history                                inf
credit_tier_missing                                    inf
employment_status_Unemployed                           inf
employment_status_Self-Employed                        inf
employment_status_Employed                             inf
payment_consistency                             481.868310
num_late_payments_12m                           326.649385
num_previous_defaults                           108.552600
financial_health_index                           31.999927
risk_index                                       31.844022
savings_to_income                                23.573896
monthly_income                                   15.797124
income_stability                                 13.321996
credit_score                                     12.793245
months_since_last_delinquency                    12.344223
credit_utilization_ratio                         10.955132
```

## 4. Decision & rationale

- **Final feature set (26):** kept features appearing in >=2 of {MI, RFE, Permutation} top-25, with near-constant and lower-MI redundant partners removed.
- Removed for redundancy: ['has_delinquency_history'].
- Tree/boosting models tolerate multicollinearity, so high-VIF composites (e.g. risk_index, FHI) are retained for them; for logistic regression we monitor VIF.
- This consensus is more robust than any single method and is auditable.