# Model Finalization & Inference Report (Phase 13)
_Artifact:_ `credit_scoring_logreg_v1.joblib` (v1.0.0)  
_Created:_ 2026-08-07T06:53:16+00:00  
_Decision threshold:_ 0.64

## 1. Final performance

```
validation: {"roc_auc": 0.9425, "ks": 0.731, "brier": 0.0907, "opt_threshold": 0.64, "cost_per_applicant": 0.23}
TEST      : {"roc_auc": 0.9438, "ks": 0.7483, "brier": 0.0998, "opt_threshold": 0.52, "cost_per_applicant": 0.215}
```

## 2. Serialization

- Primary: joblib -> `credit_scoring_logreg_v1.joblib` (+ `.meta.json` sidecar)
- Alternative: pickle -> `credit_scoring_logreg_v1.pkl`
- Reload verified to reproduce predictions byte-for-byte.

## 3. Inference demo (raw applicants)

```
   p_creditworthy  p_default decision       risk_tier
0          0.1408     0.8592   Reject  Very High Risk
1          0.2361     0.7639   Reject  Very High Risk
```

### Reason codes — applicant 1

```
                 feature  contribution       direction
     payment_consistency        -1.419 favor rejection
            credit_score        -0.883 favor rejection
credit_utilization_ratio        -0.724 favor rejection
           interest_rate        -0.523 favor rejection
```

### Reason codes — applicant 2

```
                 feature  contribution       direction
               dti_ratio        -0.762 favor rejection
            credit_score        -0.640 favor rejection
credit_utilization_ratio        -0.591 favor rejection
     payment_consistency         0.586  favor approval
```