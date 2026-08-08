# Data Split & Cross-Validation Report
_Stratification column:_ `creditworthy`  
_Seed:_ pinned (random_state=42)  
_CV:_ Stratified 5-fold (on train only)

## 1. Partition sizes & class balance

```
     split  n_rows  pct_of_total  creditworthy(1)  defaulter(0)  default_rate
     train    6400          64.0           0.8222        0.1778        0.1778
validation    1600          16.0           0.8225        0.1775        0.1775
      test    2000          20.0           0.8220        0.1780        0.1780
```

**Check:** default rate is preserved across all splits (stratification working).

## 2. Cross-validation strategy

- **Stratified 5-fold** CV is used for model comparison and tuning.
- Folds are drawn from the TRAIN set only; validation/test are never in CV.
- A single validation set additionally supports early-stopping for boosting models.

## 3. Leakage-prevention protocol (enforced project-wide)

- Split FIRST, before fitting any imputer/scaler/selector.
- Preprocessing pipeline fit on TRAIN ONLY, then applied to val/test.
- Feature selection fit on TRAIN ONLY.
- Hyperparameter tuning uses CV folds from TRAIN ONLY; test set untouched.
- TEST set evaluated exactly ONCE, for final reporting.
- No target-derived or post-outcome information used as a feature.
- Point-in-time features only (measured as-of application).