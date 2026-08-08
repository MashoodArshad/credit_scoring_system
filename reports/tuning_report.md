# Hyperparameter Optimization Report (Phase 11)
_Protocol:_ RandomizedSearchCV -> GridSearchCV refinement, stratified 5-fold CV on the full leakage-safe Pipeline; early stopping demo for XGBoost on a held-out eval set.

## 1. Before vs. After tuning (validation, cost-optimal threshold)

```
              model   stage  roc_auc     ks  brier  opt_threshold  cost_per_applicant  best_cv_auc
Logistic Regression default   0.9426 0.7314 0.0904           0.64              0.2319          NaN
Logistic Regression   tuned   0.9425 0.7310 0.0907           0.64              0.2300       0.9467
            XGBoost default   0.9183 0.6731 0.0853           0.75              0.2819          NaN
            XGBoost   tuned   0.9354 0.7168 0.0905           0.56              0.2419       0.9436
```

## 2. Best hyperparameters

```json
{
  "Logistic Regression": {
    "C": 0.21568228019263963,
    "penalty": "l2",
    "solver": "lbfgs"
  },
  "XGBoost": {
    "subsample": 0.8,
    "reg_lambda": 1.0,
    "n_estimators": 600,
    "min_child_weight": 5,
    "max_depth": 5,
    "learning_rate": 0.01,
    "colsample_bytree": 1.0
  }
}
```

## 3. Early stopping (XGBoost)

- Trained up to 2000 rounds with ``early_stopping_rounds=30``.
- Stopped at **best_iteration=449** (val AUC=0.9304) — avoids overfitting & saves compute.

## 4. Notes

- Logistic Regression is low-dimensional (mainly C); gains from tuning are modest but the refined grid locks in the most regularized generalizing setting.
- XGBoost benefits more from tuning (depth, learning rate, subsampling, reg_lambda).
- If tuning does not beat the default, we keep the default — parsimony over complexity.