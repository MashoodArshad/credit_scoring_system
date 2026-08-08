# Evaluation Report (Phase 10)
_Evaluation set:_ Validation (held-out)  
_Cost model:_ FP(approve a defaulter)=5.0 x  |  FN(reject a good customer)=1.0  
_Operating point:_ cost-optimal threshold

## 1. Metric suite (all models, validation)

Metrics below use each model's **cost-optimal threshold**; AUC/PR-AUC/KS/Brier are threshold-independent.
```
 rank               model  roc_auc  pr_auc     ks  brier  opt_threshold  approval_rate@opt  bad_rate@opt  recall@opt  specificity@opt  f1@opt  cost_per_applicant@opt
    1 Logistic Regression   0.9426  0.9849 0.7314 0.0904           0.64             0.6806        0.0220      0.8093           0.9155  0.8857                  0.2319
    2   Gradient Boosting   0.9336  0.9829 0.7104 0.0693           0.93             0.6406        0.0176      0.7652           0.9366  0.8603                  0.2494
    3                 SVM   0.9329  0.9795 0.7289 0.0716           0.84             0.7206        0.0304      0.8495           0.8768  0.9056                  0.2331
    4       Random Forest   0.9326  0.9812 0.7156 0.0685           0.81             0.7400        0.0363      0.8670           0.8486  0.9128                  0.2437
    5                 KNN   0.9305  0.9772 0.7129 0.0687           0.74             0.7800        0.0441      0.9065           0.8063  0.9306                  0.2487
    6         Naive Bayes   0.9291  0.9811 0.7063 0.1026           0.99             0.7594        0.0436      0.8830           0.8134  0.9182                  0.2619
    7             XGBoost   0.9278  0.9800 0.6948 0.0841           0.63             0.7456        0.0419      0.8685           0.8239  0.9111                  0.2644
    8            CatBoost   0.9270  0.9815 0.6867 0.0813           0.81             0.6775        0.0295      0.7994           0.8873  0.8767                  0.2650
    9            LightGBM   0.9229  0.9796 0.6816 0.0820           0.80             0.7706        0.0487      0.8913           0.7887  0.9204                  0.2769
   10       Decision Tree   0.8423  0.9401 0.5883 0.1244           0.82             0.6769        0.0508      0.7812           0.8063  0.8570                  0.3519
```

## 2. Recommendation

- **Recommended model: Logistic Regression**
  - ROC-AUC=0.9426, KS=0.7314, PR-AUC=0.9849, Brier=0.0904
  - At cost-optimal threshold 0.64: approval rate=68.1%, bad rate(among approved)=2.2%, cost/applicant=0.232
- Selected by: highest ROC-AUC, then lowest business cost within the AUC top-tier, with interpretability/calibration as tie-breakers (favoring regulatory-ready models).

## 3. Charts (reports/figures/)

- `10_roc_curves.png` / `10_pr_curves.png` — all-model overlays
- Per leading model (Logistic Regression, Gradient Boosting): `10_confusion_<model>.png`, `10_calibration_<model>.png`, `10_lift_<model>.png`, `10_gain_<model>.png`, `10_threshold_cost_<model>.png`