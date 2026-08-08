# Model Comparison Report (Phase 9)
_Protocol:_ leakage-safe Pipeline (FeatureEngineer -> preprocess -> selector -> model), stratified 5-fold CV on TRAIN + validation check.

## 1. Measured results (ranked by CV ROC-AUC)

```
 rank               model  cv_auc_mean  cv_auc_std  val_auc  cv_time_s status
    1 Logistic Regression       0.9466      0.0075   0.9426       0.68     ok
    2   Gradient Boosting       0.9416      0.0094   0.9336      20.85     ok
    3       Random Forest       0.9404      0.0093   0.9326       6.73     ok
    4             XGBoost       0.9382      0.0105   0.9278       1.71     ok
    5            CatBoost       0.9367      0.0085   0.9270       4.44     ok
    6            LightGBM       0.9348      0.0110   0.9229       2.25     ok
    7                 SVM       0.9331      0.0073   0.9329       2.95     ok
    8         Naive Bayes       0.9296      0.0129   0.9291       0.61     ok
    9                 KNN       0.9232      0.0126   0.9305       0.91     ok
   10       Decision Tree       0.8442      0.0240   0.8423       0.90     ok
```

## 2. Recommendation

- **Best by CV AUC:** **Logistic Regression** (CV AUC=0.9466 +/- 0.0075, val AUC=0.9426).
- Boosting models (XGBoost / LightGBM / CatBoost) and Random Forest are expected to lead on tabular credit data; Logistic Regression is kept as the interpretable baseline.
- Final selection considers AUC **and** interpretability/calibration/business cost (deep evaluation in Phase 10, tuning in Phase 11).

## 3. Model trade-off reference

```
              model                                                   advantages                                                disadvantages                          interpretability                            business_suitability
Logistic Regression Fast, highly interpretable, calibrated, regulatory-friendly.                 Linear only; underfits complex interactions.  Very high (coefficients = reason codes). Excellent baseline & for regulated deployments.
  Gradient Boosting                   Often top accuracy, handles mixed signals.                  Tunable, slower to train, prone to overfit.                                   Medium.      Strong when tuned; good tabular performer.
      Random Forest  Robust, low-variance, handles non-linearity & interactions.         Less interpretable, larger memory, slower inference. Medium (feature importance / tree paths).          Strong, reliable production workhorse.
            XGBoost              State-of-the-art on tabular, fast, regularized.                  Many hyperparameters, memory on large data.                   Medium (SHAP-friendly).              Industry standard for credit risk.
           CatBoost          Native categorical handling, strong out-of-the-box.                           Slower training, larger footprint.           Medium-High (ordered boosting).        Great when many categoricals; less prep.
           LightGBM                Very fast, memory-efficient, strong accuracy.                     Leaf-wise growth can overfit small data.                                   Medium.              Excellent for large-scale scoring.
                SVM                Strong margins in high-dim, flexible kernels.             Slow on large n, poor scaling, weak calibration.                                      Low.  Rare in credit (slow, opaque); benchmark only.
        Naive Bayes            Very fast, works with little data, probabilistic.             Strong independence assumption, usually weakest.                                     High.     Useful baseline/sanity-check, rarely final.
                KNN                         Simple, no training, non-parametric. Slow inference, curse of dimensionality, sensitive to scale.               Low-Medium (similar cases).            Poor for real-time credit decisions.
      Decision Tree                 Interpretable rules, captures non-linearity.                              High variance, overfits easily.                  High (single tree path).    Good for explanation, weak as a final model.
```