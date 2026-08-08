"""Model registry: production-default estimators for credit scoring.

Each estimator is configured with sensible, imbalanced-data-aware defaults.
``class_weight='balanced'`` (or its boosting equivalent) compensates for the
~18% minority class so models don't trivially predict the majority class.

The trade-off reference (advantages / disadvantages / interpretability /
business suitability) is encoded in :data:`MODELS_REFERENCE` so it can be
rendered into the comparison report automatically.
"""
from __future__ import annotations

from typing import Any

from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

# Boosting libraries (all confirmed available in the environment).
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier


def get_models(
    seed: int = 42,
    n_jobs: int = -1,
    scale_pos_weight: float = 4.6,
) -> dict[str, Any]:
    """Return a registry of candidate estimators with imbalanced-aware defaults.

    Args:
        seed: Random seed threaded into every stochastic estimator.
        n_jobs: Thread count for parallel-fit estimators (RF, KNN, boosting).
        scale_pos_weight: Neg/pos ratio used to weight the minority class for
            XGBoost (computed from the training target).

    Returns:
        Dict mapping a human-readable model name to an unfitted estimator.
    """
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=seed,
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=8, class_weight="balanced", random_state=seed,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, class_weight="balanced", random_state=seed, n_jobs=n_jobs,
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=200, max_depth=3, learning_rate=0.1, random_state=seed,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.1,
            subsample=0.9, colsample_bytree=0.9, scale_pos_weight=scale_pos_weight,
            tree_method="hist", eval_metric="logloss", random_state=seed,
            n_jobs=n_jobs, verbosity=0,
        ),
        "LightGBM": LGBMClassifier(
            n_estimators=300, num_leaves=31, learning_rate=0.1,
            subsample=0.9, colsample_bytree=0.9, is_unbalance=True,
            random_state=seed, n_jobs=n_jobs, verbose=-1,
        ),
        "CatBoost": CatBoostClassifier(
            iterations=300, depth=6, learning_rate=0.1,
            auto_class_weights="Balanced", random_state=seed, verbose=0,
        ),
        "SVM": SVC(
            kernel="rbf", C=1.0, gamma="scale", class_weight="balanced",
            probability=True, random_state=seed, cache_size=500,
        ),
        "KNN": KNeighborsClassifier(n_neighbors=15, n_jobs=n_jobs),
        "Naive Bayes": GaussianNB(),
    }


# Trade-off reference for the comparison report (business-friendly summary).
MODELS_REFERENCE: dict[str, dict[str, str]] = {
    "Logistic Regression": {
        "advantages": "Fast, highly interpretable, calibrated, regulatory-friendly.",
        "disadvantages": "Linear only; underfits complex interactions.",
        "interpretability": "Very high (coefficients = reason codes).",
        "business_suitability": "Excellent baseline & for regulated deployments.",
    },
    "Decision Tree": {
        "advantages": "Interpretable rules, captures non-linearity.",
        "disadvantages": "High variance, overfits easily.",
        "interpretability": "High (single tree path).",
        "business_suitability": "Good for explanation, weak as a final model.",
    },
    "Random Forest": {
        "advantages": "Robust, low-variance, handles non-linearity & interactions.",
        "disadvantages": "Less interpretable, larger memory, slower inference.",
        "interpretability": "Medium (feature importance / tree paths).",
        "business_suitability": "Strong, reliable production workhorse.",
    },
    "Gradient Boosting": {
        "advantages": "Often top accuracy, handles mixed signals.",
        "disadvantages": "Tunable, slower to train, prone to overfit.",
        "interpretability": "Medium.",
        "business_suitability": "Strong when tuned; good tabular performer.",
    },
    "XGBoost": {
        "advantages": "State-of-the-art on tabular, fast, regularized.",
        "disadvantages": "Many hyperparameters, memory on large data.",
        "interpretability": "Medium (SHAP-friendly).",
        "business_suitability": "Industry standard for credit risk.",
    },
    "LightGBM": {
        "advantages": "Very fast, memory-efficient, strong accuracy.",
        "disadvantages": "Leaf-wise growth can overfit small data.",
        "interpretability": "Medium.",
        "business_suitability": "Excellent for large-scale scoring.",
    },
    "CatBoost": {
        "advantages": "Native categorical handling, strong out-of-the-box.",
        "disadvantages": "Slower training, larger footprint.",
        "interpretability": "Medium-High (ordered boosting).",
        "business_suitability": "Great when many categoricals; less prep.",
    },
    "SVM": {
        "advantages": "Strong margins in high-dim, flexible kernels.",
        "disadvantages": "Slow on large n, poor scaling, weak calibration.",
        "interpretability": "Low.",
        "business_suitability": "Rare in credit (slow, opaque); benchmark only.",
    },
    "KNN": {
        "advantages": "Simple, no training, non-parametric.",
        "disadvantages": "Slow inference, curse of dimensionality, sensitive to scale.",
        "interpretability": "Low-Medium (similar cases).",
        "business_suitability": "Poor for real-time credit decisions.",
    },
    "Naive Bayes": {
        "advantages": "Very fast, works with little data, probabilistic.",
        "disadvantages": "Strong independence assumption, usually weakest.",
        "interpretability": "High.",
        "business_suitability": "Useful baseline/sanity-check, rarely final.",
    },
}
