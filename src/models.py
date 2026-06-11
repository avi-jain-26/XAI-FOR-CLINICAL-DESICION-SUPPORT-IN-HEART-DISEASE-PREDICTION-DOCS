"""
Predictive models and the stacked ensemble.

Base learners follow Section IV-A of the paper, using the "full potential"
configurations the authors reference:
  * Logistic Regression  -> ElasticNet (L1+L2)            [Zou et al.]
  * Random Forest         -> tuned + class-balanced        [Chen et al.]
  * SVM                   -> RBF kernel + C optimization    [Liu et al.]
  * Gradient Boosting     -> shrinkage + early stopping     [Fitriyani et al.]
  * XGBoost               -> advanced regularization        [Sheridan et al.]

These are combined with a Logistic Regression meta-learner in a
`StackingClassifier` (the paper's "Stacked Ensemble (LR + RF + SVM + GB + XGB)").
"""
from __future__ import annotations

from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
    StackingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.svm import SVC
from xgboost import XGBClassifier

import config as cfg


def build_base_models() -> dict:
    """Return the five individual classifiers, keyed by short name."""
    rs = cfg.RANDOM_STATE
    return {
        "Logistic Regression": LogisticRegression(
            penalty="elasticnet", solver="saga", l1_ratio=0.5, C=1.0,
            class_weight="balanced", max_iter=2000, random_state=rs,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, max_depth=None, min_samples_leaf=2,
            class_weight="balanced_subsample", n_jobs=-1, random_state=rs,
        ),
        "SVM (RBF)": SVC(
            kernel="rbf", C=2.0, gamma="scale", probability=True,
            class_weight="balanced", random_state=rs,
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=300, learning_rate=0.05, max_depth=3,        # shrinkage
            subsample=0.9, validation_fraction=0.1, n_iter_no_change=15,  # early stopping
            random_state=rs,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=400, learning_rate=0.05, max_depth=4,
            subsample=0.9, colsample_bytree=0.9,
            reg_lambda=2.0, reg_alpha=0.5,                            # advanced regularization
            eval_metric="logloss", tree_method="hist",
            n_jobs=-1, random_state=rs,
        ),
    }


def build_stacking(base_models: dict) -> StackingClassifier:
    """Build the stacked ensemble from the base models (LR meta-learner)."""
    estimators = [(name, model) for name, model in base_models.items()]
    return StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(max_iter=2000, random_state=cfg.RANDOM_STATE),
        stack_method="predict_proba",
        cv=cfg.CV_FOLDS,           # cross-validation -> reduces overfitting (per paper)
        n_jobs=-1,
        passthrough=False,
    )


def evaluate(model, X_test, y_test) -> dict:
    """Compute the paper's metrics: Accuracy, weighted Precision/Recall/F1."""
    y_pred = model.predict(X_test)
    return {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, average="weighted", zero_division=0),
        "Recall": recall_score(y_test, y_pred, average="weighted", zero_division=0),
        "F1-Score": f1_score(y_test, y_pred, average="weighted", zero_division=0),
    }
