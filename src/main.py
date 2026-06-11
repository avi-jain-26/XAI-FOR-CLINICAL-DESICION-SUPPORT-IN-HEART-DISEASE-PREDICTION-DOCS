"""
Entry point: reproduces the paper's pipeline end-to-end.

    1. Load data (real Kaggle CSV if present, else synthetic)
    2. Preprocess (encode, scale, split)
    3. Train + evaluate the 5 individual models
    4. Train + evaluate the STACKED ENSEMBLE  (the paper's proposed model)
    5. SHAP explanations (global + local)        -- Explainable AI
    6. Graph-based feature-dependency visualization

Run:  python src/main.py
Outputs (metrics table + plots) are written to ./outputs/.
"""
from __future__ import annotations

import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

import config as cfg
import data as data_mod
import explain as xai
import models as model_mod

warnings.filterwarnings("ignore")


def main() -> None:
    cfg.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1-2. Data -------------------------------------------------------------
    df = data_mod.load_data()
    X_train, X_test, y_train, y_test, feature_names = data_mod.preprocess(df)

    # 3. Individual models --------------------------------------------------
    results = {}
    base_models = model_mod.build_base_models()
    for name, model in base_models.items():
        print(f"[train] Fitting {name} ...")
        model.fit(X_train, y_train)
        results[name] = model_mod.evaluate(model, X_test, y_test)

    # 4. Stacked ensemble (proposed model) ---------------------------------
    print("[train] Fitting Stacked Ensemble (LR + RF + SVM + GB + XGB) ...")
    stack = model_mod.build_stacking(model_mod.build_base_models())
    stack.fit(X_train, y_train)
    results["Stacked Ensemble (proposed)"] = model_mod.evaluate(stack, X_test, y_test)

    # ---- Metrics table (mirrors Tables II/III of the paper) --------------
    metrics_df = pd.DataFrame(results).T[["Accuracy", "Precision", "Recall", "F1-Score"]]
    print("\n================ MODEL PERFORMANCE ================")
    print((metrics_df * 100).round(2).to_string())
    print("===================================================\n")
    metrics_df.to_csv(cfg.OUTPUT_DIR / "metrics.csv")
    _plot_metrics(metrics_df)

    # 5. SHAP explainability ------------------------------------------------
    xai.shap_explain(stack, X_test)

    # 6. Graph-based visualization -----------------------------------------
    xai.feature_dependency_graph(X_train, y_train)

    print(f"\nDone. See results in: {cfg.OUTPUT_DIR}")


def _plot_metrics(metrics_df: pd.DataFrame) -> None:
    """Grouped bar chart comparing all models (the paper's Fig. 2 / Fig. 3)."""
    ax = metrics_df.plot(kind="bar", figsize=(11, 6), ylim=(0, 1.0), rot=20, width=0.8)
    ax.set_ylabel("Score")
    ax.set_title("Performance Comparison: Individual Models vs. Stacked Ensemble")
    ax.legend(loc="lower right")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(cfg.OUTPUT_DIR / "model_comparison.png", dpi=130, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    main()
