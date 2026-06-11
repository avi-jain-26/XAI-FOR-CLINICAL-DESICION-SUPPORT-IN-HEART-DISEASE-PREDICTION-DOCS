"""
Explainable AI (SHAP) and graph-based visualization.

Part 2 of the paper's framework: explain predictions with SHAP (global feature
importance + a per-patient local explanation).
Part 3: a graph-based visualization of dependencies between clinical features.

SHAP is computed on the fitted XGBoost member of the stacked ensemble. Tree
explainers are exact and fast, and XGBoost is one of the ensemble's strongest
learners -- a tractable, reliable stand-in for attributing the ensemble's
tree-based signal (KernelExplainer on the full stack is far slower).
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")          # headless: save figures to files, never block
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

import config as cfg


# ---------------------------------------------------------------------------
# SHAP
# ---------------------------------------------------------------------------
def shap_explain(stack_model, X_test: pd.DataFrame) -> None:
    """Global (beeswarm + bar) and local (force) SHAP explanations."""
    try:
        import shap
    except Exception as e:                                   # pragma: no cover
        print(f"[xai] SHAP unavailable ({e}); skipping SHAP plots.")
        return

    # Pull the fitted XGBoost base estimator out of the stacking ensemble.
    try:
        xgb_model = stack_model.named_estimators_["XGBoost"]
    except Exception as e:
        print(f"[xai] Could not access XGBoost member ({e}); skipping SHAP.")
        return

    sample = X_test.iloc[: min(cfg.SHAP_SAMPLE_SIZE, len(X_test))].copy()

    try:
        explainer = shap.TreeExplainer(xgb_model)
        shap_values = explainer.shap_values(sample)
        expected_value = explainer.expected_value
        if isinstance(expected_value, (list, np.ndarray)):
            expected_value = np.ravel(expected_value)[0]
    except Exception as e:
        print(f"[xai] SHAP computation failed ({e}); falling back to XGBoost gain.")
        _fallback_importance(xgb_model, X_test.columns)
        return

    # Global: beeswarm summary
    plt.figure()
    shap.summary_plot(shap_values, sample, show=False)
    plt.title("SHAP Summary (global feature impact on heart-disease risk)")
    plt.tight_layout()
    _save("shap_summary_beeswarm.png")

    # Global: mean(|SHAP|) bar
    plt.figure()
    shap.summary_plot(shap_values, sample, plot_type="bar", show=False)
    plt.title("SHAP Global Feature Importance")
    plt.tight_layout()
    _save("shap_global_importance.png")

    # Local: explain the single highest-risk patient in the sample
    try:
        risk = xgb_model.predict_proba(sample)[:, 1]
        idx = int(np.argmax(risk))
        shap.force_plot(
            expected_value, shap_values[idx, :], sample.iloc[idx, :],
            matplotlib=True, show=False,
        )
        plt.title(f"Local SHAP explanation (patient #{idx}, risk={risk[idx]:.2f})")
        _save("shap_local_patient.png")
    except Exception as e:
        print(f"[xai] Local SHAP plot skipped ({e}).")

    print("[xai] SHAP plots saved to outputs/.")


def _fallback_importance(model, columns) -> None:
    """If SHAP fails, plot XGBoost's built-in gain importance instead."""
    try:
        importance = model.feature_importances_
        order = np.argsort(importance)[::-1][:20]
        plt.figure(figsize=(8, 6))
        plt.barh([columns[i] for i in order][::-1], importance[order][::-1])
        plt.xlabel("XGBoost importance (gain)")
        plt.title("Feature Importance (SHAP fallback)")
        plt.tight_layout()
        _save("feature_importance_fallback.png")
    except Exception as e:                                   # pragma: no cover
        print(f"[xai] Fallback importance also failed ({e}).")


# ---------------------------------------------------------------------------
# Graph-based visualization
# ---------------------------------------------------------------------------
def feature_dependency_graph(X: pd.DataFrame, y: pd.Series, threshold: float = 0.15) -> None:
    """Draw a graph of feature-feature dependencies (|correlation| >= threshold).

    Node size/color encodes each feature's correlation with the heart-disease
    target, surfacing the critical risk factors discussed in the paper.
    """
    corr = X.corr(numeric_only=True).fillna(0.0)
    target_corr = X.apply(lambda c: np.corrcoef(c, y)[0, 1] if c.std() > 0 else 0.0)
    target_corr = target_corr.fillna(0.0)

    G = nx.Graph()
    for col in corr.columns:
        G.add_node(col, target=abs(float(target_corr[col])))

    for i, a in enumerate(corr.columns):
        for b in corr.columns[i + 1:]:
            w = abs(float(corr.loc[a, b]))
            if w >= threshold:
                G.add_edge(a, b, weight=w)

    plt.figure(figsize=(12, 9))
    pos = nx.spring_layout(G, seed=cfg.RANDOM_STATE, k=0.9)
    node_sizes = [300 + 6000 * G.nodes[n]["target"] for n in G.nodes]
    node_colors = [G.nodes[n]["target"] for n in G.nodes]
    edge_widths = [3.0 * G[u][v]["weight"] for u, v in G.edges]

    nodes = nx.draw_networkx_nodes(
        G, pos, node_size=node_sizes, node_color=node_colors,
        cmap=plt.cm.YlOrRd, vmin=0,
    )
    nx.draw_networkx_edges(G, pos, width=edge_widths, alpha=0.4)
    nx.draw_networkx_labels(G, pos, font_size=8)
    if nodes is not None:
        cbar = plt.colorbar(nodes)
        cbar.set_label("|correlation with HeartDisease|")

    plt.title("Graph-based Feature Dependency Map\n"
              f"(edges: |feature-feature correlation| >= {threshold})")
    plt.axis("off")
    plt.tight_layout()
    _save("feature_dependency_graph.png")
    print("[xai] Feature dependency graph saved to outputs/.")


# ---------------------------------------------------------------------------
def _save(filename: str) -> None:
    cfg.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = cfg.OUTPUT_DIR / filename
    plt.savefig(path, dpi=130, bbox_inches="tight")
    plt.close()
