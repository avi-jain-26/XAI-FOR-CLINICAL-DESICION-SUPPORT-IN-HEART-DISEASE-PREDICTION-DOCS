# Explainable AI for Clinical Decision Support in Heart Disease Prediction

A small, runnable implementation of the research paper
*"Explainable AI for Clinical Decision Support in Heart Disease Prediction"*.

It reproduces the paper's three-part framework:

1. **Prediction** — 5 ML models (Logistic Regression, Random Forest, SVM,
   Gradient Boosting, XGBoost) trained individually, then combined in a
   **stacked ensemble** (the paper's proposed model).
2. **Explainability (XAI)** — **SHAP** global + per-patient explanations of the
   risk factors driving each prediction.
3. **Graph-based visualization** — a network graph of dependencies between
   clinical features (age, BMI, lifestyle, comorbidities, …).

> Runs **out of the box on synthetic data** (no download). Drop the real Kaggle
> CSV into `data/` to train on the real CDC dataset — see [Use the real dataset](#use-the-real-dataset).

---

## Tech stack

| Purpose | Library |
|---|---|
| Data handling | `pandas`, `numpy` |
| Models + **stacking ensemble** | `scikit-learn` (`StackingClassifier`), `xgboost` |
| Explainable AI | `shap` |
| Visualization | `matplotlib`, `seaborn`, `networkx` |

Language: **Python 3.9 – 3.12**.

## Project structure

```
heart-disease-xai/
├── README.md
├── requirements.txt
├── .vscode/                  # VS Code run + interpreter config (F5 to run)
├── data/                     # put the real heart_2020_cleaned.csv here (optional)
├── outputs/                  # generated metrics + plots (created on run)
└── src/
    ├── config.py             # paths, feature lists, run constants
    ├── data.py               # load real CSV OR generate synthetic data + preprocess
    ├── models.py             # 5 base models + StackingClassifier + metrics
    ├── explain.py            # SHAP explanations + networkx feature graph
    └── main.py               # entry point — runs the whole pipeline
```

---

## Run it from VS Code

### One-time setup

1. **Open the folder** in VS Code: `File ▸ Open Folder…` → select `heart-disease-xai`.
2. Install the **Python extension** (Microsoft) if you don't have it
   (Extensions panel → search "Python" → Install).
3. **Create a virtual environment.** Open the terminal (`` Ctrl+` ``) and run:

   **Windows (PowerShell):**
   ```powershell
   py -m venv .venv
   .\.venv\Scripts\Activate.ps1
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```

   **macOS / Linux:**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```

   > If PowerShell blocks the activate script, run once:
   > `Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned`

4. **Select the interpreter:** press `Ctrl+Shift+P` → *Python: Select Interpreter*
   → choose the one inside `.venv`. (`.vscode/settings.json` already points here.)

### Run

- **Easiest:** press **`F5`** (uses `.vscode/launch.json` → *Run: Heart Disease XAI pipeline*).
- **Or** open `src/main.py` and click the **▷ Run** button (top-right).
- **Or** from the terminal:
  ```powershell
  python src/main.py
  ```

The first run trains all models (synthetic default ≈ **1–3 minutes**; the SVM
inside the stacking ensemble is the slowest step).

---

## What you get (in `outputs/`)

| File | Description |
|---|---|
| `metrics.csv` | Accuracy / Precision / Recall / F1 for every model (like the paper's Tables II–III) |
| `model_comparison.png` | Bar chart: individual models vs. stacked ensemble |
| `shap_summary_beeswarm.png` | Global SHAP — how each feature pushes risk up/down |
| `shap_global_importance.png` | Global SHAP — mean feature importance ranking |
| `shap_local_patient.png` | Local SHAP — why one high-risk patient was flagged |
| `feature_dependency_graph.png` | networkx graph of feature dependencies |

A performance table is also printed in the terminal.

---

## Use the real dataset

1. Download **"Personal Key Indicators of Heart Disease"** from Kaggle:
   <https://www.kaggle.com/datasets/kamilpytlak/personal-key-indicators-of-heart-disease>
2. Save the file as `data/heart_2020_cleaned.csv`.
3. Run again — it's auto-detected and used instead of synthetic data.

> The real file has ~320k rows; `MAX_SAMPLES` in `src/config.py` subsamples it so
> the RBF SVM stays fast. Increase it (or remove the cap) for full-data runs.

## Tuning knobs (`src/config.py`)

- `MAX_SAMPLES` — rows used (raise for accuracy, lower for speed)
- `CV_FOLDS` — cross-validation folds for the stacking ensemble
- `SHAP_SAMPLE_SIZE` — test rows used for SHAP (raise for stabler plots)
- `RANDOM_STATE` — reproducibility seed

## Notes

- SHAP is computed on the ensemble's fitted **XGBoost** member (tree explainers
  are exact and fast); it's a tractable stand-in for the full stack's tree signal.
- Synthetic labels are generated from a clinically-plausible latent risk score,
  so the models learn real signal and the SHAP plots are meaningful — but
  **absolute numbers differ from the paper** until you use the real dataset.
