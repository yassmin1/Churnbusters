# train.py
import argparse
import sys
import json
import mlflow
import mlflow.sklearn
import shap
import matplotlib.pyplot as plt

from itertools import product
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_recall_fscore_support
)
from mlflow.models.signature import infer_signature

# Local module
import data as data


# ------------------------------------------------------------------------------
# 🧠 Model factory (final step of the pipeline)
def build_estimator(name, C=None, n_estimators=100, max_depth=None):
    """
    - Logistic Regression uses C
    - RandomForest uses n_estimators and max_depth
    """
    if name == "logreg":
        return LogisticRegression(max_iter=1000, C=float(C), solver="liblinear")
    elif name == "rf":
        return RandomForestClassifier(
            n_estimators=int(n_estimators),
            max_depth=None if max_depth in (None, "None", "") else int(max_depth),
            random_state=42,
            class_weight="balanced",
        )
    else:
        raise ValueError("Unknown model type. Use 'logreg' or 'rf'.")


# ------------------------------------------------------------------------------
# 📈 Metrics
def log_metrics(y_true, y_pred, y_score=None):
    """Log AUC, accuracy, precision, recall, f1 to MLflow."""
    auc = roc_auc_score(y_true, y_score) if y_score is not None else roc_auc_score(y_true, y_pred)
    acc = accuracy_score(y_true, y_pred)
    prec, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary")
    mlflow.log_metrics({"auc": auc, "accuracy": acc, "precision": prec, "recall": recall, "f1": f1})
    print(f"✅ Metrics logged: AUC={auc:.3f} | Acc={acc:.3f} | F1={f1:.3f}")


# ------------------------------------------------------------------------------
# 🎯 Single training run: build Pipeline(preprocessor → estimator), fit, log
def train_once(args):
    import json
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import shap
    from sklearn.pipeline import Pipeline
    from mlflow.models.signature import infer_signature

    # =========================
    # 1) Load data & build pipe
    # =========================
    X_train_raw, X_test_raw, y_train, y_test, preprocessor, raw_feature_names = data.load_data()

    clf = build_estimator(args.estimator, args.C, args.n_estimators, args.max_depth)
    pipe = Pipeline(steps=[("prep", preprocessor), ("clf", clf)])
    pipe.fit(X_train_raw, y_train)

    # =========================
    # 2) Predict & log metrics
    # =========================
    y_pred = pipe.predict(X_test_raw)
    y_score = pipe.predict_proba(X_test_raw)[:, 1] if hasattr(pipe, "predict_proba") else None
    mlflow.set_tag("model_family", args.estimator)

    params = {"estimator": args.estimator}
    if args.estimator == "logreg":
        params["C"] = args.C
    elif args.estimator == "rf":
        params["n_estimators"] = args.n_estimators
        params["max_depth"] = args.max_depth
    mlflow.log_params(params)

    log_metrics(y_test, y_pred, y_score)

    # =========================================
    # 3) Log model with robust schema/signature
    # =========================================
    # Use a reasonably sized input_example to capture realistic dtypes
    input_example = X_train_raw.head(200).copy()

    # --> IMPORTANT: Upcast int columns to float64 to avoid MLflow schema issues when NaNs appear
    int_cols = input_example.select_dtypes(include=["int", "int32", "int64"]).columns.tolist()
    if int_cols:
        input_example[int_cols] = input_example[int_cols].astype("float64")

    # Signature from the realistic sample
    signature = infer_signature(input_example, pipe.predict(input_example))

    # Use artifact_path for broad MLflow compatibility (2.x/3.x)
    mlflow.sklearn.log_model(
        sk_model=pipe,
        artifact_path="model",
        signature=signature,
        input_example=input_example,
        # registered_model_name="ChurnModel",  # uncomment to register in one step
    )

    # ==================================================
    # 4) Resolve & log transformed feature names (SHAP)
    # ==================================================
    transformed_feature_names = None
    try:
        prep = pipe.named_steps["prep"]
        if hasattr(prep, "get_feature_names_out"):
            transformed_feature_names = prep.get_feature_names_out().tolist()
        else:
            # Fallback to raw names (not ideal if encoder expands features)
            transformed_feature_names = list(raw_feature_names)

        mlflow.log_dict({"feature_names": transformed_feature_names},
                        "columns_after_preprocessing.json")
    except Exception as e:
        print(f"⚠️ Could not resolve transformed feature names: {e}")


    # ============================
    # 5) SHAP: global bar plot
    # ============================
    import numpy as np
    import pandas as pd
    import shap
    import matplotlib.pyplot as plt
    from shap import Explanation

    def _to_df(X, cols):
        """Convert transformed matrix to a DataFrame with safe column names."""
        if hasattr(X, "toarray"):
            X = X.toarray()
        X = np.asarray(X)
        if cols is None or len(cols) != X.shape[1]:
            cols = [f"f{i}" for i in range(X.shape[1])]
        return pd.DataFrame(X, columns=cols)

    def _pick_positive_class_index(n_classes):
        return 1 if n_classes == 2 else 0

    try:
        # Transform inputs and wrap with names
        X_train_tr = pipe.named_steps["prep"].transform(X_train_raw)
        X_test_tr  = pipe.named_steps["prep"].transform(X_test_raw)

        X_train_df = _to_df(X_train_tr, transformed_feature_names)
        X_test_df  = _to_df(X_test_tr,  transformed_feature_names)

        # Keep SHAP fast & stable
        bg = X_train_df.sample(min(200, len(X_train_df)), random_state=42)
        X_plot = X_test_df.head(min(500, len(X_test_df)))

        est = pipe.named_steps["clf"]
        est_name = est.__class__.__name__.lower()
        is_tree_like = any(k in est_name for k in ["forest", "tree", "xgb", "lgbm", "catboost", "gradientboost"])

        # Build explainer
        if is_tree_like:
            explainer = shap.TreeExplainer(est, data=bg, model_output="probability")
            shap_values = explainer(X_plot, check_additivity=False)
        else:
            explainer = shap.Explainer(est, bg)
            shap_values = explainer(X_plot)

        # ---- Normalize to 2D Explanation for bar plotting ----
        sv_to_plot = None
        if isinstance(shap_values, list):
            # list per class → pick positive class or the class with largest mean |SHAP|
            pos_idx = _pick_positive_class_index(len(shap_values))
            if len(shap_values) > 2:
                means = [np.abs(sv).mean() for sv in shap_values]
                pos_idx = int(np.argmax(means))
            sv = shap_values[pos_idx]
            sv_to_plot = Explanation(values=sv, data=X_plot.values, feature_names=list(X_plot.columns))

        elif hasattr(shap_values, "values"):  # Explanation
            vals = shap_values.values
            if getattr(vals, "ndim", 2) == 3:  # (n_samples, n_features, n_classes)
                n_classes = vals.shape[2]
                pos_idx = _pick_positive_class_index(n_classes)
                if n_classes > 2:
                    per_class = np.abs(vals).mean(axis=(0,1))
                    pos_idx = int(np.argmax(per_class))
                sv_to_plot = Explanation(
                    values=vals[:, :, pos_idx],
                    base_values=(shap_values.base_values[:, pos_idx]
                                if getattr(shap_values.base_values, "ndim", 1) == 2
                                else shap_values.base_values),
                    data=shap_values.data,
                    feature_names=shap_values.feature_names,
                )
            else:
                sv_to_plot = shap_values  # already 2D

        else:
            # raw numpy fallback
            arr = np.asarray(shap_values)
            if arr.ndim == 3:
                n_classes = arr.shape[2]
                pos_idx = _pick_positive_class_index(n_classes)
                if n_classes > 2:
                    per_class = np.abs(arr).mean(axis=(0,1))
                    pos_idx = int(np.argmax(per_class))
                arr = arr[:, :, pos_idx]
            sv_to_plot = Explanation(values=arr, data=X_plot.values, feature_names=list(X_plot.columns))

        # ---- Global importance bar plot ----
        plt.figure()
        shap.plots.bar(sv_to_plot, max_display=min(10, X_plot.shape[1]), show=False)  # change 30 to your top-k
        plt.tight_layout()
        plt.savefig("shap_bar.png", dpi=160)
        plt.close()
        mlflow.log_artifact("shap_bar.png")

    except Exception as e:
        print(f"⚠️ SHAP bar plotting skipped: {e}")


# ------------------------------------------------------------------------------
# 🔁 Grid search over both models
def grid_search():
    """
    - Logistic Regression: sweep C
    - Random Forest: sweep n_estimators × max_depth
    Uses nested runs so everything is grouped under one parent in the MLflow UI.
    """
    logreg_Cs = [0.01, 0.1, 1.0, 10.0]
    rf_n_estimators = [50, 100, 200, 300]
    rf_max_depths = [None, 5, 10]  # None = unlimited depth

    with mlflow.start_run(run_name="grid_search_all_models", nested=True):
        # LogReg sweep
        for C in logreg_Cs:
            with mlflow.start_run(run_name=f"logreg_C={C}", nested=True):
                mlflow.set_tag("model_family", "logreg")
                args = argparse.Namespace(estimator="logreg", C=C, n_estimators=None, max_depth=None, do_tuning=False)
                train_once(args)

        # RF sweep
        for n, d in product(rf_n_estimators, rf_max_depths):
            run_name = f"rf_n={n}_depth={d if d is not None else 'None'}"
            with mlflow.start_run(run_name=run_name, nested=True):
                mlflow.set_tag("model_family", "rf")
                args = argparse.Namespace(estimator="rf", C=None, n_estimators=n, max_depth=d, do_tuning=False)
                train_once(args)


# ------------------------------------------------------------------------------
# 🚀 CLI entry
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--estimator", type=str, default="logreg", help="Model type: 'logreg' or 'rf'")
    parser.add_argument("--C", type=float, default=1.0, help="C for Logistic Regression")
    parser.add_argument("--n_estimators", type=int, default=100, help="Trees for RandomForest")
    parser.add_argument("--max_depth", type=str, default=None, help="Max depth for RF (int or None)")
    parser.add_argument("--do_tuning", action="store_true", help="Run grid search over both models")

    # Notebook/VSCode interactive safety
    if hasattr(sys, "ps1") or "ipykernel" in sys.modules:
        args = parser.parse_args(args=[])
    else:
        args = parser.parse_args()

    # Local file store (Windows/VSCode friendly)
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("Telco-Churn")

    with mlflow.start_run():
        if args.do_tuning:
            grid_search()
        else:
            train_once(args)


if __name__ == "__main__":
    main()

