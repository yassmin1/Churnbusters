# train.py
import argparse
import sys
import json
import joblib
import mlflow
import mlflow.sklearn
import shap
import matplotlib.pyplot as plt

from itertools import product
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_recall_fscore_support
)

# 👇 your local module; expected to return:
# X_train, X_test, y_train, y_test, preprocessor, feature_names
import old_files.data as data


# ------------------------------------------------------------------------------
# 🧠 Model factory
def build_model(name, C=None, n_estimators=100, max_depth=None):
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
    """
    Logs AUC, accuracy, precision, recall, f1.
    - AUC uses probabilities if provided; falls back to hard labels if not.
    """
    # Prefer scores/probabilities for AUC
    if y_score is not None:
        auc = roc_auc_score(y_true, y_score)
    else:
        # Fallback (less informative than prob-based AUC)
        auc = roc_auc_score(y_true, y_pred)

    acc = accuracy_score(y_true, y_pred)
    prec, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary"
    )

    metrics = {"auc": auc, "accuracy": acc, "precision": prec, "recall": recall, "f1": f1}
    mlflow.log_metrics(metrics)
    print(f"✅ Metrics logged: AUC={auc:.3f} | Acc={acc:.3f} | F1={f1:.3f}")


# ------------------------------------------------------------------------------
# 🎯 Single training run
def train_once(args):
    from mlflow.models.signature import infer_signature
    # Load prepared features + fitted preprocessor from your data module
    X_train, X_test, y_train, y_test, preprocessor, feature_names = data.load_data()
    

    # Build & fit estimator ON TRANSFORMED FEATURES
    model = build_model(args.estimator, args.C, args.n_estimators, args.max_depth)
    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)
    y_score = None
    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X_test)[:, 1]
    #
    # Build a small input example and signature (using your transformed X)
    input_example = X_train.head(2)
# If you have predict_proba, signature can be inferred from predict; both are fine
    signature = infer_signature(input_example, model.predict(input_example))    

    # ---- MLflow logging (only relevant params) ----
    mlflow.set_tag("model_family", args.estimator)
    params = {"estimator": args.estimator}
    if args.estimator == "logreg":
        params["C"] = args.C
    elif args.estimator == "rf":
        params["n_estimators"] = args.n_estimators
        params["max_depth"] = args.max_depth
    mlflow.log_params(params)

    # Metrics
    log_metrics(y_test, y_pred, y_score)

    # Model (this logs the trained estimator on transformed features)
    mlflow.sklearn.log_model(model, name="model",  signature=signature, input_example=input_example,)

    # --- ✅ RECOMMENDED: log the preprocessor + columns as artifacts (files) ---
    # IMPORTANT: log_artifact needs a *file path*, not a Python object.
    joblib.dump(preprocessor, "preprocessor.pkl")
    mlflow.log_artifact("preprocessor.pkl", artifact_path="preprocessor")

    with open("columns.json", "w") as f:
        json.dump(list(feature_names), f)
    mlflow.log_artifact("columns.json", artifact_path="preprocessor")

    # SHAP quick-look (limit to 100 rows for speed)
    try:
        explainer = shap.Explainer(model, X_train, feature_names=X_train.columns)
        shap_values = explainer(X_test.iloc[:100])
        shap.plots.waterfall(shap_values[0], show=False)
        plt.tight_layout()
        plt.savefig("shap_waterfall.png", dpi=120)
        plt.close()
        mlflow.log_artifact("shap_waterfall.png")
    except Exception as e:
        # SHAP can fail for some combos; don't kill the run
        print(f"⚠️ SHAP plotting skipped: {e}")


# ------------------------------------------------------------------------------
# 🔁 Grid search over both models
def grid_search():
    """
    - Logistic Regression: sweep C
    - Random Forest: sweep n_estimators × max_depth
    """
    logreg_Cs = [0.01, 0.1, 1.0, 10.0]
    rf_n_estimators = [50, 100, 200, 300]
    rf_max_depths = [None, 5, 10]  # None = unlimited depth

    with mlflow.start_run(run_name="grid_search_all_models", nested=True):
        # LogReg sweep
        for C in logreg_Cs:
            with mlflow.start_run(run_name=f"logreg_C={C}", nested=True):
                mlflow.set_tag("model_family", "logreg")
                args = argparse.Namespace(
                    estimator="logreg", C=C, n_estimators=None, max_depth=None, do_tuning=False
                )
                train_once(args)

        # RF sweep (Cartesian)
        for n, d in product(rf_n_estimators, rf_max_depths):
            run_name = f"rf_n={n}_depth={d if d is not None else 'None'}"
            with mlflow.start_run(run_name=run_name, nested=True):
                mlflow.set_tag("model_family", "rf")
                args = argparse.Namespace(
                    estimator="rf", C=None, n_estimators=n, max_depth=d, do_tuning=False
                )
                train_once(args)


# ------------------------------------------------------------------------------
# 🚀 CLI entry
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--estimator", type=str, default="logreg",
                        help="Model type: 'logreg' or 'rf'")
    parser.add_argument("--C", type=float, default=1.0,
                        help="Regularization strength (for Logistic Regression)")
    parser.add_argument("--n_estimators", type=int, default=100,
                        help="Number of trees (for RandomForest)")
    parser.add_argument("--max_depth", type=str, default=None,
                        help="Max depth for RandomForest (int or None). Default: None")
    parser.add_argument("--do_tuning", action="store_true",
                        help="Run grid search instead of single model")

    # Notebook/VSCode interactive safety
    if hasattr(sys, 'ps1') or 'ipykernel' in sys.modules:
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
