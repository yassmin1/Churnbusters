# register_best_model.py
"""
Pick the best run from an experiment by a metric, register its model, and (optionally)
transition the new model version to a stage.

Example:
  python register_best_model.py ^
    --experiment-name "Telco-Churn" ^
    --metric auc --higher-is-better ^
    --model-name ChurnModel ^
    --artifact-path model ^
    --stage Production
"""

import argparse
import sys
import time
import mlflow
from mlflow.tracking import MlflowClient
import pandas as pd
import os
import json
from mlflow.tracking import MlflowClient


def get_experiment_id(name: str | None, exp_id: str | None) -> str:
    """Resolve experiment id from name or return provided id; exit with message if not found."""
    if exp_id:
        return exp_id
    if name:
        exp = MlflowClient().get_experiment_by_name(name)
        if not exp:
            print(f"[ERROR] No experiment named '{name}'", file=sys.stderr)
            sys.exit(2)
        return exp.experiment_id
    print("[ERROR] Provide --experiment-name or --experiment-id", file=sys.stderr)
    sys.exit(2)


def find_metric_col(df: pd.DataFrame, metric: str) -> str:
    """
    Accepts 'auc' or 'metrics.auc', returns the exact column name in df (case-insensitive).
    Shows available metric columns when not found.
    """
    wanted = metric.lower()
    if not wanted.startswith("metrics."):
        wanted = "metrics." + wanted

    # exact, case-insensitive match
    for col in df.columns:
        if col.lower() == wanted:
            return col

    # fallback: suffix match (e.g., user passes 'auc', column is 'metrics.AUC')
    suffix = wanted.split(".", 1)[1]
    candidates = [
        c for c in df.columns
        if c.lower().startswith("metrics.") and c.lower().endswith("." + suffix)
    ]
    if len(candidates) == 1:
        return candidates[0]

    metric_cols = [c for c in df.columns if c.startswith("metrics.")]
    print(f"[ERROR] Metric '{metric}' not found.", file=sys.stderr)
    print("Available metric columns:", metric_cols or "(none)", file=sys.stderr)
    sys.exit(2)


def wait_for_model_version_ready(client: MlflowClient, name: str, version: str | int,
                                 timeout_sec: int = 180, poll_interval: float = 1.0):
    """Poll the model registry until the model version is READY or FAILED."""
    t0 = time.time()
    last_status = None
    while time.time() - t0 < timeout_sec:
        mv = client.get_model_version(name=name, version=str(version))
        if mv.status != last_status:
            print(f"  ModelVersion status: {mv.status}")
            last_status = mv.status
        if mv.status == "READY":
            return mv
        if mv.status == "FAILED":
            raise RuntimeError(f"Model version {name} v{version} failed to register.")
        time.sleep(poll_interval)
    raise TimeoutError(f"Timed out after {timeout_sec}s waiting for {name} v{version} to be READY.")


def main():
    
    p = argparse.ArgumentParser(
        description="Register best run's model and optionally transition stage."
    )

    # One of these identifies the experiment. Not required on CLI because we set defaults.
    g = p.add_mutually_exclusive_group(required=False)
    g.add_argument(
        "--experiment-name",
        default="Telco-Churn",
        help="Experiment name to search (default: Telco-Churn)"
    )
    g.add_argument(
        "--experiment-id",
        default=None,
        help="Experiment ID to search (default: None; uses --experiment-name if not provided)"
    )

    p.add_argument(
        "--metric",
        default="auc",
        help="Metric to sort by, e.g. auc, rmse, f1 or metrics.auc (default: auc)"
    )

    p.add_argument(
        "--higher-is-better",
        action="store_true",
        default=True,
        help="Set when larger metric values are better (e.g., AUC, accuracy). Default: True"
    )

    p.add_argument(
        "--model-name",
        default="ChurnModel",
        help="Registered model name to create/update (default: ChurnModel)"
    )

    p.add_argument(
        "--artifact-path",
        default="model",
        help="Artifact path used when logging the model in the run (default: model)"
    )

    p.add_argument(
        "--stage",
        default="Staging",
        help="Optional stage to transition to: Staging | Production | Archived (default: Staging)"
    )

    p.add_argument(
        "--only-finished",
        action="store_true",
        default=True,
        help="Ignore runs that did not finish successfully (default: True)"
    )

    p.add_argument(
        "--tracking-uri",
        default=os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns"),
        help="MLflow tracking URI (default: env MLFLOW_TRACKING_URI or file:./mlruns)"
    )
    
    p.add_argument(
    "--alias",
    default="production",   # choose your default; or "" to require explicit
    help="Registered model alias to set for this version (e.g., 'production', 'staging'). Empty to skip."
)

    args = p.parse_args()

    if args.tracking_uri:
        mlflow.set_tracking_uri(args.tracking_uri)

    exp_id = get_experiment_id(args.experiment_name, args.experiment_id)

    # Pull runs and (optionally) filter to FINISHED
    runs_df = mlflow.search_runs(experiment_ids=[exp_id])
    if runs_df.empty:
        print("[ERROR] No runs found in the experiment.", file=sys.stderr)
        sys.exit(2)

    if args.only_finished:
        if "status" in runs_df.columns:
            runs_df = runs_df[runs_df["status"] == "FINISHED"]
        else:
            # Older MLflow might not include status; ignore filter gracefully
            pass

    metric_col = find_metric_col(runs_df, args.metric)

    # Drop runs where metric is missing (NaN)
    runs_df = runs_df.dropna(subset=[metric_col])
    if runs_df.empty:
        print(f"[ERROR] No runs with metric '{metric_col}'.", file=sys.stderr)
        sys.exit(2)

    ascending = not args.higher_is_better
    best = runs_df.sort_values(metric_col, ascending=ascending).iloc[0]
    run_id = best["run_id"]
    metric_val = best[metric_col]

    model_uri = f"runs:/{run_id}/{args.artifact_path}"

    print(f"Best run_id: {run_id}")
    print(f"  Using metric {metric_col} = {metric_val}")
    print(f"Registering model from URI: {model_uri}  →  name: {args.model_name}")

    # Register model (returns a ModelVersion with version number, but may be in PENDING_REGISTRATION)
    mv = mlflow.register_model(model_uri=model_uri, name=args.model_name)
    version = mv.version
    print(f"Created model version: {version}")

    # Wait until READY before transitioning stage


    client = MlflowClient()

    # -- Figure out the real model URI that was logged in this run --
    run = client.get_run(run_id)

    model_uri = None
    hist_tag = run.data.tags.get("mlflow.log-model.history")
    if hist_tag:
        try:
            history = json.loads(hist_tag)  # list of dicts
            # Prefer the last logged model in this run
            entry = history[-1]
            # MLflow 3.x: if logged with `name=`, we have a model_uuid
            if "model_uuid" in entry:
                model_uri = f"models:/{entry['model_uuid']}"
            # Fallback: if we logged with artifact_path, match it; else take first available
            if model_uri is None:
                target_path = args.artifact_path or "model"
                # Try to find a matching artifact path
                match = next((e for e in reversed(history) if e.get("artifact_path") == target_path), None)
                if match and "artifact_path" in match:
                    model_uri = f"runs:/{run_id}/{match['artifact_path']}"
                elif "artifact_path" in entry:
                    model_uri = f"runs:/{run_id}/{entry['artifact_path']}"
        except Exception as e:
            print(f"[WARN] Could not parse log-model history tag: {e}")
    # Final fallback if tag missing
    if model_uri is None:
        model_uri = f"runs:/{run_id}/{args.artifact_path}"

    print(f"Registering: {model_uri}  →  {args.model_name}")
    mv = mlflow.register_model(model_uri=model_uri, name=args.model_name)
    version = mv.version
    print(f"Created model version: {version}")

    # Wait until READY (your existing wait helper)
    wait_for_model_version_ready(client, args.model_name, version)

    # ✅ Use aliases instead of stages (avoids YAML error & deprecation)
    # Default alias name if user passes --stage for backward-compat
    alias = getattr(args, "alias", None)
    if not alias and args.stage:
        # map common stages to aliases
        mapping = {"Production": "production", "Staging": "staging", "Archived": "archived", "": None}
        alias = mapping.get(args.stage, args.stage.lower())

    if alias:
        client.set_registered_model_alias(args.model_name, alias, version)
        print(f"Alias set: {args.model_name}@{alias} → v{version}")
    else:
        print("No alias requested. (Tip: pass --alias production)")


if __name__ == "__main__":
    main()
""" 


# python register_best_model.py   --experiment-name "Telco-Churn"   --metric auc --higher-is-better   --model-name ChurnModel   --artifact-path model   --stage Production   --only-finished   --tracking-uri "file:./mlruns"
#--artifact-path must match what you used when logging (e.g., mlflow.sklearn.log_model(model, "model")).
"""

#mlflow models serve -m "models:/ChurnModel@production" -p 5001 --no-conda