"""
Read result CSVs (same discovery as ct_experiment_plot) from results_dir in config,
and output figures/results_stats/results.csv with columns: task, model_name, experiment, AUROC, AUPRC, accuracy.
For each (task, model), the experiment with the best accuracy is selected and metrics are computed from that experiment.
"""

import json
import re
from pathlib import Path

import pandas as pd
import yaml
from sklearn.metrics import roc_auc_score, average_precision_score

# Ensure these imports exist in your environment
from results.results_analyzer import _extract_answer, is_answer_correct, map_label_to_answer


def extract_experiment_from_filename(filename):
    """Extract experiment name from filename like task_name_results_experiment.csv"""
    match = re.search(r'_results_(.+)\.csv$', filename)
    if match:
        return match.group(1)
    return "default"


def load_config(config_path):
    """Load results_dir, tasks, models, experiments from all_tasks.yaml."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    paths = config.get("paths", {})
    results_dir = paths.get("results_dir") or "/home/dcunhrya/results"
    base_dir = paths.get("base_dir") or "/home/dcunhrya/vista_bench"
    tasks = list(config.get("tasks", []))
    models_list = config.get("models", [])
    valid_models = set()
    for model in models_list or []:
        if isinstance(model, dict) and "name" in model:
            name = model["name"]
            file_name = name.split("/")[-1].replace("/", "_")
            valid_models.add(file_name)
            valid_models.add(name.replace("/", "_"))
            if "/" in name:
                valid_models.add(name.split("/")[-1])
    experiments = set(config.get("experiments", []))
    return results_dir, base_dir, tasks, valid_models, experiments


def calculate_accuracy_like_plot(df, mapping):
    """
    Same accuracy logic as ct_experiment_plot: is_answer_correct on every row (all rows, no filtering).
    Returns accuracy as percentage, or None if cannot compute.
    """
    if not mapping or "label" not in df.columns or "model_response" not in df.columns:
        return None
    df = df.copy()
    df["mapped_label"] = df["label"].apply(lambda lbl: map_label_to_answer(lbl, mapping))
    df["is_correct"] = df.apply(
        lambda x: is_answer_correct(x["model_response"], x["mapped_label"]),
        axis=1,
    )
    return df["is_correct"].mean() * 100


def binary_labels_and_scores(df, mapping):
    """
    Build y_true (binary) and y_score (prediction score 0/1 from extracted answer).
    Excludes rows where label is -1 or maps to "Insufficient follow-up or missing data".
    
    FIX: Now includes ALL valid ground truth rows. 
         If model output matches 'positive' -> score 1.0
         Anything else (negative match, garbage, ambiguous) -> score 0.0
    """
    if not mapping or "label" not in df.columns or "model_response" not in df.columns:
        return None, None
    df = df.copy()
    df["mapped_label"] = df["label"].apply(lambda lbl: map_label_to_answer(lbl, mapping))
    
    # 1. Filter Ground Truth: Keep only rows with valid labels
    insufficient = "Insufficient follow-up or missing data"
    valid = (
        df["label"].notna()
        & (df["label"].astype(str).str.strip() != "-1")
        & (df["mapped_label"] != insufficient)
        & (df["mapped_label"].notna())
    )
    df = df.loc[valid].copy()
    
    if df.empty:
        return None, None

    pos_str = mapping.get("1")
    # We don't strictly need neg_str for binary score calculation if we treat "not pos" as 0
    if not pos_str:
        return None, None

    df["cleaned_response"] = df["model_response"].apply(_extract_answer)
    
    # 2. Build y_true based on mapped_label
    # 1 if Label is positive, 0 otherwise
    y_true = (df["mapped_label"].str.strip().str.lower() == pos_str.strip().lower()).astype(int).values

    # 3. Build y_score based on prediction
    # If the response text matches the positive string -> 1.0
    # Otherwise -> 0.0 (This correctly penalizes garbage/ambiguous answers as failures to detect positive)
    # If you have a 'probability' column in your CSV, you should use that here instead.
    y_score = (df["cleaned_response"].str.strip().str.lower() == pos_str.strip().lower()).astype(float).values

    return y_true, y_score


def compute_metrics(df, mapping):
    """
    Compute AUROC, AUPRC (%), and accuracy (%).
    Accuracy: same as ct_experiment_plot (is_answer_correct on all rows).
    AUROC/AUPRC: on rows with valid label (exclude -1/insufficient).
    """
    # Accuracy: match ct_experiment_plot exactly (all rows, is_answer_correct)
    accuracy = calculate_accuracy_like_plot(df, mapping)
    if accuracy is None:
        return None, None, None

    # AUROC/AUPRC
    y_true, y_score = binary_labels_and_scores(df, mapping)
    
    # If filtering resulted in empty data
    if y_true is None or len(y_true) == 0:
        return float("nan"), float("nan"), accuracy

    try:
        # AUROC requires at least one sample of each class (0 and 1) in y_true
        if len(set(y_true)) < 2:
            auroc = float("nan")
            auprc = float("nan")
        else:
            auroc = roc_auc_score(y_true, y_score)
            auprc = average_precision_score(y_true, y_score)
    except Exception:
        auroc = float("nan")
        auprc = float("nan")
        
    return auroc, auprc, accuracy


def collect_result_files(results_base, valid_tasks, valid_models, valid_experiments):
    """Discover and filter result files like ct_experiment_plot."""
    results_base = Path(results_base)
    # Use rglob to find all csvs, but be careful with permissions/depth if needed
    all_files = list(results_base.rglob("*_results_*.csv"))
    result_files = []
    for file_path in all_files:
        try:
            rel = file_path.relative_to(results_base).parts
        except ValueError:
            continue
        # Expect structure: .../results/source/task/model/filename.csv
        if len(rel) < 4:
            continue
        
        # Adjust indices based on your folder structure depth
        # Assuming: /results_dir/source_folder/task_name/model_name/filename
        task_name, model_name, filename = rel[1], rel[2], rel[-1]
        
        experiment = extract_experiment_from_filename(filename)
        
        if valid_tasks and task_name not in valid_tasks:
            continue
        if valid_models:
            model_ok = model_name in valid_models or any(
                model_name == m or model_name.endswith("_" + m) or model_name.endswith("/" + m)
                for m in valid_models
            )
            if not model_ok:
                continue
        if valid_experiments and experiment not in valid_experiments:
            continue
        result_files.append(file_path)
    return result_files


def main(config_path=None, output_path=None):
    if config_path is None:
        config_path = Path(__file__).resolve().parents[2] / "configs" / "all_tasks.yaml"
    config_path = Path(config_path)
    
    results_dir, base_dir, tasks, valid_models, experiments = load_config(config_path)
    results_base = Path(results_dir)
    base_path = Path(base_dir)

    # Task registry for mappings
    tasks_json = base_path / "tasks" / "valid_tasks.json"
    task_registry = {}
    if tasks_json.exists():
        with open(tasks_json, "r") as f:
            task_registry = {t["task_name"]: t for t in json.load(f)}

    result_files = collect_result_files(results_base, set(tasks), valid_models, experiments)
    if not result_files:
        print("No result files found matching config.")
        return

    # Group files by (task_name, model_name, experiment)
    by_task_model_exp = {}
    for file_path in result_files:
        try:
            rel = file_path.relative_to(results_base).parts
            task_name, model_name, filename = rel[1], rel[2], rel[-1]
            experiment = extract_experiment_from_filename(filename)
            key = (task_name, model_name, experiment)
            if key not in by_task_model_exp:
                by_task_model_exp[key] = []
            by_task_model_exp[key].append(file_path)
        except Exception as e:
            print(f"Skipping file {file_path}: {e}")

    rows = []
    for task_name in tasks:
        mapping = task_registry.get(task_name, {}).get("mapping", {})
        
        # Find all models available for this task
        task_keys = [k for k in by_task_model_exp.keys() if k[0] == task_name]
        unique_models = list({k[1] for k in task_keys})
        
        for model_name in unique_models:
            # For this (task, model), get accuracy for each experiment and pick best
            best_accuracy = None
            best_experiment = None
            best_df = None
            
            # Iterate over all experiments for this specific task & model
            relevant_keys = [k for k in task_keys if k[1] == model_name]
            
            for key in relevant_keys:
                experiment = key[2]
                files = by_task_model_exp[key]
                dfs = []
                for fp in files:
                    try:
                        df = pd.read_csv(fp)
                        if not df.empty:
                            dfs.append(df)
                    except Exception as e:
                        print(f"  Error reading {fp.name}: {e}")
                
                if not dfs:
                    continue
                    
                combined = pd.concat(dfs, ignore_index=True)
                if "index" in combined.columns:
                    combined = combined.drop_duplicates(subset=["index"], keep="first")
                
                # Compute provisional metrics just to get accuracy
                _, _, acc = compute_metrics(combined, mapping)
                
                if acc is None:
                    continue
                    
                if best_accuracy is None or acc > best_accuracy:
                    best_accuracy = acc
                    best_experiment = experiment
                    best_df = combined
            
            # After finding best experiment, save row
            if best_df is None or best_experiment is None:
                continue
            
            auroc, auprc, accuracy = compute_metrics(best_df, mapping)
            
            # Even if auroc is NaN (e.g. only 1 class in GT), we might want to save the Accuracy
            rows.append({
                "task": task_name,
                "model_name": model_name,
                "experiment": best_experiment,
                "AUROC": auroc,
                "AUPRC": auprc,
                "accuracy": accuracy,
            })

    if not rows:
        print("No metrics computed.")
        return

    out_df = pd.DataFrame(rows)
    if output_path is None:
        output_path = Path(__file__).resolve().parents[2] / "figures" / "results_stats" / "results.csv"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output_path, index=False)
    print(f"Wrote {len(rows)} rows to {output_path}")


if __name__ == "__main__":
    main()