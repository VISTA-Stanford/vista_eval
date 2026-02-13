"""
Build a per-question metrics table (one row per question per model per task per experiment)
with model response, predicted numerical label, and ground-truth label.
For constrained decoding, map model_response directly to task mapping labels (multiclass included),
using normalized string matching (trim/case/whitespace).
Uses the same result file discovery as final_metrics.py (config tasks, models, experiments).
Output: figures/results_stats/constrained_all_model_response.csv with columns model_name, task,
experiment, index, person_id (if present in source), model_response, predicted_label, ground_truth.
"""

import json
from pathlib import Path

import pandas as pd

from results.results_analyzer import map_label_to_answer
from results.final_metrics import (
    load_config,
    collect_result_files,
    extract_experiment_from_filename,
)


EXPERIMENT_DISPLAY_NAMES = {
    "no_timeline": "image_only",
    "no_report": "image_and_timeline",
    "report": "report_and_timeline",
}


def extract_response_logprob(log_probs_str, model_response):
    """
    Extract the log probability of the model_response token from the log_probs JSON.
    This works best for single-token responses (e.g., Yes/No).
    Falls back to cumulative_logprob when available.
    """
    if pd.isna(model_response) or not str(model_response).strip():
        return None
    response_str = str(model_response).strip().lower()
    if not log_probs_str or (isinstance(log_probs_str, float) and pd.isna(log_probs_str)):
        return None
    try:
        data = json.loads(log_probs_str)
        if not data:
            return None
        for pos_data in data:
            if pos_data is None:
                continue
            for entry in pos_data:
                if isinstance(entry, dict):
                    decoded = entry.get("decoded_token", "")
                    if decoded and str(decoded).strip().lower() == response_str:
                        lp = entry.get("logprob")
                        return float(lp) if lp is not None else None
        return None
    except (json.JSONDecodeError, TypeError):
        return None


def normalize_label_text(value):
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    return " ".join(str(value).strip().lower().split())


def sorted_mapping_items(mapping):
    def sort_key(item):
        key = item[0]
        try:
            return (0, float(key))
        except (TypeError, ValueError):
            return (1, str(key))
    return sorted(mapping.items(), key=sort_key)


def map_response_to_label(response, mapping):
    """
    Map a constrained response directly to a mapping key using normalized text matching.
    Returns -1 if no output is present or if the response does not match any mapping value.
    """
    if not mapping:
        return -1
    if pd.isna(response) or not str(response).strip():
        return -1
    response_norm = normalize_label_text(response)
    if not response_norm:
        return -1

    # Primary: match mapping values (label text).
    for key, value in sorted_mapping_items(mapping):
        if normalize_label_text(value) == response_norm:
            return key

    # Fallback: allow direct key outputs (e.g., "0", "1", "-1").
    for key, _ in sorted_mapping_items(mapping):
        if normalize_label_text(key) == response_norm:
            return key
    return -1


def main(config_path=None, output_path=None):
    if config_path is None:
        config_path = Path(__file__).resolve().parents[2] / "configs" / "all_tasks.yaml"
    config_path = Path(config_path)

    results_dir, base_dir, tasks, valid_models, experiments = load_config(config_path)
    results_base = Path(results_dir)
    base_path = Path(base_dir)

    tasks_json = base_path / "tasks" / "valid_tasks.json"
    task_registry = {}
    if tasks_json.exists():
        with open(tasks_json, "r") as f:
            task_registry = {t["task_name"]: t for t in json.load(f)}

    result_files = collect_result_files(results_base, set(tasks), valid_models, experiments)
    if not result_files:
        print("No result files found matching config.")
        return

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

    out_chunks = []
    for task_name in tasks:
        mapping = task_registry.get(task_name, {}).get("mapping", {})

        task_keys = [k for k in by_task_model_exp.keys() if k[0] == task_name]
        unique_models = list({k[1] for k in task_keys})

        for model_name in unique_models:
            relevant_keys = [k for k in task_keys if k[1] == model_name]

            for key in relevant_keys:
                experiment_raw = key[2]
                experiment = EXPERIMENT_DISPLAY_NAMES.get(experiment_raw, experiment_raw)
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

                if "model_response" not in combined.columns:
                    continue

                # Direct mapping for constrained decoding output (binary and multiclass).
                combined["model_response_stripped"] = combined["model_response"].apply(
                    lambda x: str(x).strip() if pd.notna(x) and str(x).strip() else ""
                )
                combined["predicted_label"] = combined["model_response"].apply(
                    lambda x: map_response_to_label(x, mapping)
                )
                combined["ground_truth"] = combined["label"].apply(
                    lambda lbl: map_label_to_answer(lbl, mapping)
                )
                # Map ground_truth (mapped answer string) back to numerical label key
                combined["ground_truth_label"] = combined["ground_truth"].apply(
                    lambda x: map_response_to_label(x, mapping)
                )

                # Extract log prob of the model_response token from log_probs JSON
                def get_response_logprob(row):
                    lp = extract_response_logprob(
                        row.get("log_probs"),
                        row.get("model_response_stripped", row.get("model_response")),
                    )
                    if lp is not None:
                        return lp
                    return row.get("cumulative_logprob")  # fallback for single-token output

                combined["response_log_prob"] = combined.apply(get_response_logprob, axis=1)

                chunk = pd.DataFrame({
                    "model_name": model_name,
                    "task": task_name,
                    "experiment": experiment,
                    "index": combined["index"] if "index" in combined.columns else combined.index,
                    "person_id": combined["person_id"] if "person_id" in combined.columns else None,
                    "model_response": combined["model_response_stripped"],
                    "predicted_label": combined["predicted_label"],
                    "ground_truth": combined["ground_truth"],
                    "ground_truth_label": combined["ground_truth_label"],
                    "cumulative_logprob": combined["cumulative_logprob"] if "cumulative_logprob" in combined.columns else None,
                    "log_probs": combined["log_probs"] if "log_probs" in combined.columns else None,
                    "response_log_prob": combined["response_log_prob"] if "response_log_prob" in combined.columns else None,
                })
                out_chunks.append(chunk)

    if not out_chunks:
        print("No rows to write.")
        return

    out_df = pd.concat(out_chunks, ignore_index=True)
    is_minus_one = lambda x: str(x).strip() == "-1"
    # count the number of rows where predicted_label is -1 by model
    minus_one_count = out_df[out_df["predicted_label"].apply(is_minus_one)].groupby("model_name").size()
    print(f"Number of rows where predicted_label is -1 by model: {minus_one_count}")
    minus_one_count_gt = out_df[out_df["ground_truth_label"].apply(is_minus_one)].groupby("model_name").size()
    print(f"Number of rows where ground_truth_label is -1 by model: {minus_one_count_gt}")
    # Drop rows where ground_truth_label is -1
    before_drop = len(out_df)
    out_df = out_df[~out_df["ground_truth_label"].apply(is_minus_one)]
    print(f"Dropped {before_drop - len(out_df)} rows where ground_truth_label is -1")
    if output_path is None:
        output_path = Path(__file__).resolve().parents[2] / "figures" / "results_stats" / "constrained_all_model_response.csv"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output_path, index=False)
    print(f"Wrote {len(out_df)} rows to {output_path}")


if __name__ == "__main__":
    main()
