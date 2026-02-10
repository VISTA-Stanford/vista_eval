"""
Build a per-question metrics table (one row per question per model per task per experiment)
with extracted model response, predicted numerical label, and ground-truth label.
Uses the same result file discovery as final_metrics.py (config tasks, models, experiments).
Output: figures/results_stats/all_metrics.csv with columns model_name, task, experiment,
index, model_response_cleaned, predicted_label, ground_truth.
"""

import json
import re
from pathlib import Path

import pandas as pd

from results.results_analyzer import _extract_answer, map_label_to_answer
from results.final_metrics import (
    load_config,
    collect_result_files,
    extract_experiment_from_filename,
)


def strip_answer_choice_prefix(text):
    """
    Strip answer choice prefixes like 'A:', 'B:', 'C:', etc. from the beginning of text.
    Example: 'C: Surgery alone' -> 'Surgery alone'
    """
    if pd.isna(text) or not str(text).strip():
        return text
    text = str(text).strip()
    # Match pattern: letter (A-Z) followed by colon and optional whitespace at the start
    pattern = r'^[A-Z]:\s*'
    stripped = re.sub(pattern, '', text, flags=re.IGNORECASE)
    return stripped.strip() if stripped != text else text


def map_answer_to_label_key(cleaned_answer, mapping):
    """
    Map an extracted answer string back to the numerical label key (e.g. "0", "1", "-1")
    using the task's mapping. Returns None if no mapping matches (case-insensitive).
    """
    if not mapping or (cleaned_answer is not None and pd.isna(cleaned_answer)):
        return -1
    cleaned = str(cleaned_answer).strip().lower()
    if not cleaned:
        return -1
    for key, value in mapping.items():
        if value is not None and str(value).strip().lower() == cleaned:
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

                if "model_response" not in combined.columns:
                    continue

                combined["model_response_cleaned"] = combined["model_response"].apply(_extract_answer)
                # Strip answer choice prefixes (e.g., "C: Surgery alone" -> "Surgery alone")
                combined["model_response_cleaned"] = combined["model_response_cleaned"].apply(strip_answer_choice_prefix)
                combined["predicted_label"] = combined["model_response_cleaned"].apply(
                    lambda x: map_answer_to_label_key(x, mapping)
                )
                combined["ground_truth"] = combined["label"].apply(
                    lambda lbl: map_label_to_answer(lbl, mapping)
                )
                # Map ground_truth (mapped answer string) back to numerical label key
                combined["ground_truth_label"] = combined["ground_truth"].apply(
                    lambda x: map_answer_to_label_key(x, mapping)
                )

                chunk = pd.DataFrame({
                    "model_name": model_name,
                    "task": task_name,
                    "experiment": experiment,
                    "index": combined["index"] if "index" in combined.columns else combined.index,
                    "model_response_cleaned": combined["model_response_cleaned"],
                    "predicted_label": combined["predicted_label"],
                    "ground_truth": combined["ground_truth"],
                    "ground_truth_label": combined["ground_truth_label"],
                })
                out_chunks.append(chunk)

    if not out_chunks:
        print("No rows to write.")
        return

    out_df = pd.concat(out_chunks, ignore_index=True)
    # count the number of rows where predicted_label is -1 by model
    minus_one_count = out_df[out_df["predicted_label"] == -1].groupby("model_name").size()
    print(f"Number of rows where predicted_label is -1 by model: {minus_one_count}")
    if output_path is None:
        output_path = Path(__file__).resolve().parents[2] / "figures" / "results_stats" / "all_model_response.csv"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output_path, index=False)
    print(f"Wrote {len(out_df)} rows to {output_path}")


if __name__ == "__main__":
    main()
