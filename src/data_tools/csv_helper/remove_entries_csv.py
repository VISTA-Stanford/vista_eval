"""
Remove rows where the 'report' column is NaN or contains 'This exam has no report in the radiology system'.
Reads tasks from config (all_tasks.yaml), finds each task's _subsampled_no_img_report CSV,
drops those rows, and saves back to the same path.
"""

from pathlib import Path

import pandas as pd

from data_tools.utils.config_utils import load_tasks_and_base_dir, load_task_source_csv


def run(
    config_path: str,
    valid_tasks_json_path: str,
) -> None:
    tasks, base_dir = load_tasks_and_base_dir(config_path)
    task_to_source = load_task_source_csv(valid_tasks_json_path)
    base_path = Path(base_dir)

    if not base_path.exists():
        print(f"Error: base_dir not found: {base_path}")
        return

    for task_name in tasks:
        source_csv = task_to_source.get(task_name)
        if not source_csv:
            print(f"[SKIP] No task_source_csv for task '{task_name}'")
            continue

        csv_path = base_path / source_csv / f"{task_name}_subsampled_no_img_report.csv"
        if not csv_path.exists():
            print(f"[SKIP] File not found: {csv_path}")
            continue

        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            print(f"[ERR]  {csv_path.name}: {e}")
            continue

        if "report" not in df.columns:
            print(f"[SKIP] {csv_path.name}: no 'report' column")
            continue

        n_before = len(df)
        # Drop rows where report is NaN or contains the no-report placeholder
        no_report_substring = "This exam has no report in the radiology system"
        has_no_report_text = df["report"].astype(str).str.contains(
            no_report_substring, case=False, regex=False, na=True
        )
        mask_keep = df["report"].notna() & ~has_no_report_text
        df = df[mask_keep]
        n_after = len(df)
        removed = n_before - n_after

        if removed == 0:
            print(f"[OK]   {csv_path.name}: no rows to remove ({n_before} rows)")
            continue

        df.to_csv(csv_path, index=False)
        print(f"[OK]   {csv_path.name}: removed {removed} row(s) (NaN or no-report text) ({n_before} -> {n_after})")

    print("Done.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Remove rows where 'report' is NaN or contains 'This exam has no report in the radiology system'; save to same path.",
    )
    parser.add_argument(
        "--config",
        default="/home/rdcunha/vista_project/vista_eval_vlm/configs/all_tasks.yaml",
        help="Path to all_tasks.yaml",
    )
    parser.add_argument(
        "--valid-tasks",
        default="/home/rdcunha/vista_project/vista_bench/tasks/valid_tasks.json",
        help="Path to valid_tasks.json",
    )
    args = parser.parse_args()

    run(
        config_path=args.config,
        valid_tasks_json_path=args.valid_tasks,
    )
