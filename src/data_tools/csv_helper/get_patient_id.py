"""
Collect person_id from each task's subsampled CSV (per all_tasks.yaml) into one CSV
with columns 'task' and 'person_id', saved to eval/data_stats/person_id_subsampled.csv.
"""

import json
from pathlib import Path

import pandas as pd
import yaml
from data_tools.utils.config_utils import load_tasks_and_base_dir, load_task_source_csv


def main(
    config_path: str,
    valid_tasks_json_path: str,
    output_path: str | Path,
):
    tasks, base_dir = load_tasks_and_base_dir(config_path)
    task_to_source = load_task_source_csv(valid_tasks_json_path)
    base_path = Path(base_dir)

    rows = []
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
            person_id_col = next((c for c in df.columns if c.lower() == "person_id"), None)
            if person_id_col is None:
                print(f"[SKIP] No person_id column in {csv_path.name}")
                continue
            for pid in df[person_id_col].dropna().unique():
                rows.append({"task": task_name, "person_id": pid})
        except Exception as e:
            print(f"[ERR]  {csv_path.name}: {e}")

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"Wrote {len(rows)} rows to {out}")


if __name__ == "__main__":
    CONFIG_PATH = "/home/rdcunha/vista_project/vista_eval_vlm/configs/all_tasks.yaml"
    VALID_TASKS_JSON_PATH = "/home/rdcunha/vista_project/vista_bench/tasks/valid_tasks.json"
    # Output under vista_eval/eval/data_stats/
    OUTPUT_PATH = Path(__file__).resolve().parents[3] / "figures" / "data_stats" / "person_id_subsampled.csv"

    main(
        config_path=CONFIG_PATH,
        valid_tasks_json_path=VALID_TASKS_JSON_PATH,
        output_path=OUTPUT_PATH,
    )
