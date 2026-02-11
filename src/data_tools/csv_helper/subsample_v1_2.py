"""Subsample CSVs from vista_bench/bigquery_data_2_3 for v1_2.

Reads from bigquery CSVs where the file name matches task_source_csv from valid_tasks.json.
Filters rows where: split == 'test', nifti_path and note_text are both populated.
Writes _subsampled CSVs to vista_bench/v1_2/{task_source_csv}/{task_name}_subsampled.csv.
Copies all columns from the source bigquery CSV.
"""

import csv
import json
import random
import sys
from pathlib import Path

import pandas as pd

from data_tools.utils.config_utils import load_tasks_from_config

csv.field_size_limit(sys.maxsize)


def load_valid_tasks(valid_tasks_json_path: str) -> list[dict]:
    """Load valid tasks from JSON. Returns list of task dicts with task_name and task_source_csv."""
    with open(valid_tasks_json_path) as f:
        tasks = json.load(f)
    return tasks


def subsample_one_task(
    source_csv_path: Path,
    task_name: str,
    output_dir: Path,
    target_n: int | None = None,
    overwrite: bool = False,
) -> tuple[str, str, str]:
    """
    Filter and subsample one task from a bigquery CSV.
    Returns (status, path_or_name, message).
    """
    output_path = output_dir / f"{task_name}_subsampled.csv"
    if output_path.exists() and not overwrite:
        return (
            "skip",
            str(output_path),
            f"output file already exists (use overwrite=True to replace)",
        )

    try:
        df = pd.read_csv(source_csv_path, sep=None, engine="python", on_bad_lines="warn")

        # Filter by task
        task_col = next((c for c in df.columns if c.lower() == "task"), None)
        if task_col is None:
            return ("skip", str(source_csv_path), f"no 'task' column. cols={list(df.columns)}")
        df = df[df[task_col].astype(str) == str(task_name)]

        if len(df) == 0:
            return ("skip", str(output_path), f"no rows for task '{task_name}'")

        # Filter: split == 'test'
        split_col = next((c for c in df.columns if c.lower() == "split"), None)
        if split_col is None:
            return ("skip", str(source_csv_path), f"no 'split' column. cols={list(df.columns)}")
        df = df[df[split_col].astype(str).str.strip().str.lower() == "test"]

        if len(df) == 0:
            return ("skip", str(output_path), f"no test split rows for task '{task_name}'")

        # Filter: nifti_path and note_text both populated
        nifti_col = next((c for c in df.columns if c.lower() == "nifti_path"), None)
        note_col = next((c for c in df.columns if c.lower() == "note_text"), None)
        if nifti_col is None:
            return ("skip", str(source_csv_path), f"no 'nifti_path' column. cols={list(df.columns)}")
        if note_col is None:
            return ("skip", str(source_csv_path), f"no 'note_text' column. cols={list(df.columns)}")

        mask_nifti = df[nifti_col].notna() & (df[nifti_col].astype(str).str.strip() != "")
        mask_note = df[note_col].notna() & (df[note_col].astype(str).str.strip() != "")
        df = df[mask_nifti & mask_note]

        if len(df) == 0:
            return (
                "skip",
                str(output_path),
                f"no rows with nifti_path and note_text for task '{task_name}'",
            )

        # Subsample if target_n specified
        if target_n is not None and len(df) > target_n:
            random.seed(42)
            idx = random.sample(range(len(df)), target_n)
            df = df.iloc[idx].reset_index(drop=True)

        output_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        return (
            "ok",
            str(output_path),
            f"wrote {output_path.name} ({len(df)} rows)",
        )

    except Exception as e:
        return ("err", str(output_path), repr(e))


def subsample_v1_2(
    vista_bench_path: str | Path,
    valid_tasks_json_path: str | Path,
    config_path: str | Path | None = None,
    bigquery_dir: str | Path | None = None,
    output_dir: str | Path | None = None,
    target_n: int | None = 100,
    overwrite: bool = False,
):
    """
    Subsample tasks from bigquery_data_2_3 into vista_bench/v1_2.

    Args:
        vista_bench_path: Path to vista_bench (e.g. /home/rdcunha/vista_project/vista_bench)
        valid_tasks_json_path: Path to valid_tasks.json
        config_path: Path to all_tasks.yaml; only tasks in config tasks list are processed
        bigquery_dir: Directory with bigquery CSVs (default: vista_bench_path/bigquery_data_2_3)
        output_dir: Output directory (default: vista_bench_path/v1_2)
        target_n: Max rows per task (None = no subsampling)
        overwrite: If True, overwrite existing _subsampled.csv files
    """
    vista_bench_path = Path(vista_bench_path)
    if bigquery_dir is None:
        bigquery_dir = vista_bench_path / "bigquery_data_2_3"
    else:
        bigquery_dir = Path(bigquery_dir)
    if output_dir is None:
        output_dir = vista_bench_path / "v1_2"
    else:
        output_dir = Path(output_dir)

    config_tasks = None
    if config_path:
        config_tasks = load_tasks_from_config(str(config_path))
        if config_tasks:
            print(f"Loaded {len(config_tasks)} tasks from config: {sorted(config_tasks)}")
        else:
            print("Warning: No tasks found in config. Processing all tasks from valid_tasks.json.")

    tasks = load_valid_tasks(str(valid_tasks_json_path))
    print(f"Loaded {len(tasks)} tasks from {valid_tasks_json_path}")

    # Group by task_source_csv; only include tasks in config if config_path provided
    source_to_tasks: dict[str, list[str]] = {}
    for t in tasks:
        src = t.get("task_source_csv")
        name = t.get("task_name")
        if not src or not name:
            continue
        if config_tasks is not None and name not in config_tasks:
            continue
        source_to_tasks.setdefault(src, []).append(name)

    ok = skip = err = 0
    for task_source_csv, task_names in source_to_tasks.items():
        source_path = bigquery_dir / task_source_csv
        if not source_path.exists():
            # Try with .csv extension
            source_path = bigquery_dir / f"{task_source_csv}.csv"
        if not source_path.exists():
            print(f"[SKIP] Source not found: {task_source_csv}")
            skip += len(task_names)
            continue

        out_task_dir = output_dir / task_source_csv
        for task_name in task_names:
            status, path_or_name, msg = subsample_one_task(
                source_csv_path=source_path,
                task_name=task_name,
                output_dir=out_task_dir,
                target_n=target_n,
                overwrite=overwrite,
            )
            if status == "ok":
                ok += 1
                print(f"[OK]   {task_source_csv}/{task_name}: {msg}")
            elif status == "skip":
                skip += 1
                print(f"[SKIP] {task_source_csv}/{task_name}: {msg}")
            else:
                err += 1
                print(f"[ERR]  {task_source_csv}/{task_name}: {msg}")

    print(f"\nDone. OK={ok}, SKIP={skip}, ERR={err}")


if __name__ == "__main__":
    VISTA_BENCH_PATH = "/home/rdcunha/vista_project/vista_bench"
    VALID_TASKS_JSON_PATH = "/home/rdcunha/vista_project/vista_bench/tasks/valid_tasks.json"
    CONFIG_PATH = "/home/rdcunha/vista_project/vista_eval_vlm/configs/all_tasks.yaml"
    subsample_v1_2(
        vista_bench_path=VISTA_BENCH_PATH,
        valid_tasks_json_path=VALID_TASKS_JSON_PATH,
        config_path=CONFIG_PATH,
        target_n=100,
        overwrite=True,
    )
