"""
Remove STANFORD_NOTE/imaging from subsampled CSVs for tasks in all_tasks.yaml.

- Drops entire rows where patient_string contains 'STANFORD_NOTE/imaging'.
- For kept rows, removes from patient_string any timeline line matching
  "[datetime] | STANFORD_NOTE/imaging" (and recomputes patient_string_character_len).

Only processes tasks listed in vista_eval/configs/all_tasks.yaml and only
subsampled CSVs when config has subsample: true.
"""

import re
import sys
import warnings
from pathlib import Path

import pandas as pd
import yaml

from data_tools.utils.config_utils import load_task_source_csv

# Pattern to match a timeline entry starting with a datetime and containing imaging markers.
# It matches from [datetime] | marker ... until the next [datetime] or end of string.
# Markers: STANFORD_NOTE/imaging or STANFORD_NOTE/imaging-non-reportable
# Note: We use a non-greedy .*? to stop at the first lookahead match.
# We also use \n? to catch a trailing newline if it exists.
# TIMELINE_IMAGING_PATTERN = re.compile(
#     r"\[\d{4}-\d{2}-\d{2} \d{2}:\d{2}\] \| STANFORD_NOTE/imaging(?:-non-reportable)?.*?(?=\[\d{4}-\d{2}-\d{2} \d{2}:\d{2}\]|$)\n?",
#     re.DOTALL,
# )
# IMAGING_MARKERS = ["STANFORD_NOTE/imaging", "STANFORD_NOTE/imaging-non-reportable"]


def load_tasks_and_config(config_path: str):
    """Load task list and paths from YAML config."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    tasks = config.get("tasks", [])
    base_dir = config.get("paths", {}).get("base_dir", "")
    subsample = config.get("subsample", False)
    return list(tasks), base_dir, subsample


def remove_imaging_lines_from_timeline(patient_string: str) -> str:
    """
    Splits the timeline by newline, removes any line containing 
    'STANFORD_NOTE/imaging', and rejoins the remaining lines.
    """
    # Safety check for NaN or non-string inputs (e.g. if column has floats/NaNs)
    # if pd.isna(patient_string) or not isinstance(patient_string, str):
    #     return patient_string

    # 1. Split the massive string into a list of individual event lines
    lines = patient_string.split('\n')

    # 2. Keep only the lines that DO NOT contain your target string
    filtered_lines = [
        line for line in lines 
        if 'STANFORD_NOTE' not in line
    ]

    # 3. Join the valid lines back together with newlines
    return '\n'.join(filtered_lines)


def process_csv(csv_path: Path, output_path: Path) -> tuple[str, str, str]:
    """
    Read subsampled CSV, drop rows containing STANFORD_NOTE/imaging, clean patient_string, write new CSV.

    Returns (status, path_str, message).
    """
    try:
        with warnings.catch_warnings():
            # Suppress CSV parser warnings from malformed quoting in long patient_string fields
            warnings.simplefilter("ignore")
            # Explicitly set quotechar and escapechar to handle messy medical notes
            df = pd.read_csv(
                csv_path, 
                # sep=",", 
                # engine="python", 
                # on_bad_lines="warn",
                # quoting=0, # csv.QUOTE_MINIMAL
                # quotechar='"',
                # doublequote=True
            )
    except Exception as e:
        return ("err", str(csv_path), repr(e))

    patient_col = next((c for c in df.columns if c.strip().lower() == "patient_string"), None)
    if not patient_col:
        return ("skip", str(csv_path), "no 'patient_string' column")

    # Clean patient_string by removing imaging sections
    # We do this for all rows now, as we aren't dropping rows entirely anymore
    # but rather cleaning the timeline within each row.
    df[patient_col] = df[patient_col].apply(remove_imaging_lines_from_timeline)

    # Recompute patient_string_character_len if present
    # char_len_col = next((c for c in df.columns if c.strip().lower() == "patient_string_character_len"), None)
    # if char_len_col:
    #     df[char_len_col] = df[patient_col].fillna("").str.len()

    # Save with quoting=1 (QUOTE_ALL) to ensure long strings with newlines/commas are safely contained
    df.to_csv(output_path, index=False)
    return ("ok", str(csv_path), f"wrote {output_path.name} ({len(df)} rows)")


def main(
    config_path: str,
    valid_tasks_json_path: str,
    output_suffix: str = "_no_report",
    overwrite: bool = True,
):
    tasks, base_dir, subsample = load_tasks_and_config(config_path)
    if not subsample:
        print("Config has subsample: false; this script only processes subsampled CSVs. Exiting.")
        return

    base_path = Path(base_dir)
    if not base_path.exists():
        print(f"Base dir not found: {base_path}")
        return

    task_to_source = load_task_source_csv(valid_tasks_json_path)

    to_process = []
    for task_name in tasks:
        source_csv = task_to_source.get(task_name)
        if not source_csv:
            print(f"[SKIP] No task_source_csv for task '{task_name}'")
            continue
        csv_path = base_path / source_csv / f"{task_name}_subsampled.csv"
        if not csv_path.exists():
            print(f"[SKIP] File not found: {csv_path}")
            continue
        out_path = csv_path.with_name(f"{csv_path.stem}{output_suffix}.csv")
        if out_path.exists() and not overwrite:
            print(f"[SKIP] Output exists (use overwrite=True): {out_path.name}")
            continue
        to_process.append((csv_path, out_path))

    if not to_process:
        print("No CSVs to process.")
        return

    print(f"Processing {len(to_process)} subsampled CSV(s)...")
    for csv_path, out_path in to_process:
        status, path_str, msg = process_csv(csv_path, out_path)
        name = csv_path.name
        if status == "ok":
            print(f"[OK]   {name}: {msg}")
        elif status == "skip":
            print(f"[SKIP] {name}: {msg}")
        else:
            print(f"[ERR]  {name}: {msg}")


if __name__ == "__main__":
    CONFIG_PATH = "/home/dcunhrya/vista_eval/configs/all_tasks.yaml"
    VALID_TASKS_JSON_PATH = "/home/dcunhrya/vista_bench/tasks/valid_tasks.json"
    import argparse

    parser = argparse.ArgumentParser(description="Remove STANFORD_NOTE/imaging rows from subsampled task CSVs.")
    parser.add_argument("--config", default=CONFIG_PATH, help="Path to all_tasks.yaml")
    parser.add_argument("--valid-tasks", default=VALID_TASKS_JSON_PATH, help="Path to valid_tasks.json")
    parser.add_argument("--suffix", default="_no_report", help="Suffix for output filenames (default: _no_report)")
    parser.add_argument("--overwrite", default=True, action="store_true", help="Overwrite existing output files")
    args = parser.parse_args()
    main(
        config_path=args.config,
        valid_tasks_json_path=args.valid_tasks,
        output_suffix=args.suffix,
        overwrite=args.overwrite,
    )
