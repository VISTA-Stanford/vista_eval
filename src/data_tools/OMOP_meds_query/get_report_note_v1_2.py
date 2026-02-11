"""Add 'report' column to v1_2 _subsampled_no_img_report CSVs by combining latest_img_date and note_text.

No BigQuery - all data is in the CSV. Uses same format as get_report_note.py:
[YYYY-MM-DD HH:MM] | NOTE: clean_text
Skips rows where note_text contains "non-reportable" (sets report to empty/NaN).
"""

from pathlib import Path

import pandas as pd

from data_tools.utils.config_utils import load_tasks_and_base_dir, load_task_source_csv
from data_tools.utils.report_utils import format_note_text_like_text_value


def _format_datetime(dt_val) -> str:
    """Format as '[YYYY-MM-DD HH:MM]' or return empty string if missing/invalid."""
    if pd.isna(dt_val):
        return ""
    try:
        dt = pd.to_datetime(dt_val)
        return f"[{dt.strftime('%Y-%m-%d %H:%M')}]"
    except (ValueError, TypeError, AttributeError):
        return ""


def build_report_from_row(
    latest_img_date,
    note_text: str,
    max_text_len: int | None = None,
):
    """
    Build report string from latest_img_date and note_text, same format as get_report_note.
    Returns empty string if note_text contains "non-reportable" (case insensitive).
    """
    if pd.isna(note_text) or str(note_text).strip() == "":
        return ""
    if "non-reportable" in str(note_text).lower():
        return ""
    note_line = format_note_text_like_text_value(note_text, max_text_len=max_text_len)
    if not note_line:
        return ""
    dt_prefix = _format_datetime(latest_img_date)
    if dt_prefix:
        return f"{dt_prefix} | {note_line}"
    return note_line


def process_csv_add_report(
    csv_path: Path,
    overwrite_report: bool = False,
    max_text_len: int | None = None,
) -> pd.DataFrame:
    """
    Read _subsampled_no_img_report CSV and add 'report' column by combining latest_img_date
    and note_text. Skips rows where note_text contains "non-reportable".
    """
    df = pd.read_csv(csv_path)

    if "report" in df.columns and not overwrite_report:
        return df

    latest_col = next((c for c in df.columns if c.lower() == "latest_img_date"), None)
    note_col = next((c for c in df.columns if c.lower() == "note_text"), None)
    if latest_col is None:
        raise ValueError(f"No 'latest_img_date' column. cols={list(df.columns)}")
    if note_col is None:
        raise ValueError(f"No 'note_text' column. cols={list(df.columns)}")

    df["report"] = [
        build_report_from_row(row[latest_col], row[note_col], max_text_len=max_text_len)
        for _, row in df.iterrows()
    ]
    return df


def run(
    config_path: str,
    valid_tasks_json_path: str,
    v1_2_base_dir: str | Path | None = None,
    overwrite: bool = False,
    max_text_len: int | None = None,
):
    """
    For each task in config, load the v1_2 _subsampled_no_img_report CSV, add 'report' column
    (combining latest_img_date and note_text from CSV), and save back to the same file.
    """
    tasks, base_dir = load_tasks_and_base_dir(config_path)
    task_to_source = load_task_source_csv(valid_tasks_json_path)
    base_path = Path(base_dir)

    if v1_2_base_dir is None:
        v1_2_base_dir = base_path / "v1_2"
    else:
        v1_2_base_dir = Path(v1_2_base_dir)

    if not v1_2_base_dir.exists():
        print(f"Error: v1_2 base dir not found: {v1_2_base_dir}")
        return

    for task_name in tasks:
        source_csv = task_to_source.get(task_name)
        if not source_csv:
            print(f"[SKIP] No task_source_csv for task '{task_name}'")
            continue

        csv_path = v1_2_base_dir / source_csv / f"{task_name}_subsampled_no_img_report.csv"
        if not csv_path.exists():
            print(f"[SKIP] File not found: {csv_path}")
            continue

        print(f"Processing: {csv_path.name}")
        try:
            df_out = process_csv_add_report(
                csv_path,
                overwrite_report=overwrite,
                max_text_len=max_text_len,
            )
            df_out.to_csv(csv_path, index=False)
            n_empty = (df_out["report"] == "").sum()
            print(f"  Wrote {len(df_out)} rows with 'report' column to {csv_path}")
            print(f"  'report' entries that are empty: {n_empty}")
        except Exception as e:
            print(f"[ERROR] {csv_path.name}: {e}")
            raise

    print("Done.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Add 'report' column to v1_2 _subsampled_no_img_report CSVs from latest_img_date and note_text (no BQ).",
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
    parser.add_argument(
        "--v1-2-dir",
        default=None,
        help="Base dir for v1_2 CSVs (default: {base_dir}/v1_2)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing 'report' column if present",
    )
    parser.add_argument(
        "--max-text-len",
        type=int,
        default=None,
        help="If set, truncate each note_text to this many characters.",
    )
    args = parser.parse_args()

    run(
        config_path=args.config,
        valid_tasks_json_path=args.valid_tasks,
        v1_2_base_dir=args.v1_2_dir,
        overwrite=args.overwrite,
        max_text_len=args.max_text_len,
    )
