"""
Extract average max input tokens from log files and add them to results.csv.

Reads logs from the logs/ folder (structure: logs/YYYYMMDD_HHMMSS/model_X_*.log).
Extracts model name from filename, task and experiment from log lines
('>>> Starting Task: {task} | Experiment: {experiment}'), and computes the
average of 'Batch N: Max input tokens = X' values for each task/experiment block.
Updates figures/results_stats/results.csv with a 'tokens' column.

Usage:
  python add_tokens_from_logs.py [--logs-dir LOGS_DIR] [--results-csv PATH] [--dry-run]
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd


def extract_model_name_from_log_filename(filename: str) -> str | None:
    """
    Extract model name for results.csv from log filename.

    Log files are named: model_{gpu_id}_{org}_{model_short}.log
    e.g. model_3_OctoMed_OctoMed-7B.log -> OctoMed-7B
         model_0_OpenGVLab_InternVL3_5-8B.log -> InternVL3_5-8B

    The model_name in results.csv is the part after the first underscore
    in the 'org_model' segment (org/model with / replaced by _).
    """
    m = re.match(r"model_\d+_(.+)\.log$", filename, re.IGNORECASE)
    if not m:
        return None
    org_model = m.group(1)
    if "_" in org_model:
        return org_model.split("_", 1)[1]
    return org_model


def parse_log_file(log_path: Path) -> dict[tuple[str, str, str], float]:
    """
    Parse a log file and return {(model_name, task, experiment): avg_tokens}.

    Looks for:
      - >>> Starting Task: {task} | Experiment: {experiment}
      - Batch N: Max input tokens = X  (until next Starting Task)
    """
    model_name = extract_model_name_from_log_filename(log_path.name)
    if not model_name:
        return {}

    result: dict[tuple[str, str, str], float] = {}
    task_start = re.compile(r">>> Starting Task: (.+?) \| Experiment: (.+?)\s*$")
    batch_tokens = re.compile(r"Batch \d+: Max input tokens = (\d+)\s*$")

    current_task: str | None = None
    current_experiment: str | None = None
    token_values: list[int] = []

    with open(log_path, "r", errors="replace") as f:
        for line in f:
            m = task_start.search(line)
            if m:
                # Save previous block if we have tokens
                if current_task is not None and current_experiment is not None and token_values:
                    key = (model_name, current_task.strip(), current_experiment.strip())
                    result[key] = sum(token_values) / len(token_values)
                    token_values = []

                current_task = m.group(1)
                current_experiment = m.group(2)
                continue

            b = batch_tokens.search(line)
            if b and current_task is not None:
                token_values.append(int(b.group(1)))

        # Don't forget the last block
        if current_task is not None and current_experiment is not None and token_values:
            key = (model_name, current_task.strip(), current_experiment.strip())
            result[key] = sum(token_values) / len(token_values)

    return result


def find_log_files(logs_dir: Path) -> list[Path]:
    """Find all .log files under logs_dir (recursively)."""
    return list(logs_dir.rglob("*.log"))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Add average tokens from logs to results.csv",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    vista_eval_root = Path(__file__).resolve().parents[3]  # vista_eval_vlm/
    parser.add_argument(
        "--logs-dir",
        type=Path,
        default=vista_eval_root / "logs",
        help="Directory containing log files (default: vista_eval_vlm/logs)",
    )
    parser.add_argument(
        "--results-csv",
        type=Path,
        default=vista_eval_root / "figures" / "results_stats" / "results.csv",
        help="Path to results.csv to update",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be updated without writing",
    )
    args = parser.parse_args()

    logs_dir = args.logs_dir
    results_path = args.results_csv

    if not logs_dir.exists():
        print(f"Logs directory does not exist: {logs_dir}")
        return

    if not results_path.exists():
        print(f"Results CSV does not exist: {results_path}")
        return

    # Collect (model_name, task, experiment) -> avg_tokens from all logs
    token_map: dict[tuple[str, str, str], float] = {}
    log_files = find_log_files(logs_dir)

    if not log_files:
        print(f"No .log files found under {logs_dir}")
        return

    for log_path in log_files:
        parsed = parse_log_file(log_path)
        for key, avg_tokens in parsed.items():
            # If multiple logs have same (model, task, experiment), keep first (or average?)
            if key not in token_map:
                token_map[key] = avg_tokens
            else:
                # Average if we have duplicates from different log runs
                token_map[key] = (token_map[key] + avg_tokens) / 2

    print(f"Parsed {len(log_files)} log file(s), found {len(token_map)} (model, task, experiment) entries")

    # Load results CSV
    df = pd.read_csv(results_path)

    if "tokens" not in df.columns:
        df["tokens"] = float("nan")

    matched = 0
    for (model_name, task, experiment), avg_tokens in token_map.items():
        mask = (
            (df["model_name"] == model_name)
            & (df["task"] == task)
            & (df["experiment"] == experiment)
        )
        n = mask.sum()
        if n > 0:
            df.loc[mask, "tokens"] = round(avg_tokens, 1)
            matched += n

    print(f"Matched {matched} rows in results.csv")

    if args.dry_run:
        print("\nDry run - not writing. Sample of updates:")
        has_tokens = df["tokens"].notna()
        if has_tokens.any():
            print(df.loc[has_tokens, ["task", "model_name", "experiment", "tokens"]].head(15).to_string())
        return

    df.to_csv(results_path, index=False)
    print(f"Wrote {results_path}")


if __name__ == "__main__":
    main()
