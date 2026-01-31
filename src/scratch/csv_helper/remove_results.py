"""
Remove CSV files from the results folder whose filenames contain a given phrase.

Supports filtering by task (source folder), subtask, and model. When not specified,
all matching CSVs across tasks/subtasks/models are considered.

Usage examples:
  # Remove all CSVs containing "no_report" in filename (dry run)
  python remove_results.py --phrase no_report --dry-run

  # Remove no_report CSVs only for a specific subtask
  python remove_results.py --phrase no_report --subtask pneumonitis_infection_answer --yes

  # Remove no_report CSVs only for a specific model
  python remove_results.py --phrase no_report --model Qwen3-VL-8B-Instruct --yes

  # Remove no_report CSVs for a specific task (source folder)
  python remove_results.py --phrase no_report --task tb_v1_1_tb_classification_tasks --yes
"""

import argparse
import sys
from pathlib import Path

import yaml


def load_results_dir_from_config(config_path: Path) -> str | None:
    """Load results_dir from vista_eval config if it exists."""
    if not config_path.exists():
        return None
    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        return config.get("paths", {}).get("results_dir")
    except Exception:
        return None


def find_matching_csvs(
    results_dir: Path,
    phrase: str,
    task: str | None = None,
    subtask: str | None = None,
    model: str | None = None,
) -> list[Path]:
    """
    Find all CSV files under results_dir whose filename contains phrase.
    Path structure: results_dir / task / subtask / model / {subtask}_results_{experiment}.csv

    Args:
        results_dir: Root results directory
        phrase: Substring that must appear in the CSV filename
        task: If set, only consider CSVs under this source folder (e.g. tb_v1_1_tb_classification_tasks)
        subtask: If set, only consider CSVs under this task name (e.g. pneumonitis_infection_answer)
        model: If set, only consider CSVs under this model folder (e.g. Qwen3-VL-8B-Instruct)

    Returns:
        List of Path objects to matching CSV files
    """
    matches: list[Path] = []
    if not results_dir.exists():
        return matches

    for csv_path in results_dir.rglob("*_results_*.csv"):
        if phrase not in csv_path.name:
            continue
        parts = csv_path.relative_to(results_dir).parts
        if len(parts) < 4:
            continue  # Need at least task/subtask/model/filename
        path_task, path_subtask, path_model = parts[0], parts[1], parts[2]
        if task is not None and path_task != task:
            continue
        if subtask is not None and path_subtask != subtask:
            continue
        if model is not None and path_model != model:
            continue
        matches.append(csv_path)

    return sorted(matches)


def main():
    parser = argparse.ArgumentParser(
        description="Remove result CSVs whose filenames contain a given phrase.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--phrase",
        type=str,
        required=True,
        help="Phrase that must appear in the CSV filename (e.g. no_report)",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default=None,
        help="Results directory (default: from configs/all_tasks.yaml or /home/dcunhrya/results)",
    )
    parser.add_argument(
        "--task",
        type=str,
        default=None,
        help="Restrict to this source folder (e.g. tb_v1_1_tb_classification_tasks)",
    )
    parser.add_argument(
        "--subtask",
        type=str,
        default=None,
        help="Restrict to this subtask (e.g. pneumonitis_infection_answer)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Restrict to this model folder (e.g. Qwen3-VL-8B-Instruct)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be removed without deleting",
    )
    parser.add_argument(
        "--yes", "-y",
        action="store_true",
        help="Skip confirmation prompt and delete immediately",
    )
    args = parser.parse_args()

    # Resolve results directory
    if args.results_dir:
        results_dir = Path(args.results_dir)
    else:
        config_path = Path(__file__).resolve().parents[3] / "configs" / "all_tasks.yaml"
        results_dir_str = load_results_dir_from_config(config_path)
        results_dir = Path(results_dir_str) if results_dir_str else Path("/home/dcunhrya/results")

    if not results_dir.exists():
        print(f"Error: Results directory does not exist: {results_dir}", file=sys.stderr)
        sys.exit(1)

    matches = find_matching_csvs(
        results_dir,
        args.phrase,
        task=args.task,
        subtask=args.subtask,
        model=args.model,
    )

    if not matches:
        filters = []
        if args.task:
            filters.append(f"task={args.task}")
        if args.subtask:
            filters.append(f"subtask={args.subtask}")
        if args.model:
            filters.append(f"model={args.model}")
        filter_str = " " + " ".join(filters) if filters else ""
        print(f"No CSV files containing '{args.phrase}' found{filter_str}.")
        return

    print(f"Found {len(matches)} CSV(s) containing '{args.phrase}':")
    for p in matches:
        rel = p.relative_to(results_dir)
        print(f"  {rel}")

    if args.dry_run:
        print("\n[DRY RUN] No files were removed.")
        return

    if not args.yes:
        confirm = input("\nRemove these files? [y/N]: ").strip().lower()
        if confirm not in ("y", "yes"):
            print("Aborted.")
            return

    removed = 0
    for p in matches:
        try:
            p.unlink()
            removed += 1
            print(f"  Removed: {p.relative_to(results_dir)}")
        except OSError as e:
            print(f"  Error removing {p}: {e}", file=sys.stderr)
    print(f"\nRemoved {removed} file(s).")


if __name__ == "__main__":
    main()
