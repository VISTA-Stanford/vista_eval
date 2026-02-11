"""
Download patient XML files from GCS for tasks defined in all_tasks.yaml.

Uses _subsampled CSVs to get person_ids, downloads {person_id}.xml from
gs://vista_bench/thoracic_cohort_lumia/ to a local directory.

Reports: total downloaded, existing (already on disk), not present in bucket.
"""

import csv
import json
import sys
from pathlib import Path

import pandas as pd
import yaml
from google.cloud import storage
from tqdm import tqdm

# Increase the limit to handle very large clinical text fields
csv.field_size_limit(sys.maxsize)


def load_tasks_and_paths(config_path: str, valid_tasks_path: str) -> tuple[list[str], Path, dict]:
    """Load task names, base_dir, and task_name -> task_source_csv mapping."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    tasks = config.get("tasks", [])
    base_dir = Path(config.get("paths", {}).get("base_dir", ""))

    with open(valid_tasks_path, "r") as f:
        task_defs = json.load(f)
    task_to_source = {
        t["task_name"]: t["task_source_csv"]
        for t in task_defs
        if t.get("task_source_csv")
    }

    return tasks, base_dir, task_to_source


def collect_person_ids_from_tasks(
    base_dir: Path,
    tasks: list[str],
    task_to_source: dict[str, str],
    file_suffix: str = "_subsampled",
) -> set[str]:
    """Collect unique person_ids from _subsampled CSVs for the given tasks."""
    person_ids = set()
    for task_name in tasks:
        source_csv = task_to_source.get(task_name)
        if not source_csv:
            continue
        csv_path = base_dir / source_csv / f"{task_name}{file_suffix}.csv"
        if not csv_path.exists():
            continue
        try:
            df = pd.read_csv(csv_path, sep=None, engine="python", on_bad_lines="warn")
            pid_col = next(
                (c for c in df.columns if c.lower() in ("person_id", "patient_id")),
                None,
            )
            if pid_col:
                ids = df[pid_col].dropna().astype(str).str.strip()
                person_ids.update(ids.unique())
        except Exception as e:
            print(f"  [SKIP] {csv_path.name}: {e}")
    return person_ids


def download_xmls(
    config_path: str,
    valid_tasks_path: str,
    bucket_name: str = "vista_bench",
    prefix: str = "thoracic_cohort_lumia",
    download_dir: str = "/data/fries/datasets/vista_bench_ryan/thoracic_cohort_lumia",
    file_suffix: str = "_subsampled",
    dry_run: bool = False,
    person_id: str | None = None,
) -> dict:
    """
    Download XML files for all tasks defined in config, or for a single person_id.

    Args:
        config_path: Path to all_tasks.yaml.
        valid_tasks_path: Path to valid_tasks.json.
        bucket_name: GCS bucket name.
        prefix: Object prefix (folder) in bucket.
        download_dir: Local directory to save XMLs.
        file_suffix: CSV filename suffix (e.g. _subsampled).
        dry_run: If True, only report without downloading.
        person_id: If set, download only this person's XML (skips task-based collection).

    Returns:
        Dict with keys: downloaded, existing, not_in_bucket.
    """
    if person_id:
        person_ids = {str(person_id).strip()}
        print(f"Downloading XML for single person_id: {person_id}")
    else:
        tasks, base_dir, task_to_source = load_tasks_and_paths(config_path, valid_tasks_path)
        if not tasks:
            print("No tasks found in config.")
            return {"downloaded": 0, "existing": 0, "not_in_bucket": 0}

        print(f"Loading person_ids from {len(tasks)} tasks...")
        person_ids = collect_person_ids_from_tasks(
            base_dir, tasks, task_to_source, file_suffix
        )
        print(f"Found {len(person_ids)} unique person_ids across tasks.")

    client = storage.Client()
    bucket = client.bucket(bucket_name)
    out_dir = Path(download_dir)
    if not dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)

    stats = {"downloaded": 0, "existing": 0, "not_in_bucket": 0}

    for person_id in tqdm(sorted(person_ids), desc="Processing"):
        if not person_id or not str(person_id).strip():
            continue
        blob_name = f"{prefix}/{person_id}.xml"
        blob = bucket.blob(blob_name)
        local_path = out_dir / f"{person_id}.xml"

        if not blob.exists():
            stats["not_in_bucket"] += 1
            continue

        if local_path.exists():
            stats["existing"] += 1
            continue

        stats["downloaded"] += 1
        if not dry_run:
            blob.download_to_filename(str(local_path))

    return stats


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Download XML files for vista_eval_vlm tasks")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to all_tasks.yaml (default: vista_eval_vlm/configs/all_tasks.yaml)",
    )
    parser.add_argument(
        "--valid-tasks",
        type=str,
        default=None,
        help="Path to valid_tasks.json (default: vista_bench/tasks/valid_tasks.json)",
    )
    parser.add_argument(
        "--bucket",
        type=str,
        default="vista_bench",
        help="GCS bucket name",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="thoracic_cohort_lumia",
        help="Object prefix in bucket",
    )
    parser.add_argument(
        "--download-dir",
        type=str,
        default="/data/fries/datasets/vista_bench_ryan/thoracic_cohort_lumia",
        help="Local directory to save XMLs",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only report, do not download",
    )
    parser.add_argument(
        "--person-id",
        type=str,
        default=None,
        help="Download XML for this specific person_id only (skips task-based collection)",
    )
    args = parser.parse_args()

    # Resolve default paths relative to vista_eval_vlm
    script_dir = Path(__file__).resolve().parent
    vista_eval_root = script_dir.parent.parent.parent
    vista_bench_root = vista_eval_root.parent / "vista_bench"

    config_path = args.config or str(vista_eval_root / "configs" / "all_tasks.yaml")
    valid_tasks_path = args.valid_tasks or str(vista_bench_root / "tasks" / "valid_tasks.json")

    if args.person_id:
        if not str(args.person_id).strip():
            print("Error: --person-id must be non-empty")
            return 1
    else:
        if not Path(config_path).exists():
            print(f"Error: Config not found: {config_path}")
            return 1
        if not Path(valid_tasks_path).exists():
            print(f"Error: valid_tasks.json not found: {valid_tasks_path}")
            return 1

    print("=" * 50)
    print("XML Download for vista_eval_vlm tasks")
    print("=" * 50)
    print(f"Config: {config_path}")
    print(f"Bucket: gs://{args.bucket}/{args.prefix}/")
    print(f"Download to: {args.download_dir}")
    print(f"Mode: {'DRY RUN' if args.dry_run else 'LIVE'}")
    print("=" * 50)

    stats = download_xmls(
        config_path=config_path,
        valid_tasks_path=valid_tasks_path,
        bucket_name=args.bucket,
        prefix=args.prefix,
        download_dir=args.download_dir,
        dry_run=args.dry_run,
        person_id=args.person_id,
    )

    print("\n" + "=" * 50)
    print(" SUMMARY ")
    print("=" * 50)
    print(f"Total downloaded: {stats['downloaded']}")
    print(f"Already existing (on disk): {stats['existing']}")
    print(f"Not present in bucket: {stats['not_in_bucket']}")
    total = stats["downloaded"] + stats["existing"] + stats["not_in_bucket"]
    print(f"Total unique person_ids: {total}")
    if total > 0:
        pct = 100 * (stats["downloaded"] + stats["existing"]) / total
        print(f"Successfully available: {pct:.1f}%")
    print("=" * 50)

    return 0


if __name__ == "__main__":
    sys.exit(main())
