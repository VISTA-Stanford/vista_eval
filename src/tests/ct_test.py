"""
CT / NIfTI path tests: check how many nifti_path from subsampled CSVs exist in GCS,
and how many nifti_path for a task exist in the bucket (from BigQuery).
"""
from pathlib import Path

import pandas as pd

from src.data_tools.utils.config_utils import (
    load_tasks_and_base_dir,
    get_bq_client,
)
from data_tools.utils.ct_utils import check_nifti_exists_in_bucket
from data_tools.utils.task_utils import load_task_mappings
from data_tools.utils.query_utils import DEFAULT_BQ_PROJECT_ID

# GCS bucket settings (same as subsample_csv / download_subsampled_ct)
DEFAULT_GCS_BUCKET_NAME = "su-vista-uscentral1"
DEFAULT_GCS_PREFIX = "chaudhari_lab/ct_data/ct_scans/vista/nov25"


def count_subsampled_nifti_paths_existing(
    config_path: str,
    valid_tasks_json_path: str | None = None,
    bucket_name: str = DEFAULT_GCS_BUCKET_NAME,
    prefix: str = DEFAULT_GCS_PREFIX,
):
    """
    For subsampled CSVs that have a nifti_path column (using tasks from config and
    base_dir), check how many of those nifti_paths actually exist in the GCS bucket.

    Uses vista_eval_vlm/configs/all_tasks.yaml for tasks and base_dir, and
    src/data_tools/utils/ct_utils.py for bucket existence checks.

    Returns:
        dict: task_name -> {"total": int, "existing": int, "csv_path": str}
    """
    tasks, base_dir = load_tasks_and_base_dir(config_path)
    base_path = Path(base_dir)
    if not base_path.exists():
        return {}

    valid_tasks_path = valid_tasks_json_path
    if valid_tasks_path is None:
        valid_tasks_path = str(base_path / "tasks" / "valid_tasks.json")
    _, task_table_map = load_task_mappings(valid_tasks_path)

    subsampled_files = list(base_path.rglob("*_subsampled.csv"))
    results = {}

    try:
        from google.cloud import storage
        client = storage.Client()
        bucket = client.bucket(bucket_name)
    except Exception as e:
        print(f"Warning: Could not create GCS client: {e}")
        return results

    for csv_path in subsampled_files:
        task_name = csv_path.stem.replace("_subsampled", "")
        if task_name not in tasks:
            continue
        try:
            df = pd.read_csv(csv_path, sep=None, engine="python", on_bad_lines="warn")
        except Exception as e:
            print(f"  Skip {csv_path.name}: {e}")
            continue
        path_col = next((c for c in df.columns if c.lower() == "nifti_path"), None)
        if path_col is None:
            continue
        paths = df[path_col].dropna().astype(str).str.strip()
        paths = paths[paths != ""].unique()
        total = len(paths)
        if total == 0:
            results[task_name] = {"total": 0, "existing": 0, "csv_path": str(csv_path)}
            continue
        existing = sum(
            1
            for p in paths
            if check_nifti_exists_in_bucket(p, bucket_name, prefix, bucket)
        )
        results[task_name] = {
            "total": total,
            "existing": existing,
            "csv_path": str(csv_path),
        }
    return results


def count_task_nifti_paths_in_bucket_from_bigquery(
    task_name: str,
    config_path: str,
    valid_tasks_json_path: str | None = None,
    dataset_id: str = "vista_bench_v1_1",
    project_id: str = DEFAULT_BQ_PROJECT_ID,
    bucket_name: str = DEFAULT_GCS_BUCKET_NAME,
    prefix: str = DEFAULT_GCS_PREFIX,
):
    """
    For a given task, query BigQuery for all rows with non-null nifti_path in that
    task's table, then check how many of those nifti_paths exist in the GCS bucket.

    Returns:
        dict with keys: total (from BQ), in_bucket (count that exist in bucket),
        table_name.
    """
    _, base_dir = load_tasks_and_base_dir(config_path)
    base_path = Path(base_dir)
    valid_tasks_path = valid_tasks_json_path
    if valid_tasks_path is None:
        valid_tasks_path = str(base_path / "tasks" / "valid_tasks.json")
    _, task_table_map = load_task_mappings(valid_tasks_path)
    table_name = task_table_map.get(task_name)
    if not table_name:
        return {
            "total": 0,
            "in_bucket": 0,
            "table_name": None,
            "error": f"No BigQuery table mapping for task '{task_name}'",
        }

    full_table_id = f"{project_id}.{dataset_id}.{table_name}"
    query = f"""
        SELECT DISTINCT nifti_path
        FROM `{full_table_id}`
        WHERE nifti_path IS NOT NULL
        AND TRIM(CAST(nifti_path AS STRING)) != ''
    """
    try:
        client = get_bq_client()
        df = client.query(query).to_dataframe()
    except Exception as e:
        return {
            "total": 0,
            "in_bucket": 0,
            "table_name": table_name,
            "error": str(e),
        }

    paths = df["nifti_path"].astype(str).str.strip().unique().tolist()
    total = len(paths)
    if total == 0:
        return {"total": 0, "in_bucket": 0, "table_name": table_name}

    try:
        from google.cloud import storage
        gcs_client = storage.Client()
        bucket = gcs_client.bucket(bucket_name)
    except Exception as e:
        return {
            "total": total,
            "in_bucket": 0,
            "table_name": table_name,
            "error": f"GCS client: {e}",
        }

    in_bucket = sum(
        1
        for p in paths
        if check_nifti_exists_in_bucket(p, bucket_name, prefix, bucket)
    )
    return {
        "total": total,
        "in_bucket": in_bucket,
        "table_name": table_name,
    }
