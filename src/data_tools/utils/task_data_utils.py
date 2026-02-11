"""
Task data loading utilities: path resolution, BQ/CSV merging for vista_run.
"""
from pathlib import Path
from typing import Optional

import pandas as pd

from data_tools.utils.query_utils import VISTA_BENCH_DATASET, fetch_task_data_from_bq


def resolve_local_bq_cache_path(base_path: Path, source_csv: str) -> Path:
    """Path to locally cached BigQuery data."""
    return base_path / "bigquery_data_2_3" / source_csv


def resolve_timeline_csv_filename(
    task_name: str, use_subsampled: bool, use_no_report_csv: bool
) -> str:
    """Resolve CSV filename for timeline data based on config."""
    if use_no_report_csv and use_subsampled:
        return f"{task_name}_subsampled_no_img_report.csv"
    if use_subsampled:
        return f"{task_name}_subsampled.csv"
    return f"{task_name}.csv"


def resolve_timeline_csv_path(
    base_path: Path, source_csv: str, task_name: str, use_subsampled: bool, use_no_report_csv: bool
) -> Path:
    """Full path to timeline CSV file."""
    filename = resolve_timeline_csv_filename(task_name, use_subsampled, use_no_report_csv)
    return base_path / source_csv / filename


def find_timeline_column(df: pd.DataFrame) -> Optional[str]:
    """Find patient timeline column (patient_string or patient_timeline). Returns None if not found."""
    return next(
        (c for c in df.columns if 'patient_string' in c.lower() or 'patient_timeline' in c.lower()),
        None,
    )


def find_bq_timeline_column(df: pd.DataFrame) -> str:
    """Find timeline column in BQ data; default to 'patient_string' if not found."""
    return next((c for c in df.columns if 'patient_string' in c.lower()), None) or 'patient_string'


def merge_bq_with_timeline_csv(
    bq_df: pd.DataFrame,
    csv_df: pd.DataFrame,
    timeline_col: str,
    use_no_report_csv: bool,
) -> Optional[pd.DataFrame]:
    """
    Merge BigQuery data with CSV timeline data on person_id.
    Returns merged DataFrame or None on error.
    """
    if 'person_id' not in bq_df.columns:
        return None
    if 'person_id' not in csv_df.columns:
        return None

    csv_timeline_col = find_timeline_column(csv_df)
    if csv_timeline_col is None:
        return None

    merge_cols = ['person_id', csv_timeline_col]
    if use_no_report_csv and 'report' in csv_df.columns:
        merge_cols.append('report')
    if 'embed_time' in csv_df.columns:
        merge_cols.append('embed_time')

    merge_df = csv_df[merge_cols].copy()
    merge_df = merge_df.rename(columns={csv_timeline_col: timeline_col})

    df = bq_df.copy()
    if timeline_col in df.columns:
        df = df.drop(columns=[timeline_col])
    if 'report' in df.columns:
        df = df.drop(columns=['report'])
    # Avoid duplicate embed_time when merging (prefer CSV value)
    if 'embed_time' in merge_cols and 'embed_time' in df.columns:
        df = df.drop(columns=['embed_time'])

    return df.merge(merge_df, on='person_id', how='inner')


def load_task_data_from_bq_or_cache(
    bq_client,
    base_path: Path,
    project_id: str,
    source_csv: str,
    task_name: str,
) -> Optional[pd.DataFrame]:
    """
    Load task data from local cache if exists, else from BigQuery.
    Returns DataFrame or None.
    """
    local_path = resolve_local_bq_cache_path(base_path, source_csv)
    if local_path.exists():
        try:
            df_all = pd.read_csv(local_path)
            df = df_all[df_all["task"] == task_name].copy()
            return df if not df.empty else None
        except Exception:
            return None

    dataset_id = VISTA_BENCH_DATASET
    full_table_id = f"{project_id}.{dataset_id}.{source_csv}"
    return fetch_task_data_from_bq(bq_client, full_table_id, task_name)
