"""
BigQuery query templates and execution helpers for VISTA data tools.
Use these from subsample_csv, download scripts, or any future BQ query callers.
"""
import numpy as np
import pandas as pd
from google.cloud import bigquery

from data_tools.utils.config_utils import get_bq_client

# Project and table naming
DEFAULT_BQ_PROJECT_ID = "som-nero-plevriti-deidbdf"
BATCH_SIZE = 10000

# Note table (OMOP) for report/note_text lookups
DEFAULT_NOTE_DATASET = "oncology_omop54_confidential_irb76049_nov2025"
DEFAULT_NOTE_TABLE = "note"


def get_note_by_person_accession_query(dataset: str, table: str) -> str:
    """Return SQL for note_id, note_text, note_datetime by person_id and _accession_number. Uses @person_id (INT64), @accession_number (STRING)."""
    return f"""
        SELECT note_id, note_text, note_datetime
        FROM `{dataset}.{table}`
        WHERE person_id = @person_id AND _accession_number = @accession_number
    """


def get_note_ids_by_person_accession_query(dataset: str, table: str) -> str:
    """Return SQL for note_id only by person_id and _accession_number. Uses @person_id (INT64), @accession_number (STRING)."""
    return f"""
        SELECT note_id
        FROM `{dataset}.{table}`
        WHERE person_id = @person_id AND _accession_number = @accession_number
    """


def get_note_ids_from_bq(
    bq_client: bigquery.Client,
    person_id: int,
    accession_number: str,
    dataset: str = DEFAULT_NOTE_DATASET,
    table: str = DEFAULT_NOTE_TABLE,
) -> list:
    """
    Query BigQuery 'note' for note_id(s) matching person_id and _accession_number.
    Returns a list of note_id (may be multiple rows).
    """
    if pd.isna(person_id) or pd.isna(accession_number) or str(accession_number).strip() == "":
        return []
    query = get_note_ids_by_person_accession_query(dataset, table)
    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("person_id", "INT64", int(person_id)),
            bigquery.ScalarQueryParameter(
                "accession_number", "STRING", str(accession_number).strip()
            ),
        ]
    )
    df = bq_client.query(query, job_config=job_config).to_dataframe()
    return df["note_id"].astype(int).tolist()


def get_notes_from_bq(
    bq_client: bigquery.Client,
    person_id: int,
    accession_number: str,
    dataset: str = DEFAULT_NOTE_DATASET,
    table: str = DEFAULT_NOTE_TABLE,
) -> pd.DataFrame:
    """
    Query BigQuery 'note' for note_id and note_text matching person_id and _accession_number.
    Returns a DataFrame with columns note_id, note_text (may be multiple rows).
    """
    if pd.isna(person_id) or pd.isna(accession_number) or str(accession_number).strip() == "":
        return pd.DataFrame(columns=["note_id", "note_text"])
    query = get_note_by_person_accession_query(dataset, table)
    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("person_id", "INT64", int(person_id)),
            bigquery.ScalarQueryParameter(
                "accession_number", "STRING", str(accession_number).strip()
            )
        ],
    )
    return bq_client.query(query, job_config=job_config).to_dataframe()


# Batch size for get_notes_from_bq_batch (pairs per query; 2 params per pair, BQ limit 10000)
NOTES_BATCH_SIZE = 500


def get_notes_from_bq_batch(
    bq_client: bigquery.Client,
    person_accession_pairs: list[tuple[int | str, str]],
    dataset: str = DEFAULT_NOTE_DATASET,
    table: str = DEFAULT_NOTE_TABLE,
    batch_size: int = NOTES_BATCH_SIZE,
) -> pd.DataFrame:
    """
    Fetch note_id, note_text for many (person_id, accession_number) pairs in batched queries.
    Returns a DataFrame with columns person_id, _accession_number, note_id, note_text.
    Pairs with null/empty accession are skipped.
    """
    pairs = []
    seen = set()
    for person_id, accession in person_accession_pairs:
        if pd.isna(person_id) or pd.isna(accession) or str(accession).strip() == "":
            continue
        try:
            pid = int(person_id)
            acc = str(accession).strip()
        except (ValueError, TypeError):
            continue
        key = (pid, acc)
        if key in seen:
            continue
        seen.add(key)
        pairs.append((pid, acc))

    if not pairs:
        return pd.DataFrame(columns=["person_id", "_accession_number", "note_id", "note_text", "note_datetime"])

    full_table = f"{dataset}.{table}"
    out_dfs = []
    for start in range(0, len(pairs), batch_size):
        chunk = pairs[start : start + batch_size]
        # WHERE (person_id, _accession_number) IN ((@p0,@a0), (@p1,@a1), ...)
        in_parts = [f"(@p{i}, @a{i})" for i in range(len(chunk))]
        in_clause = ", ".join(in_parts)
        query = f"""
            SELECT person_id, _accession_number, note_id, note_text, note_datetime
            FROM `{full_table}`
            WHERE (person_id, _accession_number) IN ({in_clause})
        """
        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                param
                for i, (pid, acc) in enumerate(chunk)
                for param in [
                    bigquery.ScalarQueryParameter(f"p{i}", "INT64", pid),
                    bigquery.ScalarQueryParameter(f"a{i}", "STRING", acc),
                ]
            ]
        )
        df_chunk = bq_client.query(query, job_config=job_config).to_dataframe()
        if not df_chunk.empty:
            out_dfs.append(df_chunk)
    if not out_dfs:
        return pd.DataFrame(columns=["person_id", "_accession_number", "note_id", "note_text", "note_datetime"])
    return pd.concat(out_dfs, ignore_index=True)


def get_person_id_nifti_paths_query(full_table_id: str) -> str:
    """Return SQL for fetching (person_id, nifti_path) for person_ids in @person_ids (INT64 array)."""
    return f"""
        SELECT person_id, nifti_path
        FROM `{full_table_id}`
        WHERE person_id IN UNNEST(@person_ids)
        AND nifti_path IS NOT NULL
        AND TRIM(CAST(nifti_path AS STRING)) != ''
    """


def get_ct_available_person_ids_query(full_table_id: str) -> str:
    """Return SQL for distinct person_id with non-null nifti_path; uses @person_ids (INT64 array)."""
    return f"""
        SELECT DISTINCT person_id
        FROM `{full_table_id}`
        WHERE person_id IN UNNEST(@person_ids)
        AND nifti_path IS NOT NULL
        AND TRIM(CAST(nifti_path AS STRING)) != ''
    """


def get_ct_available_person_ids_query_fallback(full_table_id: str, person_ids_in_clause: str) -> str:
    """Return SQL for distinct person_id with inline IN clause (fallback when parameterized query fails)."""
    return f"""
        SELECT DISTINCT person_id
        FROM `{full_table_id}`
        WHERE person_id IN ({person_ids_in_clause})
        AND nifti_path IS NOT NULL
        AND TRIM(CAST(nifti_path AS STRING)) != ''
    """


def fetch_person_id_nifti_paths(
    person_ids,
    dataset_id: str,
    table_name: str,
    project_id: str = DEFAULT_BQ_PROJECT_ID,
    batch_size: int = BATCH_SIZE,
):
    """
    Query BigQuery for (person_id, nifti_path) for the given person_ids.
    Returns list of (person_id, nifti_path) with non-null, non-empty nifti_path.
    """
    if not person_ids or not table_name:
        return []
    person_ids_int = []
    for pid in person_ids:
        try:
            v = int(pid)
            if -(2**63) <= v < 2**63:
                person_ids_int.append(v)
        except (ValueError, TypeError, OverflowError):
            continue
    if not person_ids_int:
        return []
    try:
        client = get_bq_client()
        full_table_id = f"{project_id}.{dataset_id}.{table_name}"
        query = get_person_id_nifti_paths_query(full_table_id)
        pairs = []
        for i in range(0, len(person_ids_int), batch_size):
            batch = person_ids_int[i : i + batch_size]
            job_config = bigquery.QueryJobConfig(
                query_parameters=[bigquery.ArrayQueryParameter("person_ids", "INT64", batch)]
            )
            result = client.query(query, job_config=job_config).to_dataframe()
            for _, row in result.iterrows():
                pid = row.get("person_id")
                path = row.get("nifti_path")
                if pd.notna(pid) and pd.notna(path) and str(path).strip():
                    pairs.append((pid, str(path).strip()))
        return pairs
    except Exception as e:
        print(f"  Warning: Error fetching (person_id, nifti_path) from BQ: {e}")
        return []


def check_ct_available_batch(
    person_ids,
    dataset_id: str = "vista_bench_v1_1",
    table_name: str = None,
    project_id: str = DEFAULT_BQ_PROJECT_ID,
    batch_size: int = BATCH_SIZE,
):
    """
    Batch check if CT scans are available for multiple person_ids (non-null nifti_path in BQ).
    Returns a set of person_ids that have nifti_path available.
    """
    if not person_ids or table_name is None:
        return set()

    person_ids_list = []
    for pid in person_ids:
        try:
            if isinstance(pid, (pd.Series, np.ndarray)):
                pid = pid.item() if len(pid) > 0 else None
            elif isinstance(pid, (list, tuple)):
                pid = pid[0] if len(pid) > 0 else None
            if pid is None:
                continue
            try:
                if np.isnan(float(pid)):
                    continue
            except (ValueError, TypeError):
                pass
            person_ids_list.append(pid)
        except Exception:
            continue
    if not person_ids_list:
        return set()

    person_ids_int = []
    for pid in person_ids_list:
        try:
            v = int(pid)
            if -(2**63) <= v < 2**63:
                person_ids_int.append(v)
        except (ValueError, TypeError, OverflowError):
            continue
    if not person_ids_int:
        return set()

    available_person_ids = set()
    try:
        client = get_bq_client()
        full_table_id = f"{project_id}.{dataset_id}.{table_name}"
        query = get_ct_available_person_ids_query(full_table_id)
        for i in range(0, len(person_ids_int), batch_size):
            batch = person_ids_int[i : i + batch_size]
            job_config = bigquery.QueryJobConfig(
                query_parameters=[bigquery.ArrayQueryParameter("person_ids", "INT64", batch)]
            )
            result = client.query(query, job_config=job_config).to_dataframe()
            if len(result) > 0:
                for pid in result["person_id"].dropna():
                    available_person_ids.add(int(pid))
                    available_person_ids.add(str(pid))
        return available_person_ids
    except Exception as e:
        try:
            client = get_bq_client()
            full_table_id = f"{project_id}.{dataset_id}.{table_name}"
            for i in range(0, len(person_ids_int), batch_size):
                batch = person_ids_int[i : i + batch_size]
                person_ids_in_clause = ", ".join(str(x) for x in batch)
                query_fallback = get_ct_available_person_ids_query_fallback(
                    full_table_id, person_ids_in_clause
                )
                result = client.query(query_fallback).to_dataframe()
                if len(result) > 0:
                    for pid in result["person_id"].dropna():
                        available_person_ids.add(int(pid))
                        available_person_ids.add(str(pid))
        except Exception as e2:
            print(f"  Warning: Error in batch CT availability check (fallback): {e2}")
        return available_person_ids
