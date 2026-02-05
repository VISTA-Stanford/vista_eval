import os
import pandas as pd
import numpy as np
import csv
import sys
import yaml
import json
import re
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from google.cloud import bigquery
from google.cloud import storage
from google.auth import default

csv.field_size_limit(sys.maxsize)

# GCS bucket settings for NIfTI CT scans (same as download_subsampled_ct.py)
DEFAULT_GCS_BUCKET_NAME = "su-vista-uscentral1"
DEFAULT_GCS_PREFIX = "chaudhari_lab/ct_data/ct_scans/vista/nov25"

def load_tasks_from_config(config_path):
    """Load task list from YAML config file."""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            tasks = config.get('tasks', [])
            return set(tasks)
    except Exception as e:
        print(f"Error loading config from {config_path}: {e}")
        return set()


def group_related_tasks(task_names):
    """
    Group tasks that differ only by year number (e.g., died_of_cancer_1_yr, died_of_cancer_2_yr).
    
    Returns:
        dict mapping base_name to sorted list of task names (sorted by year)
    """
    task_groups = {}
    
    # Pattern to match task names ending with _N_yr where N is a digit
    pattern = r'^(.+)_(\d+)_yr$'
    
    for task_name in task_names:
        match = re.match(pattern, task_name)
        if match:
            base_name = match.group(1)  # e.g., "died_of_cancer" or "has_recurrence"
            year = int(match.group(2))  # e.g., 1, 2, 3, 4, 5
            
            if base_name not in task_groups:
                task_groups[base_name] = []
            task_groups[base_name].append((year, task_name))
    
    # Sort each group by year and extract just task names
    result = {}
    for base_name, tasks in task_groups.items():
        if len(tasks) > 1:  # Only group if there are multiple related tasks
            sorted_tasks = sorted(tasks, key=lambda x: x[0])
            result[base_name] = [task_name for _, task_name in sorted_tasks]
    
    return result


def load_task_mappings(valid_tasks_json_path):
    """Load task mappings from valid_tasks.json and return dicts mapping task_name to excluded label values and table names.
    
    Returns:
        task_exclusions: dict mapping task_name to sets of excluded label values
        task_table_map: dict mapping task_name to BigQuery table name (task_source_csv)
    """
    try:
        with open(valid_tasks_json_path, 'r') as f:
            tasks = json.load(f)
        
        task_exclusions = {}
        task_table_map = {}
        for task in tasks:
            task_name = task.get('task_name')
            mapping = task.get('mapping', {})
            task_source_csv = task.get('task_source_csv')
            
            # Store table mapping
            if task_source_csv:
                task_table_map[task_name] = task_source_csv
            
            # Exclude: (1) label -1 always, (2) any label that maps to "Insufficient follow-up or missing data"
            excluded_labels = set()
            # Never use label -1
            excluded_labels.add(-1)
            excluded_labels.add("-1")
            for label_str, mapped_value in mapping.items():
                if mapped_value == "Insufficient follow-up or missing data":
                    # Add both string and numeric representations for flexible matching
                    excluded_labels.add(label_str)  # Keep original string
                    try:
                        # Also add integer representation if possible
                        excluded_labels.add(int(label_str))
                    except ValueError:
                        pass
                    try:
                        # Also add float representation if possible
                        excluded_labels.add(float(label_str))
                    except ValueError:
                        pass
            
            if excluded_labels:
                task_exclusions[task_name] = excluded_labels
        
        return task_exclusions, task_table_map
    except Exception as e:
        print(f"Error loading task mappings from {valid_tasks_json_path}: {e}")
        return {}, {}


# Global BigQuery client (initialized once per process)
_bq_client = None

def get_bq_client():
    global _bq_client
    if _bq_client is None:
        project_id = "som-nero-plevriti-deidbdf"
        
        # 1. Broaden the scope to cloud-platform
        # 2. Let the library find the JSON file automatically (it checks ADC paths)
        credentials, auth_project_id = default(
            scopes=["https://www.googleapis.com/auth/cloud-platform"]
        )
        
        # Use the project_id you want for data, but ensure credentials have a quota project
        _bq_client = bigquery.Client(
            project=project_id, 
            credentials=credentials
        )
    return _bq_client


def nifti_path_to_blob_path(path_str, bucket_name, prefix):
    """
    Convert a nifti_path value to the GCS blob path (same logic as download_subsampled_ct.py).
    path_str may be local path, bucket-relative path, or filename.
    Returns the blob path string (relative to bucket).
    """
    if not path_str or (isinstance(path_str, float) and np.isnan(path_str)):
        return None
    path_str = str(path_str).strip()
    if not path_str:
        return None
    # Remove /mnt/ prefix if present
    if path_str.startswith("/mnt/"):
        path_str = path_str[5:]
    # Remove bucket name prefix if present
    if path_str.startswith(f"{bucket_name}/"):
        path_str = path_str[len(bucket_name) + 1:]
    # Already a full bucket-relative path
    if path_str.startswith(prefix):
        return path_str
    # Build from path parts
    parts = path_str.split("/")
    filename = parts[-1]
    if not filename.endswith(".nii.gz"):
        if len(parts) >= 2:
            filename_no_ext = parts[-1].replace(".zip", "")
            bucket_filename = f"{parts[-2]}__{filename_no_ext}.nii.gz"
        else:
            bucket_filename = filename if filename.endswith(".nii.gz") else f"{filename}.nii.gz"
    else:
        bucket_filename = filename
    return f"{prefix}/{bucket_filename}"


def check_nifti_exists_in_bucket(path_str, bucket_name, prefix, bucket):
    """Return True if the nifti_path exists in the GCS bucket (same check as download_subsampled_ct.py)."""
    blob_path = nifti_path_to_blob_path(path_str, bucket_name, prefix)
    if not blob_path:
        return False
    try:
        blob = bucket.blob(blob_path)
        return blob.exists()
    except Exception:
        return False


def filter_person_ids_by_bucket_existence(
    person_ids,
    path_pairs,
    bucket_name=DEFAULT_GCS_BUCKET_NAME,
    prefix=DEFAULT_GCS_PREFIX,
):
    """
    Filter person_ids to only those who have at least one nifti_path that exists in the GCS bucket.
    path_pairs: iterable of (person_id, nifti_path) with non-null, non-empty nifti_path.
    Returns set of person_ids (int and str) for which at least one path exists in bucket.
    """
    if not person_ids or not path_pairs:
        return set()
    # Deduplicate by (person_id, path) and collect unique paths to minimize GCS calls
    pairs = []
    seen = set()
    for pid, path in path_pairs:
        if path is None or (isinstance(path, float) and np.isnan(path)):
            continue
        path = str(path).strip()
        if not path:
            continue
        key = (str(pid), path)
        if key in seen:
            continue
        seen.add(key)
        pairs.append((pid, path))
    if not pairs:
        return set()
    try:
        client = storage.Client()
        bucket = client.bucket(bucket_name)
    except Exception as e:
        print(f"  Warning: Could not create GCS client for bucket existence check: {e}")
        return set()
    # Check each unique path once
    path_to_exists = {}
    for _pid, path in pairs:
        if path not in path_to_exists:
            path_to_exists[path] = check_nifti_exists_in_bucket(path, bucket_name, prefix, bucket)
    # person_id is "present" if any of their paths exist
    result = set()
    for pid, path in pairs:
        if path_to_exists.get(path, False):
            result.add(pid)
            try:
                result.add(int(pid))
                result.add(str(pid))
            except (ValueError, TypeError):
                result.add(pid)
                result.add(str(pid))
    return result


def fetch_person_id_nifti_paths_from_bq(person_ids, dataset_id, table_name):
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
        project_id = "som-nero-plevriti-deidbdf"
        full_table_id = f"{project_id}.{dataset_id}.{table_name}"
        batch_size = 10000
        pairs = []
        for i in range(0, len(person_ids_int), batch_size):
            batch = person_ids_int[i : i + batch_size]
            query = f"""
                SELECT person_id, nifti_path
                FROM `{full_table_id}`
                WHERE person_id IN UNNEST(@person_ids)
                AND nifti_path IS NOT NULL
                AND TRIM(CAST(nifti_path AS STRING)) != ''
            """
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


def check_ct_available_batch(person_ids, dataset_id='vista_bench_v1_1', table_name=None, local_bq_data_dir=None):
    """
    Batch check if CT scans are available for multiple person_ids.
    When local_bq_data_dir is set, reads from vista_bench/bigquery_data_2_3/<table_name> (same name as BQ table).
    Otherwise uses BigQuery (see commented-out code below).

    Returns a set of person_ids that have nifti_path or local_path available.
    """
    if not person_ids or table_name is None:
        return set()

    # Normalize requested person_ids to set of int (and str for lookup)
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
    requested_set = set()
    for pid in person_ids_list:
        try:
            requested_set.add(int(pid))
            requested_set.add(str(pid))
        except (ValueError, TypeError):
            requested_set.add(pid)
            requested_set.add(str(pid))

    # --- Local: read from base_path/bigquery_data_2_3/<table_name> (file has no extension) ---
    # if local_bq_data_dir is not None:
    #     try:
    #         data_dir = Path(local_bq_data_dir)
    #         table_path = data_dir / table_name
    #         if not table_path.exists():
    #             print(f"  Warning: Local table file not found: {table_path}")
    #             return set()
    #         df = pd.read_csv(table_path, sep=None, engine="python", on_bad_lines="warn")
    #         pid_col = next((c for c in df.columns if c.lower() == "person_id"), None)
    #         if pid_col is None:
    #             return set()
    #         path_col = next((c for c in df.columns if c.lower() == "nifti_path"), None)
    #         if path_col is None:
    #             path_col = next((c for c in df.columns if c.lower() == "local_path"), None)
    #         if path_col is None:
    #             return set()
    #         has_path = df[path_col].notna() & (df[path_col].astype(str).str.strip() != "")
    #         df_with_ct = df.loc[has_path]
    #         if df_with_ct.empty:
    #             return set()
    #         available = set(df_with_ct[pid_col].dropna().unique())
    #         out = set()
    #         for pid in available:
    #             try:
    #                 pint = int(pid)
    #                 if pint in requested_set or str(pid) in requested_set:
    #                     out.add(pint)
    #                     out.add(str(pid))
    #             except (ValueError, TypeError):
    #                 if pid in requested_set or str(pid) in requested_set:
    #                     out.add(pid)
    #                     out.add(str(pid))
    #         return out
    #     except Exception as e:
    #         print(f"  Warning: Error reading local BigQuery data: {e}")
    #         return set()

    # --- BigQuery (commented out; uncomment and remove/switch local branch to use BQ later) ---
    # BigQuery person_id is INTEGER; convert to INT64 before querying.
    #
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
    batch_size = 10000
    available_person_ids = set()
    try:
        client = get_bq_client()
        project_id = "som-nero-plevriti-deidbdf"
        full_table_id = f"{project_id}.{dataset_id}.{table_name}"
        for i in range(0, len(person_ids_int), batch_size):
            batch = person_ids_int[i:i + batch_size]
            query = f"""
                SELECT DISTINCT person_id
                FROM `{full_table_id}`
                WHERE person_id IN UNNEST(@person_ids)
                AND nifti_path IS NOT NULL
                AND TRIM(CAST(nifti_path AS STRING)) != ''
            """
            job_config = bigquery.QueryJobConfig(
                query_parameters=[bigquery.ArrayQueryParameter("person_ids", "INT64", batch)]
            )
            result = client.query(query, job_config=job_config).to_dataframe()
            if len(result) > 0:
                for pid in result['person_id'].dropna():
                    available_person_ids.add(int(pid))
                    available_person_ids.add(str(pid))
        return available_person_ids
    except Exception as e:
        for i in range(0, len(person_ids_int), batch_size):
            batch = person_ids_int[i:i + batch_size]
            try:
                person_ids_in_clause = ", ".join(str(x) for x in batch)
                query_fallback = f"""
                    SELECT DISTINCT person_id
                    FROM `{full_table_id}`
                    WHERE person_id IN ({person_ids_in_clause})
                    AND nifti_path IS NOT NULL
                    AND TRIM(CAST(nifti_path AS STRING)) != ''
                """
                result = client.query(query_fallback).to_dataframe()
                if len(result) > 0:
                    for pid in result['person_id'].dropna():
                        available_person_ids.add(int(pid))
                        available_person_ids.add(str(pid))
            except Exception as e2:
                print(f"  Warning: Error in batch CT availability check (fallback): {e2}")
        return available_person_ids
    except Exception as e:
        print(f"  Warning: Error in batch CT availability check: {e}")
        return set()

    return set()


def process_one_csv(args):
    """Worker: read one CSV, sample up to target_n person_ids that have CT (nifti_path), write _subsampled.csv.
    Only counts nifti_path as present if the CT scan actually exists in the GCS bucket (same check as download_subsampled_ct).
    No class balancing - representative of the dataset.
    If selected_person_ids is provided (for related tasks, e.g. died_of_cancer_2_yr reusing 1_yr patients),
    filter to those person_ids and keep only rows with CT - same patients across the group."""
    csv_path, target_n, excluded_labels, overwrite, bq_dataset_id, task_table_map, selected_person_ids, local_bq_data_dir, gcs_bucket_name, gcs_prefix = args
    csv_path = Path(csv_path)

    if csv_path.stem.endswith("_subsampled"):
        return ("skip", str(csv_path), "already subsampled")
    
    # Check if output file already exists
    new_path = csv_path.with_name(f"{csv_path.stem}_subsampled.csv")
    if new_path.exists() and not overwrite:
        return ("skip", str(csv_path), f"output file {new_path.name} already exists (use overwrite=True to replace)")

    # Get task name from CSV filename (stem)
    task_name = csv_path.stem
    bq_table_name = task_table_map.get(task_name)
    
    if bq_table_name is None:
        return ("skip", str(csv_path), f"no BigQuery table mapping found for task '{task_name}'")

    try:
        df = pd.read_csv(csv_path, sep=None, engine="python", on_bad_lines="warn")

        label_col = next((c for c in df.columns if c.lower() == "label"), None)
        if label_col is None:
            return ("skip", str(csv_path), f"no 'label' col. cols={list(df.columns)}")

        # Filter out rows where label maps to "Insufficient follow-up or missing data"
        if excluded_labels:
            def is_excluded_label(label_val):
                """Check if a label value should be excluded."""
                if pd.isna(label_val):
                    return False
                if label_val in excluded_labels:
                    return True
                if str(label_val) in [str(x) for x in excluded_labels]:
                    return True
                try:
                    numeric_val = float(label_val)
                    if numeric_val in [float(x) for x in excluded_labels if isinstance(x, (int, float))]:
                        return True
                except (ValueError, TypeError):
                    pass
                return False
            
            mask = df[label_col].apply(lambda x: not is_excluded_label(x))
            df = df[mask]
            
            if len(df) == 0:
                return ("skip", str(csv_path), "all rows excluded (insufficient follow-up labels)")

        # Check for person_id column
        person_id_col = next((c for c in df.columns if c.lower() == "person_id"), None)
        if person_id_col is None:
            return ("skip", str(csv_path), f"no 'person_id' col. cols={list(df.columns)}")

        # If reusing patients from a related task (e.g. died_of_cancer_2_yr reusing 1_yr), filter to those first
        if selected_person_ids is not None and len(selected_person_ids) > 0:
            selected_strs = {str(pid) for pid in selected_person_ids}
            df = df[df[person_id_col].astype(str).isin(selected_strs)]
            if len(df) == 0:
                return ("skip", str(csv_path), f"no rows matching pre-selected person_ids for task '{task_name}'")
            print(f"    Filtered to {len(df)} rows matching pre-selected person_ids (same patients as related task)")

        # Get unique person_ids and batch check CT availability (nifti_path in BigQuery)
        unique_person_ids = df[person_id_col].dropna().unique()
        all_person_ids = []
        for pid in unique_person_ids:
            try:
                if isinstance(pid, (np.ndarray, pd.Series)):
                    pid = pid.item() if hasattr(pid, 'item') else pid[0]
                elif isinstance(pid, (list, tuple)):
                    pid = pid[0] if len(pid) > 0 else None
                if pid is not None:
                    try:
                        if np.isnan(float(pid)):
                            continue
                    except (ValueError, TypeError):
                        pass
                    all_person_ids.append(pid)
            except Exception:
                continue

        print(f"    Checking CT availability for {len(all_person_ids)} person_ids (task '{task_name}')...")
        available_person_ids = check_ct_available_batch(
            all_person_ids, dataset_id=bq_dataset_id, table_name=bq_table_name, local_bq_data_dir=local_bq_data_dir
        )
        print(f"    Found {len(available_person_ids)} person_ids with non-null nifti_path in BQ")

        if not available_person_ids:
            return ("skip", str(csv_path), "no person_ids with nifti_path in BigQuery")

        # Build (person_id, nifti_path) pairs: from CSV if path column exists, else from BQ
        path_pairs = []
        nifti_path_col = next((c for c in df.columns if c.lower() == "nifti_path"), None)
        local_path_col = next((c for c in df.columns if c.lower() == "local_path"), None)
        path_col = nifti_path_col or local_path_col
        available_strs = {str(pid) for pid in available_person_ids}
        if path_col:
            df_paths = df[df[person_id_col].astype(str).isin(available_strs)][[person_id_col, path_col]].dropna(subset=[path_col])
            for _, row in df_paths.iterrows():
                path_pairs.append((row[person_id_col], row[path_col]))
        if not path_pairs:
            path_pairs = fetch_person_id_nifti_paths_from_bq(
                list(available_person_ids), bq_dataset_id, bq_table_name
            )

        # Only count as having CT if the nifti file actually exists in the GCS bucket
        available_person_ids = filter_person_ids_by_bucket_existence(
            available_person_ids, path_pairs, bucket_name=gcs_bucket_name, prefix=gcs_prefix
        )
        print(f"    After GCS bucket check: {len(available_person_ids)} person_ids with CT present in bucket")

        if not available_person_ids:
            return ("skip", str(csv_path), "no person_ids with nifti_path present in GCS bucket")

        # Deduplicate by string (BigQuery set may contain both int and str for same id)
        unique_ids_str = list({str(pid) for pid in available_person_ids})
        if selected_person_ids is not None and len(selected_person_ids) > 0:
            # Same patients as related task: keep all rows for person_ids that have CT (no sampling)
            selected_strs = set(unique_ids_str)
        else:
            # New sample: random choice of up to target_n person_ids with CT
            n_select = min(target_n, len(unique_ids_str))
            rng = np.random.default_rng(42)
            selected_strs = set(rng.choice(unique_ids_str, size=n_select, replace=False).tolist())

        # Subset to rows whose person_id (as string) is in selected set
        mask = df[person_id_col].astype(str).isin(selected_strs)
        subsampled_df = df[mask]
        n_person_ids = subsampled_df[person_id_col].nunique()

        subsampled_df.to_csv(new_path, index=False)
        return ("ok", str(csv_path), f"wrote {new_path.name} ({len(subsampled_df)} rows, {n_person_ids} person_ids)")

    except Exception as e:
        return ("err", str(csv_path), repr(e))


def subsample_csvs_parallel(base_path, target_n=50, workers=None, config_path=None, valid_tasks_json_path=None, overwrite=False, bq_dataset_id='vista_bench_v1_1', local_bq_data_dir=None, gcs_bucket_name=DEFAULT_GCS_BUCKET_NAME, gcs_prefix=DEFAULT_GCS_PREFIX):
    """Subsample each task CSV to up to target_n person_ids that have a CT scan (nifti_path) present in the GCS bucket.
    Only person_ids whose nifti_path exists in the bucket (same check as download_subsampled_ct) are counted as having CT.
    When local_bq_data_dir is set (default: base_path/bigquery_data_2_3), reads from local files there (same name as BQ table).
    Uses tasks from config_path (e.g. all_tasks.yaml). No class balancing - representative of the dataset."""
    base_dir = Path(base_path)
    if local_bq_data_dir is None:
        local_bq_data_dir = str(base_dir / "bigquery_data_2_3")
    print(f"Using local BigQuery data from: {local_bq_data_dir}")

    # Load task mappings to exclude "Insufficient follow-up or missing data" labels and get table mappings
    task_exclusions = {}
    task_table_map = {}
    if valid_tasks_json_path:
        task_exclusions, task_table_map = load_task_mappings(valid_tasks_json_path)
        if task_exclusions:
            print(f"Loaded exclusions for {len(task_exclusions)} tasks from {valid_tasks_json_path}")
        if task_table_map:
            print(f"Loaded BigQuery table mappings for {len(task_table_map)} tasks")
    
    # Load tasks from config if provided
    valid_tasks = None
    if config_path:
        valid_tasks = load_tasks_from_config(config_path)
        if valid_tasks:
            print(f"Loaded {len(valid_tasks)} tasks from config: {sorted(valid_tasks)}")
        else:
            print(f"Warning: No tasks found in config or error loading config. Processing all CSVs.")
    
    # Find all CSV files
    all_csv_files = [p for p in base_dir.rglob("*.csv") if not p.stem.endswith("_subsampled")]
    
    # Filter by task names if config provided
    if valid_tasks:
        csv_files = []
        for csv_path in all_csv_files:
            task_name = csv_path.stem
            if task_name in valid_tasks:
                csv_files.append(csv_path)
        print(f"Filtered to {len(csv_files)} CSVs matching tasks from config (out of {len(all_csv_files)} total).")
    else:
        csv_files = all_csv_files

    if not csv_files:
        print("No CSVs found.")
        return

    if workers is None:
        workers = max(1, (os.cpu_count() or 2) - 1)

    # Group related tasks (e.g. died_of_cancer_1_yr, died_of_cancer_2_yr, ...) - same patients across group
    task_names = {p.stem for p in csv_files}
    task_groups = group_related_tasks(task_names)
    if task_groups:
        print(f"Related task groups (same patients per group): {list(task_groups.keys())}")
        for base_name, group_tasks in task_groups.items():
            print(f"  {base_name}: {group_tasks}")

    # First pass: process first task in each group to get the 50 person_ids, then reuse for related tasks
    selected_person_ids_map = {}
    for base_name, group_tasks in task_groups.items():
        first_task = group_tasks[0]
        first_csv = next((p for p in csv_files if p.stem == first_task), None)
        if first_csv is None:
            continue
        print(f"\nProcessing first task in group '{base_name}': {first_task}")
        excluded_labels = task_exclusions.get(first_task, set())
        args = (str(first_csv), target_n, excluded_labels, overwrite, bq_dataset_id, task_table_map, None, local_bq_data_dir, gcs_bucket_name, gcs_prefix)
        result = process_one_csv(args)
        if result[0] == "ok":
            subsampled_path = first_csv.with_name(f"{first_csv.stem}_subsampled.csv")
            if subsampled_path.exists():
                try:
                    sub_df = pd.read_csv(subsampled_path)
                    pid_col = next((c for c in sub_df.columns if c.lower() == "person_id"), None)
                    if pid_col:
                        selected_person_ids_map[base_name] = set(sub_df[pid_col].dropna().astype(str).unique())
                        print(f"  Stored {len(selected_person_ids_map[base_name])} person_ids for reuse in {group_tasks[1:]}")
                except Exception as e:
                    print(f"  Error reading subsampled CSV: {e}")

    # Build task list: all CSVs; for first task in a group skip (already done), for others in group pass selected_person_ids
    tasks = []
    processed_first = set()
    for csv_path in csv_files:
        task_name = csv_path.stem
        excluded_labels = task_exclusions.get(task_name, set())
        selected_person_ids = None
        for base_name, group_tasks in task_groups.items():
            if task_name in group_tasks:
                if task_name == group_tasks[0]:
                    processed_first.add(task_name)
                    break
                selected_person_ids = selected_person_ids_map.get(base_name)
                break
        tasks.append((str(csv_path), target_n, excluded_labels, overwrite, bq_dataset_id, task_table_map, selected_person_ids, local_bq_data_dir, gcs_bucket_name, gcs_prefix))

    remaining = [t for t in tasks if Path(t[0]).stem not in processed_first]
    print(f"\nProcessing {len(remaining)} remaining tasks (up to {target_n} person_ids each, same patients within groups)...")
    ok = skip = err = 0
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(process_one_csv, t) for t in remaining]
        for fut in as_completed(futures):
            status, path, msg = fut.result()
            name = Path(path).name
            if status == "ok":
                ok += 1
                print(f"[OK]   {name}: {msg}")
            elif status == "skip":
                skip += 1
                print(f"[SKIP] {name}: {msg}")
            else:
                err += 1
                print(f"[ERR]  {name}: {msg}")

    print(f"\nDone. OK={ok}, SKIP={skip}, ERR={err}")


if __name__ == "__main__":
    PATH = "/home/rdcunha/vista_project/vista_bench"
    CONFIG_PATH = "/home/rdcunha/vista_project/vista_eval_vlm/configs/all_tasks.yaml"
    VALID_TASKS_JSON_PATH = "/home/rdcunha/vista_project/vista_bench/tasks/valid_tasks.json"
    subsample_csvs_parallel(PATH, target_n=50, workers=1, config_path=CONFIG_PATH, valid_tasks_json_path=VALID_TASKS_JSON_PATH, overwrite=True)
