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

csv.field_size_limit(sys.maxsize)

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
        return {}, {}, {}


# Global BigQuery client (initialized once per process)
_bq_client = None

def get_bq_client():
    """Get or create BigQuery client."""
    global _bq_client
    if _bq_client is None:
        project_id = "som-nero-plevriti-deidbdf"
        _bq_client = bigquery.Client(project=project_id)
    return _bq_client

def check_ct_available_batch(person_ids, dataset_id='vista_bench_v1_1', table_name=None):
    """
    Batch check if CT scans are available in BigQuery for multiple person_ids.
    Returns a set of person_ids that have nifti_path available.
    
    Args:
        person_ids: List/set of person_ids to check (can be strings or integers)
        dataset_id: BigQuery dataset ID (default: 'vista_bench_v1_1')
        table_name: BigQuery table name containing person_id and nifti_path (from task_source_csv)
    
    Returns:
        set: Set of person_ids that have nifti_path available
    """
    if not person_ids or table_name is None:
        return set()
    
    try:
        client = get_bq_client()
        project_id = "som-nero-plevriti-deidbdf"
        full_table_id = f"{project_id}.{dataset_id}.{table_name}"
        
        # Convert person_ids to list and filter out NaN values
        # Handle both arrays and individual values
        person_ids_list = []
        for pid in person_ids:
            # Convert to scalar if needed
            try:
                if isinstance(pid, (pd.Series, np.ndarray)):
                    if len(pid) > 0:
                        pid = pid.item() if hasattr(pid, 'item') else pid[0]
                    else:
                        continue
                elif isinstance(pid, (list, tuple)):
                    if len(pid) > 0:
                        pid = pid[0]
                    else:
                        continue
                
                # Check if not NaN (using Python native check)
                if pid is None:
                    continue
                try:
                    if np.isnan(float(pid)):
                        continue
                except (ValueError, TypeError):
                    pass  # Not a number, that's okay
                
                person_ids_list.append(pid)
            except Exception:
                continue
        
        if not person_ids_list:
            return set()
        
        # Try to determine if person_ids are numeric or string
        # Convert all to string first for the query
        person_ids_str = [str(pid) for pid in person_ids_list]
        
        # Build query with IN clause for batch checking
        # BigQuery has a limit of 10,000 items in IN clause, so we may need to batch
        batch_size = 10000
        available_person_ids = set()
        
        for i in range(0, len(person_ids_str), batch_size):
            batch = person_ids_str[i:i + batch_size]
            
            # Use UNNEST with array parameter for safer query (avoids SQL injection)
            # Convert batch to array format for BigQuery
            query = f"""
                SELECT DISTINCT person_id
                FROM `{full_table_id}`
                WHERE person_id IN UNNEST(@person_ids)
                AND nifti_path IS NOT NULL
                AND nifti_path != ''
            """
            
            try:
                # Use array parameter for IN clause
                job_config = bigquery.QueryJobConfig(
                    query_parameters=[
                        bigquery.ArrayQueryParameter("person_ids", "STRING", batch)
                    ]
                )
                result = client.query(query, job_config=job_config).to_dataframe()
                if len(result) > 0:
                    # Convert results to set, handling both string and numeric types
                    for pid in result['person_id'].dropna():
                        # Add both string and numeric representations to handle type mismatches
                        available_person_ids.add(str(pid))
                        try:
                            available_person_ids.add(int(pid))
                        except (ValueError, TypeError):
                            pass
            except Exception as e:
                # If array parameter fails, try with individual parameters (fallback)
                try:
                    # Fallback: use individual IN clause (less safe but works)
                    person_ids_in_clause = "', '".join(batch)
                    query_fallback = f"""
                        SELECT DISTINCT person_id
                        FROM `{full_table_id}`
                        WHERE CAST(person_id AS STRING) IN ('{person_ids_in_clause}')
                        AND nifti_path IS NOT NULL
                        AND nifti_path != ''
                    """
                    result = client.query(query_fallback).to_dataframe()
                    if len(result) > 0:
                        for pid in result['person_id'].dropna():
                            available_person_ids.add(str(pid))
                            try:
                                available_person_ids.add(int(pid))
                            except (ValueError, TypeError):
                                pass
                except Exception as e2:
                    print(f"  Warning: Error in batch CT availability check (both methods failed): {e2}")
                    continue
        
        return available_person_ids
        
    except Exception as e:
        print(f"  Warning: Error in batch CT availability check: {e}")
        return set()


def process_one_csv(args):
    """Worker: read one CSV, stratified sample with CT availability constraint, write _subsampled.csv."""
    csv_path, target_n, excluded_labels, overwrite, bq_dataset_id, task_table_map, selected_person_ids = args
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
            # Convert label column to handle both string and numeric label values
            # Try to match labels by converting types appropriately
            def is_excluded_label(label_val):
                """Check if a label value should be excluded."""
                if pd.isna(label_val):
                    return False
                # Check direct match
                if label_val in excluded_labels:
                    return True
                # Check string conversion
                if str(label_val) in [str(x) for x in excluded_labels]:
                    return True
                # Check numeric conversion if label is numeric
                try:
                    numeric_val = float(label_val)
                    if numeric_val in [float(x) for x in excluded_labels if isinstance(x, (int, float))]:
                        return True
                except (ValueError, TypeError):
                    pass
                return False
            
            # Apply filtering
            mask = df[label_col].apply(lambda x: not is_excluded_label(x))
            df = df[mask]
            
            if len(df) == 0:
                return ("skip", str(csv_path), "all rows excluded (insufficient follow-up labels)")

        # Check for person_id column
        person_id_col = next((c for c in df.columns if c.lower() == "person_id"), None)
        if person_id_col is None:
            return ("skip", str(csv_path), f"no 'person_id' col. cols={list(df.columns)}")

        # If selected_person_ids is provided, filter to those person_ids first
        use_selected_person_ids = False
        if selected_person_ids is not None and len(selected_person_ids) > 0:
            # Filter to only rows with person_ids in the selected set
            df_filtered = df[df[person_id_col].isin(selected_person_ids)].copy()
            
            if len(df_filtered) > 0:
                # Use the filtered dataframe for subsampling
                df = df_filtered
                use_selected_person_ids = True
                print(f"    Filtered to {len(df)} rows matching pre-selected person_ids for task '{task_name}'")
            else:
                # If none of the selected person_ids exist in this CSV, continue with normal subsampling
                print(f"    No pre-selected person_ids found in task '{task_name}', using normal subsampling")
                use_selected_person_ids = False
        
        # Batch check CT availability for all person_ids in the dataframe
        # This is much faster than checking one-by-one
        print(f"    Checking CT availability for {len(df)} rows...")
        # Get unique person_ids and convert to list of scalars
        unique_person_ids = df[person_id_col].dropna().unique()
        # Convert numpy array to list of Python scalars
        all_person_ids = []
        for pid in unique_person_ids:
            try:
                # Convert to scalar
                if isinstance(pid, (np.ndarray, pd.Series)):
                    pid = pid.item() if hasattr(pid, 'item') else pid[0]
                elif isinstance(pid, (list, tuple)):
                    pid = pid[0] if len(pid) > 0 else None
                if pid is not None:
                    all_person_ids.append(pid)
            except Exception:
                continue
        
        available_person_ids = check_ct_available_batch(all_person_ids, dataset_id=bq_dataset_id, table_name=bq_table_name)
        print(f"    Found {len(available_person_ids)} person_ids with CT scans available")
        
        # Create a set for fast lookup (include both string and numeric representations)
        available_person_ids_lookup = set()
        for pid in available_person_ids:
            available_person_ids_lookup.add(str(pid))
            try:
                available_person_ids_lookup.add(int(pid))
            except (ValueError, TypeError):
                pass
        
        # Function to sample with CT availability constraint
        def sample_with_ct_constraint(group):
            # Separate rows with and without available CT scans using the pre-computed lookup
            group_with_ct = []
            group_without_ct = []
            
            for idx, row in group.iterrows():
                # Get person_id value - iterrows() should give us a scalar, but be safe
                try:
                    person_id_val = row[person_id_col]
                    
                    # Convert to Python native type to avoid pandas array issues
                    # Use .values to get numpy array, then take first element if needed
                    if isinstance(person_id_val, pd.Series):
                        person_id_val = person_id_val.values[0] if len(person_id_val.values) > 0 else None
                    elif isinstance(person_id_val, np.ndarray):
                        person_id_val = person_id_val.item() if person_id_val.size == 1 else person_id_val.flat[0]
                    elif isinstance(person_id_val, (list, tuple)):
                        person_id_val = person_id_val[0] if len(person_id_val) > 0 else None
                    
                    # Check if value is None or NaN (using Python native check, not pd.isna)
                    if person_id_val is None:
                        group_without_ct.append(idx)
                        continue
                    
                    # Try to check for NaN using float conversion
                    try:
                        float_val = float(person_id_val)
                        if np.isnan(float_val):
                            group_without_ct.append(idx)
                            continue
                    except (ValueError, TypeError):
                        # Not a number, that's okay - continue with the value
                        pass
                    
                    # Check against pre-computed lookup (much faster than BigQuery query)
                    # Check string version first
                    person_id_str = str(person_id_val)
                    is_available = person_id_str in available_person_ids_lookup
                    
                    # Also check numeric version if string check fails
                    if not is_available:
                        try:
                            person_id_int = int(person_id_val)
                            is_available = person_id_int in available_person_ids_lookup
                        except (ValueError, TypeError):
                            pass
                    
                    if is_available:
                        group_with_ct.append(idx)
                    else:
                        group_without_ct.append(idx)
                        
                except Exception as e:
                    # If anything goes wrong, skip this row
                    group_without_ct.append(idx)
                    continue
            
            # Only sample from rows with available CT scans in BigQuery
            # This ensures 100% of selected entries have corresponding nifti_path in BigQuery
            # Ensure at least 2 with CT scans (if available), up to target_n
            min_ct_samples = 6
            selected_indices = []
            
            if len(group_with_ct) >= min_ct_samples:
                # We have enough CT scans, sample up to target_n from CT-available group only
                sample_size = min(target_n, len(group_with_ct))
                selected_indices = pd.Series(group_with_ct).sample(
                    n=sample_size,
                    random_state=42
                ).tolist()
            elif len(group_with_ct) > 0:
                # We have some CT scans but less than min_ct_samples, take all of them
                # (This ensures we still get some samples even if fewer than 2 are available)
                selected_indices = group_with_ct.copy()
            else:
                # No CT scans available for this label group
                # Return empty selection - this will result in fewer samples for this label
                # but ensures all selected entries have CT scans available
                selected_indices = []
            
            if len(selected_indices) > 0:
                return group.loc[selected_indices]
            else:
                # Return empty dataframe with same columns if no CT scans available
                return group.iloc[0:0]
        
        # Always do stratified sampling (even with pre-selected person_ids, we still want stratified by label)
        subsampled_df = (
            df.groupby(label_col, group_keys=False)
              .apply(sample_with_ct_constraint)
        )
        
        if use_selected_person_ids:
            print(f"    Subsampled to {len(subsampled_df)} rows using pre-selected person_ids for task '{task_name}'")

        # new_path was already defined above in the overwrite check
        # Overwrite the file if it exists and overwrite=True, or create new file
        subsampled_df.to_csv(new_path, index=False)

        return ("ok", str(csv_path), f"wrote {new_path.name} ({len(subsampled_df)} rows)")

    except Exception as e:
        return ("err", str(csv_path), repr(e))


def subsample_csvs_parallel(base_path, target_n=10, workers=None, config_path=None, valid_tasks_json_path=None, overwrite=False, bq_dataset_id='vista_bench_v1_1'):
    base_dir = Path(base_path)
    
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
            # Extract task name from filename (stem without any suffixes)
            task_name = csv_path.stem
            if task_name in valid_tasks:
                csv_files.append(csv_path)
        print(f"Filtered to {len(csv_files)} CSVs matching tasks from config (out of {len(all_csv_files)} total).")
    else:
        csv_files = all_csv_files

    if not csv_files:
        print("No CSVs found.")
        return

    # Good default: leave 1 core free
    if workers is None:
        workers = max(1, (os.cpu_count() or 2) - 1)

    print(f"Found {len(csv_files)} CSVs. Using {workers} worker processes.")

    # Group related tasks (tasks that differ only by year)
    task_names = {csv_path.stem for csv_path in csv_files}
    task_groups = group_related_tasks(task_names)
    
    if task_groups:
        print(f"Found {len(task_groups)} task groups: {list(task_groups.keys())}")
        for base_name, group_tasks in task_groups.items():
            print(f"  {base_name}: {group_tasks}")

    # Process task groups: subsample first task, then reuse person_ids for related tasks
    selected_person_ids_map = {}  # Maps task_name -> set of selected person_ids
    
    # First pass: Process first task in each group to get selected person_ids
    for base_name, group_tasks in task_groups.items():
        first_task = group_tasks[0]  # First task (lowest year)
        
        # Find CSV for first task
        first_csv = None
        for csv_path in csv_files:
            if csv_path.stem == first_task:
                first_csv = csv_path
                break
        
        if first_csv is None:
            continue
        
        # Process first task normally to get selected person_ids
        print(f"\nProcessing first task in group '{base_name}': {first_task}")
        excluded_labels = task_exclusions.get(first_task, set())
        args = (str(first_csv), target_n, excluded_labels, overwrite, bq_dataset_id, task_table_map, None)
        
        # We need to extract person_ids from the subsampled result
        # For now, we'll process it and then read the result to get person_ids
        result = process_one_csv(args)
        
        if result[0] == "ok":
            # Read the subsampled CSV to get selected person_ids
            subsampled_path = first_csv.with_name(f"{first_csv.stem}_subsampled.csv")
            if subsampled_path.exists():
                try:
                    subsampled_df = pd.read_csv(subsampled_path)
                    person_id_col = next((c for c in subsampled_df.columns if c.lower() == "person_id"), None)
                    if person_id_col:
                        selected_person_ids = set(subsampled_df[person_id_col].dropna().unique())
                        selected_person_ids_map[base_name] = selected_person_ids
                        print(f"  Selected {len(selected_person_ids)} person_ids from {first_task}")
                except Exception as e:
                    print(f"  Error reading subsampled CSV for {first_task}: {e}")

    # Create tasks with excluded labels for each CSV
    # For related tasks, pass the selected person_ids
    tasks = []
    processed_first_tasks = set()
    
    for csv_path in csv_files:
        task_name = csv_path.stem
        excluded_labels = task_exclusions.get(task_name, set())
        
        # Check if this task belongs to a group
        selected_person_ids = None
        for base_name, group_tasks in task_groups.items():
            if task_name in group_tasks:
                if task_name == group_tasks[0]:
                    # First task already processed, skip it in the main loop
                    processed_first_tasks.add(task_name)
                    continue
                else:
                    # Use person_ids from first task in group
                    selected_person_ids = selected_person_ids_map.get(base_name)
                    break
        
        tasks.append((str(csv_path), target_n, excluded_labels, overwrite, bq_dataset_id, task_table_map, selected_person_ids))

    # Process remaining tasks (excluding first tasks in groups which were already processed)
    remaining_tasks = [t for t in tasks if Path(t[0]).stem not in processed_first_tasks]
    
    if remaining_tasks:
        print(f"\nProcessing {len(remaining_tasks)} remaining tasks...")
        ok = skip = err = 0
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futures = [ex.submit(process_one_csv, t) for t in remaining_tasks]

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
    else:
        print("\nNo remaining tasks to process.")


if __name__ == "__main__":
    PATH = "/home/dcunhrya/vista_bench"
    CONFIG_PATH = "/home/dcunhrya/vista_eval/configs/all_tasks.yaml"
    VALID_TASKS_JSON_PATH = "/home/dcunhrya/vista_bench/tasks/valid_tasks.json"
    subsample_csvs_parallel(PATH, target_n=20, workers=1, config_path=CONFIG_PATH, valid_tasks_json_path=VALID_TASKS_JSON_PATH, overwrite=True)