import os
import pandas as pd
import csv
import sys
import yaml
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from google.cloud import storage

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


def check_ct_available(local_path_str, bucket_name='su-vista-uscentral1', 
                       prefix='chaudhari_lab/ct_data/ct_scans/vista/nov25'):
    """
    Check if a CT scan is available in GCP bucket based on local_path.
    Returns True if available, False otherwise.
    """
    try:
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        
        # Transform local_path to bucket path (same logic as download_subsampled_ct.py)
        parts = local_path_str.split('/')
        filename_no_ext = parts[-1].replace('.zip', '')
        bucket_filename = f"{parts[-2]}__{filename_no_ext}.nii.gz"
        blob_path = f"{prefix}/{bucket_filename}"
        
        # Check if blob exists in bucket
        blob = bucket.blob(blob_path)
        return blob.exists()
    except Exception:
        # If there's any error, assume not available
        return False


def process_one_csv(args):
    """Worker: read one CSV, filter to keep only rows with CT scans available in bucket, write _all_ct.csv."""
    csv_path = args
    csv_path = Path(csv_path)

    if csv_path.stem.endswith("_all_ct"):
        return ("skip", str(csv_path), "already processed")

    try:
        df = pd.read_csv(csv_path, sep=None, engine="python", on_bad_lines="warn")

        # Check for local_path column
        local_path_col = next((c for c in df.columns if c.lower() == "local_path"), None)
        if local_path_col is None:
            return ("skip", str(csv_path), f"no 'local_path' col. cols={list(df.columns)}")

        # Filter to keep only rows with CT scans available in the bucket
        rows_with_ct = []
        for idx, row in df.iterrows():
            local_path_val = row[local_path_col]
            if pd.notna(local_path_val) and isinstance(local_path_val, str):
                if check_ct_available(local_path_val):
                    rows_with_ct.append(idx)
        
        # Create filtered dataframe with only rows that have CT scans available
        filtered_df = df.loc[rows_with_ct].copy()

        new_path = csv_path.with_name(f"{csv_path.stem}_all_ct.csv")
        filtered_df.to_csv(new_path, index=False)

        return ("ok", str(csv_path), f"wrote {new_path.name} ({len(filtered_df)} rows, {len(df)} original)")

    except Exception as e:
        return ("err", str(csv_path), repr(e))


def filter_csvs_by_ct_availability(base_path, workers=None, config_path=None):
    """
    Process CSVs to keep only entries with CT scans available in GCP bucket.
    
    Args:
        base_path: Base directory to search for CSV files
        workers: Number of worker processes (default: cpu_count - 1)
        config_path: Path to YAML config file with tasks list (optional)
    """
    base_dir = Path(base_path)
    
    # Load tasks from config if provided
    valid_tasks = None
    if config_path:
        valid_tasks = load_tasks_from_config(config_path)
        if valid_tasks:
            print(f"Loaded {len(valid_tasks)} tasks from config: {sorted(valid_tasks)}")
        else:
            print(f"Warning: No tasks found in config or error loading config. Processing all CSVs.")
    
    # Find all CSV files
    all_csv_files = [p for p in base_dir.rglob("*.csv") if not p.stem.endswith("_all_ct")]
    
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

    ok = skip = err = 0
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(process_one_csv, str(p)) for p in csv_files]

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
    PATH = "/home/dcunhrya/vista_bench"
    CONFIG_PATH = "/home/dcunhrya/vista_eval/configs/all_tasks.yaml"
    filter_csvs_by_ct_availability(PATH, workers=4, config_path=CONFIG_PATH)
