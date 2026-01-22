# import os
# import pandas as pd
# import csv
# import sys
# from pathlib import Path

# # Increase the limit to handle very large clinical text fields
# # We set it to the maximum allowed by the system
# csv.field_size_limit(sys.maxsize)

# def subsample_csvs(base_path, target_n=10):
#     base_dir = Path(base_path)
    
#     for csv_path in base_dir.rglob('*.csv'):
#         if csv_path.stem.endswith('_subsampled'):
#             continue
            
#         print(f"Processing: {csv_path.name}")
        
#         try:
#             # Using sep=None with engine='python' to auto-detect Tab vs Comma
#             # We also pass low_memory=False to handle mixed types in large files
#             df = pd.read_csv(csv_path, sep=None, engine='python', on_bad_lines='warn')
            
#             # Find the label column (case-insensitive)
#             label_col = next((c for c in df.columns if c.lower() == 'label'), None)
            
#             if label_col is None:
#                 print(f"  Skipping: No 'label' column found in {csv_path.name}")
#                 print(f"  Available columns: {list(df.columns)}")
#                 continue
                
#             # Perform stratified sampling
#             subsampled_df = df.groupby(label_col, group_keys=False).apply(
#                 lambda x: x.sample(n=min(len(x), target_n), random_state=42)
#             )
            
#             new_path = csv_path.with_name(f"{csv_path.stem}_subsampled.csv")
            
#             # Save the result
#             subsampled_df.to_csv(new_path, index=False)
#             print(f"  Saved: {new_path.name} ({len(subsampled_df)} rows)")
            
#         except Exception as e:
#             print(f"  Error processing {csv_path.name}: {e}")

# if __name__ == "__main__":
#     PATH = '/home/dcunhrya/vista_bench'
#     subsample_csvs(PATH)

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
    """Worker: read one CSV, stratified sample with CT availability constraint, write _subsampled.csv."""
    csv_path, target_n = args
    csv_path = Path(csv_path)

    if csv_path.stem.endswith("_subsampled"):
        return ("skip", str(csv_path), "already subsampled")

    try:
        df = pd.read_csv(csv_path, sep=None, engine="python", on_bad_lines="warn")

        label_col = next((c for c in df.columns if c.lower() == "label"), None)
        if label_col is None:
            return ("skip", str(csv_path), f"no 'label' col. cols={list(df.columns)}")

        # Check for local_path column
        local_path_col = next((c for c in df.columns if c.lower() == "local_path"), None)
        if local_path_col is None:
            return ("skip", str(csv_path), f"no 'local_path' col. cols={list(df.columns)}")

        # Function to sample with CT availability constraint
        def sample_with_ct_constraint(group):
            # Separate rows with and without available CT scans
            group_with_ct = []
            group_without_ct = []
            
            for idx, row in group.iterrows():
                local_path_val = row[local_path_col]
                if pd.notna(local_path_val) and isinstance(local_path_val, str):
                    if check_ct_available(local_path_val):
                        group_with_ct.append(idx)
                    else:
                        group_without_ct.append(idx)
                else:
                    group_without_ct.append(idx)
            
            # Only sample from rows with available CT scans in the bucket
            # This ensures 100% of selected entries have corresponding files in the bucket
            # Ensure at least 2 with CT scans (if available), up to target_n
            min_ct_samples = 2
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
        
        subsampled_df = (
            df.groupby(label_col, group_keys=False)
              .apply(sample_with_ct_constraint)
        )

        new_path = csv_path.with_name(f"{csv_path.stem}_subsampled.csv")
        subsampled_df.to_csv(new_path, index=False)

        return ("ok", str(csv_path), f"wrote {new_path.name} ({len(subsampled_df)} rows)")

    except Exception as e:
        return ("err", str(csv_path), repr(e))


def subsample_csvs_parallel(base_path, target_n=10, workers=None, config_path=None):
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

    tasks = [(str(p), target_n) for p in csv_files]

    ok = skip = err = 0
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(process_one_csv, t) for t in tasks]

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
    subsample_csvs_parallel(PATH, target_n=10, workers=4, config_path=CONFIG_PATH)