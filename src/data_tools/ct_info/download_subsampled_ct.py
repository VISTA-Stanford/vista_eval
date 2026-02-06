import os
import pandas as pd
import yaml
import csv
import sys
from pathlib import Path
from google.cloud import storage
from data_tools.utils.config_utils import load_tasks_from_config

# Increase the limit to handle very large clinical text fields
csv.field_size_limit(sys.maxsize)

def download_ct_scans(base_path='/home/rdcunha/vista_project/vista_bench',
                      bucket_name='su-vista-uscentral1',
                      prefix='chaudhari_lab/ct_data/ct_scans/vista/nov25',
                      dry_run=True,
                      config_path=None,
                      download_base_dir='/home/rdcunha/vista_project/downloaded_ct_scans',
                      file_suffix='_subsampled'):
    """
    Checks or downloads NIfTI files from GCP, reporting Task and Person ID.
    Finds all CSV files with the specified suffix and processes them.
    dry_run=True: Only counts present/missing files without downloading.
    config_path: Optional path to YAML config file. If provided, only processes tasks listed in config.
    download_base_dir: Base directory where files will be downloaded, maintaining bucket structure.
    file_suffix: Suffix to look for in CSV filenames (e.g., '_subsampled' or '_all_ct'). Default: '_subsampled'.
    """
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    
    base_dir = Path(base_path)
    if not base_dir.exists():
        print(f"Error: Base directory {base_path} not found.")
        return
    
    # Load tasks from config if provided
    valid_tasks = None
    if config_path:
        valid_tasks = load_tasks_from_config(config_path)
        if valid_tasks:
            print(f"Loaded {len(valid_tasks)} tasks from config: {sorted(valid_tasks)}")
        else:
            print(f"Warning: No tasks found in config or error loading config. Processing all tasks.")
    
    # Find all CSV files with the specified suffix
    all_csv_files = [p for p in base_dir.rglob("*.csv") if p.stem.endswith(file_suffix)]
    
    if not all_csv_files:
        print(f"No CSV files with suffix '{file_suffix}' found in {base_path}")
        return
    
    # Filter by task names if config provided
    csv_files = []
    if valid_tasks:
        for csv_path in all_csv_files:
            # Extract task name from filename (remove file_suffix suffix)
            task_name = csv_path.stem.replace(file_suffix, '')
            if task_name in valid_tasks:
                csv_files.append(csv_path)
        print(f"Filtered to {len(csv_files)} CSVs with suffix '{file_suffix}' matching tasks from config (out of {len(all_csv_files)} total).")
    else:
        csv_files = all_csv_files
        print(f"Found {len(csv_files)} CSV files with suffix '{file_suffix}'.")
    
    if not csv_files:
        print(f"No matching CSV files with suffix '{file_suffix}' found.")
        return
    
    # Read all matching CSV files and combine
    all_records = []
    for csv_file in csv_files:
        try:
            # Extract task name from filename
            task_name = csv_file.stem.replace(file_suffix, '')
            
            # Read CSV file
            df = pd.read_csv(csv_file, sep=None, engine='python', on_bad_lines='warn')
            
            # Check for required columns - prefer nifti_path, fallback to local_path
            nifti_path_col = next((c for c in df.columns if c.lower() == 'nifti_path'), None)
            local_path_col = next((c for c in df.columns if c.lower() == 'local_path'), None)
            person_id_col = next((c for c in df.columns if c.lower() in ['person_id', 'patient_id']), None)
            
            # Use nifti_path if available, otherwise fallback to local_path
            path_col = nifti_path_col if nifti_path_col else local_path_col
            
            if not path_col:
                print(f"  [SKIP] {csv_file.name}: Missing 'nifti_path' or 'local_path' column")
                continue
            
            if not person_id_col:
                print(f"  [SKIP] {csv_file.name}: Missing 'person_id' or 'patient_id' column")
                continue
            
            # Select and rename columns
            df_selected = df[[path_col, person_id_col]].copy()
            df_selected = df_selected.rename(columns={path_col: 'path', person_id_col: 'person_id'})
            df_selected['task'] = task_name
            
            # Filter valid paths only
            df_selected = df_selected.dropna(subset=['path'])
            
            if not df_selected.empty:
                all_records.append(df_selected)
                print(f"  + Loaded {len(df_selected)} rows from {csv_file.name}")
        
        except Exception as e:
            print(f"  [ERROR] Failed to process {csv_file.name}: {e}")
    
    if not all_records:
        print(f"No valid records found in CSV files with suffix '{file_suffix}'.")
        return
    
    # Combine all dataframes
    df = pd.concat(all_records, ignore_index=True)
    
    # Deduplicate based on path to avoid checking the same file twice
    # We keep the first occurrence of metadata for reporting
    unique_df = df.drop_duplicates(subset=['path'])
    
    stats = {"present": 0, "missing": 0, "already_on_vm": 0}
    
    # Setup download base directory
    download_base = Path(download_base_dir)
    
    print(f"{' [DRY RUN MODE] ' if dry_run else ' [LIVE DOWNLOAD MODE] '}")
    print(f"Processing {len(unique_df)} unique paths...")
    if not dry_run:
        print(f"Files will be downloaded to: {download_base_dir}")

    for _, row in unique_df.iterrows():
        path_str = row['path']
        task_name = row['task']
        person_id = row['person_id']
        
        try:
            # 1. Path Normalization - Handle different nifti_path formats
            # Remove /mnt/ prefix if present
            if path_str.startswith('/mnt/'):
                path_str = path_str[5:]
            
            # Remove bucket name prefix if present
            if path_str.startswith(f'{bucket_name}/'):
                path_str = path_str[len(bucket_name) + 1:]
            
            # Check if path_str is already a bucket-relative path (contains prefix)
            if path_str.startswith(prefix):
                # Already a full bucket path, use it directly
                blob_path = path_str
                # Extract filename for local download path
                bucket_filename = path_str.split('/')[-1]
            else:
                # Extract filename from path (handles both local paths and just filenames)
                parts = path_str.split('/')
                filename = parts[-1]
                
                # If filename doesn't have .nii.gz extension, construct it from parts
                if not filename.endswith('.nii.gz'):
                    # Assume format: {study_uid}__{series_uid}.nii.gz or construct from path
                    if len(parts) >= 2:
                        filename_no_ext = parts[-1].replace('.zip', '')
                        bucket_filename = f"{parts[-2]}__{filename_no_ext}.nii.gz"
                    else:
                        # Just filename, assume it's already correct or needs .nii.gz
                        bucket_filename = filename if filename.endswith('.nii.gz') else f"{filename}.nii.gz"
                else:
                    bucket_filename = filename
                
                blob_path = f"{prefix}/{bucket_filename}"
            
            # 2. Local Path Setup - Use bucket structure
            # Construct path as: download_base_dir/prefix/bucket_filename
            local_download_path = download_base / prefix / bucket_filename

            # 3. Check Bucket
            blob = bucket.blob(blob_path)
            exists_in_bucket = blob.exists()

            if exists_in_bucket:
                stats["present"] += 1
                status_msg = f"[PRESENT] (Task: {task_name}, ID: {person_id})"
                
                if local_download_path.exists():
                    stats["already_on_vm"] += 1
                    # Optional: Comment out to reduce noise if you have many files
                    # print(f"  {status_msg} - Already on VM")
                
                if not dry_run and not local_download_path.exists():
                    local_download_path.parent.mkdir(parents=True, exist_ok=True)
                    print(f"  [DOWNLOADING] {blob_path} -> {local_download_path}")
                    blob.download_to_filename(str(local_download_path))
                elif dry_run:
                     print(f"  {status_msg} - {blob_path}")

            else:
                stats["missing"] += 1
                print(f"  [MISSING] {blob_path} (Task: {task_name}, ID: {person_id})")

        except Exception as e:
            print(f"  [ERROR] Failed to process {path_str}: {e}")

    # 4. Final Summary
    print("\n" + "="*40)
    print(" SCAN SUMMARY ")
    print("="*40)
    print(f"Total Paths in CSV:    {len(unique_df)}")
    print(f"Present in Bucket:     {stats['present']}")
    print(f"Missing from Bucket:   {stats['missing']}")
    print(f"Already on VM:         {stats['already_on_vm']}")
    
    if len(unique_df) > 0:
        availability = (stats['present'] / len(unique_df)) * 100
        print(f"Bucket Availability:   {availability:.2f}%")
    print("="*40)

if __name__ == "__main__":
    # Download all CT scans from nifti_path in _subsampled CSVs chosen from config
    BASE_PATH = "/home/rdcunha/vista_project/vista_bench"
    CONFIG_PATH = "/home/rdcunha/vista_project/vista_eval_vlm/configs/all_tasks.yaml"
    download_ct_scans(
        base_path=BASE_PATH, 
        dry_run=False,  # Set to False to actually download
        config_path=CONFIG_PATH, 
        file_suffix='_subsampled'
    )