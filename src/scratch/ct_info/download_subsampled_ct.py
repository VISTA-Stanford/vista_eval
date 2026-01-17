import os
import pandas as pd
from pathlib import Path
from google.cloud import storage

def download_ct_scans(csv_path='/home/dcunhrya/results/ct_paths.csv', 
                      bucket_name='su-vista-uscentral1',
                      prefix='chaudhari_lab/ct_data/ct_scans/vista/nov25',
                      dry_run=True):
    """
    Checks or downloads NIfTI files from GCP.
    dry_run=True: Only counts present/missing files without downloading.
    """
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found.")
        return
    
    df = pd.read_csv(csv_path)
    paths = df['path'].unique().tolist() # Ensure we only check unique entries
    
    stats = {"present": 0, "missing": 0, "already_on_vm": 0}
    
    print(f"{' [DRY RUN MODE] ' if dry_run else ' [LIVE DOWNLOAD MODE] '}")
    print(f"Processing {len(paths)} unique paths...")

    for local_path_str in paths:
        try:
            # 1. Filename Transformation Logic
            parts = local_path_str.split('/')
            filename_no_ext = parts[-1].replace('.zip', '')
            bucket_filename = f"{parts[-2]}__{filename_no_ext}.nii.gz"
            blob_path = f"{prefix}/{bucket_filename}"
            
            # 2. Local Path Setup
            local_file_path = Path(local_path_str)
            local_download_path = local_file_path.with_suffix('').with_suffix('.nii.gz')

            # 3. Check Bucket
            blob = bucket.blob(blob_path)
            exists_in_bucket = blob.exists()

            if exists_in_bucket:
                stats["present"] += 1
                if local_download_path.exists():
                    stats["already_on_vm"] += 1
                
                if not dry_run and not local_download_path.exists():
                    # Ensure directory exists before downloading
                    local_download_path.parent.mkdir(parents=True, exist_ok=True)
                    print(f"  [DOWNLOADING] {bucket_filename}")
                    blob.download_to_filename(str(local_download_path))
            else:
                stats["missing"] += 1
                print(f"  [MISSING] {blob_path}")

        except Exception as e:
            print(f"  [ERROR] Failed to process {local_path_str}: {e}")

    # 4. Final Summary
    print("\n" + "="*40)
    print(" SCAN SUMMARY ")
    print("="*40)
    print(f"Total Paths in CSV:    {len(paths)}")
    print(f"Present in Bucket:     {stats['present']}")
    print(f"Missing from Bucket:   {stats['missing']}")
    print(f"Already on VM:         {stats['already_on_vm']}")
    
    if len(paths) > 0:
        availability = (stats['present'] / len(paths)) * 100
        print(f"Bucket Availability:   {availability:.2f}%")
    print("="*40)

if __name__ == "__main__":
    # Toggle dry_run to False when you are ready to pull the data
    download_ct_scans(dry_run=True)