import os
import pandas as pd
from pathlib import Path
import sys
import csv

# Increase the CSV field size limit for safety
max_int = sys.maxsize
while True:
    try:
        csv.field_size_limit(max_int)
        break
    except OverflowError:
        max_int = int(max_int / 10)

def extract_ct_paths(results_path='/home/dcunhrya/results', output_name='ct_paths.csv'):
    results_base = Path(results_path)
    unique_paths = set()
    
    # 1. Find all CSV files in subfolders containing 'subsampled'
    # This specifically looks at your analyzed results within the subsampled tasks
    csv_files = [
        f for f in results_base.rglob('*.csv') 
        if 'subsampled' in str(f).lower()
    ]
    
    print(f"Found {len(csv_files)} relevant CSV files. Extracting paths...")

    for csv_file in csv_files:
        try:
            # Peek at headers to find the local_path column
            header_df = pd.read_csv(csv_file, sep=None, engine='python', nrows=0)
            
            # Identify the correct column (handling potential casing issues)
            target_col = next((c for c in header_df.columns if c.lower() == 'local_path'), None)
            
            if not target_col:
                # If local_path isn't there, skip this file
                continue
                
            # Read only the local_path column, skipping the heavy text columns
            df = pd.read_csv(
                csv_file, 
                sep=None, 
                engine='python', 
                usecols=[target_col],
                on_bad_lines='warn'
            )
            
            # Add non-NaN paths to our set for automatic deduplication
            valid_paths = df[target_col].dropna().unique()
            unique_paths.update(valid_paths)
            
            print(f"  + Extracted {len(valid_paths)} paths from {csv_file.name}")

        except Exception as e:
            print(f"  - Error processing {csv_file.name}: {e}")

    # 2. Save the final list to /home/dcunhrya/results/ct_paths.csv
    output_path = results_base / output_name
    output_df = pd.DataFrame(sorted(list(unique_paths)), columns=['path'])
    output_df.to_csv(output_path, index=False)
    
    print("-" * 30)
    print(f"Extraction Complete!")
    print(f"Total Unique Paths Found: {len(unique_paths)}")
    print(f"Saved to: {output_path}")

if __name__ == "__main__":
    extract_ct_paths()