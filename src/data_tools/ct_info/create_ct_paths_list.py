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
    all_records = []
    
    # 1. Find all CSV files in subfolders containing 'subsampled'
    csv_files = [
        f for f in results_base.rglob('*.csv') 
        if 'subsampled' in str(f).lower()
    ]
    
    print(f"Found {len(csv_files)} relevant CSV files. Extracting metadata...")

    for csv_file in csv_files:
        try:
            # Peek at headers to find the relevant columns
            header_df = pd.read_csv(csv_file, sep=None, engine='python', nrows=0)
            
            # A. Identify Path Column
            path_col = next((c for c in header_df.columns if c.lower() == 'local_path'), None)
            
            # B. Identify ID Column (check for person_id or patient_id)
            id_col = next((c for c in header_df.columns if c.lower() in ['person_id', 'patient_id']), None)
            
            if not path_col or not id_col:
                # print(f"  [SKIP] {csv_file.name} - Missing required columns.")
                continue
                
            # Read only the necessary columns (skipping heavy text)
            df = pd.read_csv(
                csv_file, 
                sep=None, 
                engine='python', 
                usecols=[path_col, id_col],
                on_bad_lines='warn'
            )
            
            # Rename for consistency
            df = df.rename(columns={path_col: 'path', id_col: 'person_id'})
            
            # Add Task Name (derived from filename, removing extensions/suffixes)
            # e.g., 'pneumonitis_subsampled_results.csv' -> 'pneumonitis_subsampled_results'
            task_name = csv_file.stem.replace('_results', '').replace('_subsampled', '')
            df['task'] = task_name
            
            # Filter valid paths only
            df = df.dropna(subset=['path'])
            
            if not df.empty:
                all_records.append(df)
                print(f"  + Extracted {len(df)} rows from {csv_file.name}")

        except Exception as e:
            print(f"  - Error processing {csv_file.name}: {e}")

    # 2. Compile and Save
    if all_records:
        # Concatenate all dataframes
        final_df = pd.concat(all_records, ignore_index=True)
        
        # Deduplicate: Keep unique combinations of Task + Person + Path
        # If the exact same image is used for the exact same person in the same task multiple times, drop duplicates.
        final_df = final_df.drop_duplicates(subset=['task', 'person_id', 'path'])
        
        # Sort for readability
        final_df = final_df.sort_values(by=['task', 'person_id'])
        
        output_path = results_base / output_name
        final_df.to_csv(output_path, index=False)
        
        print("-" * 30)
        print(f"Extraction Complete!")
        print(f"Total Records: {len(final_df)}")
        print(f"Unique Paths:  {final_df['path'].nunique()}")
        print(f"Saved to: {output_path}")
    else:
        print("No records found.")

if __name__ == "__main__":
    extract_ct_paths()