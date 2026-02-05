import os
import pandas as pd
from pathlib import Path
import sys
import csv

# Robust handling for CSV field size limits
max_int = sys.maxsize
while True:
    try:
        csv.field_size_limit(max_int)
        break
    except OverflowError:
        max_int = int(max_int / 10)

def analyze_patient_data(base_path, output_txt="patient_summary.txt"):
    base_dir = Path(base_path)
    # Find all CSVs recursively, excluding those with 'subsampled'
    csv_files = [f for f in base_dir.rglob('*.csv') if 'subsampled' not in f.name.lower()]
    
    per_task_results = []
    global_all_patients = set()
    global_labeled_patients = set()
    
    print(f"Found {len(csv_files)} files to analyze...")

    for csv_file in csv_files:
        try:
            # STEP 1: Peek at headers to find the column to exclude
            # We read only the first row to determine column names
            header_df = pd.read_csv(csv_file, sep=None, engine='python', nrows=0)
            
            # Filter out 'patient_string' or 'patient_timeline' (case-insensitive)
            # This logic keeps the file handle small
            cols_to_use = [
                c for c in header_df.columns 
                if 'patient_string' not in c.lower() and 'patient_timeline' not in c.lower()
            ]
            
            # STEP 2: Read the actual data using only the filtered columns
            df = pd.read_csv(
                csv_file, 
                sep=None, 
                engine='python', 
                usecols=cols_to_use,
                on_bad_lines='warn'
            )
            
            # Normalize column names for processing
            df.columns = [c.strip().lower() for c in df.columns]
            
            # As per your updated variables
            target_id_col = 'person_id'
            target_label_col = 'local_path'
            
            if target_id_col not in df.columns or target_label_col not in df.columns:
                print(f"Skipping {csv_file.name}: Required columns ({target_id_col}/{target_label_col}) not found.")
                continue
            
            # 1. Total unique person_ids
            all_pids = set(df[target_id_col].dropna().unique())
            
            # 2. Unique person_ids with a valid local_path
            labeled_pids = set(df[df[target_label_col].notna()][target_id_col].unique())
            
            per_task_results.append({
                'task': csv_file.stem,
                'total_unique': len(all_pids),
                'labeled_unique': len(labeled_pids)
            })
            
            # Update global counts
            global_all_patients.update(all_pids)
            global_labeled_patients.update(labeled_pids)
            
            print(f"  Processed {csv_file.name}: {len(all_pids)} patients.")

        except Exception as e:
            print(f"Error processing {csv_file.name}: {e}")

    # Write findings to the text file
    with open(output_txt, 'w') as f:
        f.write("VISTA Bench Patient Analysis Report (Filtered)\n")
        f.write("="*45 + "\n\n")
        
        f.write(f"{'Task Name':<45} | {'Total Patients':<15} | {'Labeled Patients':<15}\n")
        f.write("-" * 80 + "\n")
        
        for res in per_task_results:
            f.write(f"{res['task']:<45} | {res['total_unique']:<15} | {res['labeled_unique']:<15}\n")
            
        f.write("\n" + "="*45 + "\n")
        f.write("GLOBAL TOTALS (Cross-File Unique)\n")
        f.write("="*45 + "\n")
        f.write(f"Total Unique Patients:         {len(global_all_patients)}\n")
        f.write(f"Total Unique Labeled Patients: {len(global_labeled_patients)}\n")

    print(f"\nAnalysis complete. Summary saved to {output_txt}")

if __name__ == "__main__":
    VISTA_PATH = '/home/dcunhrya/vista_bench'
    analyze_patient_data(VISTA_PATH)