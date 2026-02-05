import os
import pandas as pd
from pathlib import Path
import sys
import csv
import yaml

# Increase the CSV field size limit to handle large clinical text fields
max_int = sys.maxsize
while True:
    try:
        csv.field_size_limit(max_int)
        break
    except OverflowError:
        max_int = int(max_int / 10)

def load_tasks_from_config(config_path):
    """
    Load task list from YAML config file.
    
    Args:
        config_path: Path to YAML config file
    
    Returns:
        set: Set of task names
    """
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            tasks = config.get('tasks', [])
            return set(tasks)
    except Exception as e:
        print(f"Error loading config from {config_path}: {e}")
        return set()

def analyze_subsampled_patients(base_path, output_txt="subsampled_patient_summary.txt", config_path=None):
    base_dir = Path(base_path)
    
    # Load tasks from config if provided
    valid_tasks = None
    if config_path:
        valid_tasks = load_tasks_from_config(config_path)
        if valid_tasks:
            print(f"Loaded {len(valid_tasks)} tasks from config: {sorted(valid_tasks)}")
        else:
            print(f"Warning: No tasks found in config or error loading config. Processing all subsampled CSVs.")
    
    # Target only files that HAVE 'subsampled' in the name
    all_csv_files = [f for f in base_dir.rglob('*.csv') if 'subsampled' in f.name.lower()]
    
    # Filter by task names if config provided
    if valid_tasks:
        csv_files = []
        for csv_path in all_csv_files:
            # Extract task name from filename (remove _subsampled suffix)
            task_name = csv_path.stem.replace('_subsampled', '')
            if task_name in valid_tasks:
                csv_files.append(csv_path)
        print(f"Filtered to {len(csv_files)} CSVs matching tasks from config (out of {len(all_csv_files)} total).")
    else:
        csv_files = all_csv_files
    
    per_task_results = []
    global_all_patients = set()
    global_labeled_patients = set()
    
    print(f"Found {len(csv_files)} subsampled files to analyze...")

    for csv_file in csv_files:
        try:
            # Step 1: Peek at headers to find columns to exclude
            header_df = pd.read_csv(csv_file, sep=None, engine='python', nrows=0)
            
            # Filter out the heavy text columns to save memory and avoid "field limit" errors
            cols_to_use = [
                c for c in header_df.columns 
                if 'patient_string' not in c.lower() and 'patient_timeline' not in c.lower()
            ]
            
            # Step 2: Read only the necessary columns
            df = pd.read_csv(
                csv_file, 
                sep=None, 
                engine='python', 
                usecols=cols_to_use,
                on_bad_lines='warn'
            )
            
            # Normalize column names for processing
            df.columns = [c.strip().lower() for c in df.columns]
            
            # Column mapping based on your VISTA bench structure
            id_col = 'person_id'
            label_col = 'local_path'
            
            if id_col not in df.columns or label_col not in df.columns:
                print(f"Skipping {csv_file.name}: Required columns ({id_col}/{label_col}) not found.")
                continue
            
            # Calculate unique patients
            unique_pids = set(df[id_col].dropna().unique())
            
            # Calculate unique patients with a valid label path
            labeled_pids = set(df[df[label_col].notna()][id_col].unique())
            
            per_task_results.append({
                'task': csv_file.stem,
                'total_unique': len(unique_pids),
                'labeled_unique': len(labeled_pids)
            })
            
            # Aggregate for global stats
            global_all_patients.update(unique_pids)
            global_labeled_patients.update(labeled_pids)
            
            print(f"  Processed {csv_file.name}")

        except Exception as e:
            print(f"  Error processing {csv_file.name}: {e}")

    # Step 3: Write results to text file
    with open(output_txt, 'w') as f:
        f.write("VISTA Bench Subsampled Patient Analysis\n")
        f.write("="*45 + "\n\n")
        
        f.write(f"{'Subtask Name':<45} | {'Total Patients':<15} | {'Labeled Patients':<15}\n")
        f.write("-" * 80 + "\n")
        
        for res in per_task_results:
            f.write(f"{res['task']:<45} | {res['total_unique']:<15} | {res['labeled_unique']:<15}\n")
            
        f.write("\n" + "="*45 + "\n")
        f.write("GLOBAL TOTALS (Unique across all subsampled files)\n")
        f.write("="*45 + "\n")
        f.write(f"Total Unique Patients:         {len(global_all_patients)}\n")
        f.write(f"Total Unique Labeled Patients: {len(global_labeled_patients)}\n")

    print(f"\nAnalysis complete. Results saved to {output_txt}")

if __name__ == "__main__":
    VISTA_PATH = '/home/dcunhrya/vista_bench'
    CONFIG_PATH = '/home/dcunhrya/vista_eval/configs/all_tasks.yaml'
    analyze_subsampled_patients(VISTA_PATH, config_path=CONFIG_PATH)