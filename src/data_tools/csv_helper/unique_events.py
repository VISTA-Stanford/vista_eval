import os
import json
import yaml
import pandas as pd
import re
from pathlib import Path
from tqdm import tqdm


def count_unique_event_dates(timeline_text):
    """
    Count unique event dates in the patient timeline.
    
    Args:
        timeline_text: The patient timeline string with format [YYYY-MM-DD HH:MM] | ...
        
    Returns:
        int: Number of unique event dates
    """
    if pd.isna(timeline_text) or not timeline_text:
        return 0
    
    text_str = str(timeline_text)
    
    # Pattern to match date-time markers: [YYYY-MM-DD HH:MM]
    # Extract just the date part (YYYY-MM-DD)
    pattern = r'\[(\d{4}-\d{2}-\d{2})\s+\d{2}:\d{2}\]'
    
    # Find all date matches
    dates = re.findall(pattern, text_str)
    
    # Count unique dates
    unique_dates = len(set(dates)) if dates else 0
    
    return unique_dates


def process_all_tasks(config_path):
    """
    Process all tasks defined in the config file and count unique events per person_id.
    
    Args:
        config_path: Path to the YAML config file
    """
    # Load config
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    
    base_path = Path(cfg['paths']['base_dir'])
    results_dir = Path(cfg['paths']['results_dir'])
    valid_tasks_path = base_path / cfg['paths']['valid_tasks']
    
    # Load valid tasks
    with open(valid_tasks_path, 'r') as f:
        valid_tasks = json.load(f)
    
    # Create a mapping from task_name to task_info
    task_map = {task['task_name']: task for task in valid_tasks}
    
    # Get tasks from config
    tasks = cfg.get('tasks', [])
    
    if not tasks:
        print("No tasks found in config.")
        return
    
    print(f"Processing {len(tasks)} tasks...")
    
    all_results = []
    
    for task_name in tqdm(tasks, desc="Processing tasks"):
        if task_name not in task_map:
            print(f"Warning: Task '{task_name}' not found in valid_tasks.json. Skipping.")
            continue
        
        task_info = task_map[task_name]
        source_csv = task_info['task_source_csv']
        
        # Construct CSV path (same logic as run.py)
        csv_path = base_path / source_csv / f"{task_name}_subsampled_no_img_report.csv"
        
        if not csv_path.exists():
            print(f"Warning: CSV file not found at {csv_path}. Skipping task '{task_name}'.")
            continue
        
        print(f"\nProcessing task: {task_name}")
        print(f"  CSV path: {csv_path}")
        
        try:
            # Load CSV
            df = pd.read_csv(csv_path)
            print(f"  Loaded {len(df)} rows")
            
            # Check for required columns
            if 'person_id' not in df.columns:
                print(f"  Warning: 'person_id' column not found. Skipping task '{task_name}'.")
                continue
            
            # Find patient timeline column (case-insensitive)
            timeline_col = next((c for c in df.columns if 'patient_string' in c.lower() or 'patient_timeline' in c.lower()), None)
            
            if timeline_col is None:
                print(f"  Warning: Patient timeline column not found. Skipping task '{task_name}'.")
                continue
            
            print(f"  Using timeline column: {timeline_col}")
            
            # Count unique events for each person_id
            print(f"  Counting unique events per person_id...")
            df['unique_events'] = df[timeline_col].apply(count_unique_event_dates)
            
            # Group by person_id and get unique events
            # If a person_id appears multiple times, we'll take the max (or could aggregate differently)
            person_events = df.groupby('person_id')['unique_events'].max().reset_index()
            person_events['task'] = task_name
            
            # Add to results
            all_results.append(person_events[['person_id', 'unique_events', 'task']])
            
            print(f"  Found {len(person_events)} unique person_ids")
            print(f"  Total unique events range: {person_events['unique_events'].min()} - {person_events['unique_events'].max()}")
            
        except Exception as e:
            print(f"  Error processing task '{task_name}': {e}")
            continue
    
    # Combine all results
    if not all_results:
        print("\nNo results to save.")
        return
    
    final_df = pd.concat(all_results, ignore_index=True)
    
    # Save to results directory
    output_path = results_dir / 'all_unique_events.csv'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    final_df.to_csv(output_path, index=False)
    print(f"\n✓ Saved results to {output_path}")
    print(f"  Total rows: {len(final_df)}")
    print(f"  Unique person_ids: {final_df['person_id'].nunique()}")
    print(f"  Tasks processed: {final_df['task'].nunique()}")
    print(f"  Unique events range: {final_df['unique_events'].min()} - {final_df['unique_events'].max()}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Count unique events per person_id for all tasks")
    parser.add_argument("--config", type=str, default="/home/rdcunha/vista_project/vista_eval_vlm/configs/all_tasks.yaml",
                       help="Path to config YAML file")
    args = parser.parse_args()
    
    process_all_tasks(args.config)
