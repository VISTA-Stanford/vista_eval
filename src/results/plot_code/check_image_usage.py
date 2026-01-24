import os
import yaml
import pandas as pd
from pathlib import Path


def check_image_usage(config_path='/home/dcunhrya/vista_eval/configs/all_tasks.yaml'):
    """
    Check image usage statistics for tasks defined in the config.
    For each results CSV, counts how many rows with non-null nifti_path had used_image == 1.
    
    Args:
        config_path: Path to the YAML config file
    """
    # Load config
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    
    results_dir = Path(cfg['paths']['results_dir'])
    base_path = Path(cfg['paths']['base_dir'])
    valid_tasks_path = base_path / cfg['paths']['valid_tasks']
    
    # Load valid tasks to get task_source_csv mapping
    import json
    with open(valid_tasks_path, 'r') as f:
        valid_tasks = json.load(f)
    
    # Create a mapping from task_name to task_info
    task_map = {task['task_name']: task for task in valid_tasks}
    
    # Get tasks from config
    tasks = cfg.get('tasks', [])
    
    if not tasks:
        print("No tasks found in config.")
        return
    
    print(f"Checking image usage for {len(tasks)} tasks...")
    print(f"Results directory: {results_dir}\n")
    
    # Store results
    results_summary = []
    
    # Find all result CSV files
    result_files = list(results_dir.rglob("*_results_*.csv"))
    
    if not result_files:
        print("No result CSV files found.")
        return
    
    print(f"Found {len(result_files)} result file(s)\n")
    
    # Process each file
    for file_path in result_files:
        # Extract path components: results / source_folder / task_name / model_name / file
        relative_parts = file_path.relative_to(results_dir).parts
        
        if len(relative_parts) < 4:
            continue
        
        source_folder = relative_parts[0]
        task_name = relative_parts[1]
        model_name = relative_parts[2]
        filename = relative_parts[3]
        
        # Extract experiment name from filename
        import re
        match = re.search(r'_results_(.+)\.csv$', filename)
        experiment = match.group(1) if match else 'unknown'
        
        # Skip 'no_image' experiment since no images are used by definition
        if experiment == 'no_image':
            continue
        
        # Only process tasks from config
        if task_name not in tasks:
            continue
        
        try:
            df = pd.read_csv(file_path)
            
            if df.empty:
                continue
            
            # Check for required columns
            if 'nifti_path' not in df.columns or 'used_image' not in df.columns:
                print(f"  ⚠ {task_name} | {model_name} | {experiment}: Missing required columns")
                continue
            
            # Count rows with non-null nifti_path
            has_nifti = df['nifti_path'].notna()
            total_with_nifti = has_nifti.sum()
            
            # Count rows with non-null nifti_path AND used_image == 1
            used_image = (has_nifti & (df['used_image'] == 1)).sum()
            
            # Total rows
            total_rows = len(df)
            
            # Calculate percentage
            percentage = (used_image / total_with_nifti * 100) if total_with_nifti > 0 else 0
            
            results_summary.append({
                'source': source_folder,
                'task': task_name,
                'model': model_name,
                'experiment': experiment,
                'total_rows': total_rows,
                'rows_with_nifti_path': total_with_nifti,
                'rows_with_used_image_1': used_image,
                'percentage': percentage
            })
            
            print(f"  ✓ {task_name} | {model_name} | {experiment}:")
            print(f"      Total rows: {total_rows}")
            print(f"      Rows with nifti_path: {total_with_nifti}")
            print(f"      Rows with used_image=1: {used_image} ({percentage:.1f}%)")
            print()
            
        except Exception as e:
            print(f"  ✗ Error processing {file_path}: {e}")
            continue
    
    if not results_summary:
        print("No valid results found.")
        return
    
    # Create summary DataFrame
    summary_df = pd.DataFrame(results_summary)
    
    # Print summary statistics
    print("=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)
    
    # Group by task and experiment
    print("\nBy Task and Experiment:")
    print("-" * 80)
    for (task, experiment), group in summary_df.groupby(['task', 'experiment']):
        total_nifti = group['rows_with_nifti_path'].sum()
        total_used = group['rows_with_used_image_1'].sum()
        pct = (total_used / total_nifti * 100) if total_nifti > 0 else 0
        print(f"{task} | {experiment}: {total_used}/{total_nifti} ({pct:.1f}%)")
    
    # Group by experiment
    print("\nBy Experiment (across all tasks):")
    print("-" * 80)
    for experiment, group in summary_df.groupby('experiment'):
        total_nifti = group['rows_with_nifti_path'].sum()
        total_used = group['rows_with_used_image_1'].sum()
        pct = (total_used / total_nifti * 100) if total_nifti > 0 else 0
        print(f"{experiment}: {total_used}/{total_nifti} ({pct:.1f}%)")
    
    # Overall statistics
    print("\nOverall Statistics:")
    print("-" * 80)
    total_nifti_all = summary_df['rows_with_nifti_path'].sum()
    total_used_all = summary_df['rows_with_used_image_1'].sum()
    overall_pct = (total_used_all / total_nifti_all * 100) if total_nifti_all > 0 else 0
    print(f"Total rows with nifti_path: {total_nifti_all}")
    print(f"Total rows with used_image=1: {total_used_all}")
    print(f"Overall percentage: {overall_pct:.1f}%")
    
    # Save summary to CSV
    output_path = results_dir / 'image_usage_summary.csv'
    summary_df.to_csv(output_path, index=False)
    print(f"\n✓ Summary saved to: {output_path}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Check image usage statistics from result CSVs")
    parser.add_argument("--config", type=str, default="/home/dcunhrya/vista_eval/configs/all_tasks.yaml",
                       help="Path to config YAML file")
    args = parser.parse_args()
    
    check_image_usage(args.config)
