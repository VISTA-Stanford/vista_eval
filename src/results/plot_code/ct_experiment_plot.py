import os
import re
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

from results.results_analyzer import _extract_answer, is_answer_correct, map_label_to_answer


def _clean_question(text):
    """
    Filters the question to keep only the part before 'OPTIONS'.
    """
    if pd.isna(text):
        return ""
    # Split on 'OPTIONS' (case-insensitive check is safer)
    parts = re.split(r'\bOPTIONS\b', str(text), flags=re.IGNORECASE)
    return parts[0].strip()

def extract_experiment_from_filename(filename):
    """
    Extract experiment name from filename like:
    task_name_results_experiment.csv
    Returns experiment name or 'default' if pattern doesn't match
    """
    match = re.search(r'_results_(.+)\.csv$', filename)
    if match:
        return match.group(1)
    return 'default'

def parse_experiment_comments(config_path):
    """Parse YAML file to extract experiment names and their comments."""
    experiment_display_mapping = {}
    try:
        with open(config_path, 'r') as f:
            lines = f.readlines()
        
        in_experiments_section = False
        for line in lines:
            stripped = line.strip()
            # Check if we're entering the experiments section
            if stripped.startswith('experiments:'):
                in_experiments_section = True
                continue
            
            # Check if we've left the experiments section (next top-level key)
            if in_experiments_section:
                if stripped and not stripped.startswith('#') and not stripped.startswith('-') and ':' in stripped:
                    # We've hit a new top-level section
                    break
                
                # Parse experiment lines: "- experiment_name # comment"
                if stripped.startswith('-'):
                    # Remove the leading '-'
                    content = stripped[1:].strip()
                    # Split on '#' to separate name from comment
                    if '#' in content:
                        parts = content.split('#', 1)
                        exp_name = parts[0].strip()
                        comment = parts[1].strip()
                        experiment_display_mapping[exp_name] = comment
                    else:
                        # No comment, use the name itself
                        exp_name = content.strip()
                        experiment_display_mapping[exp_name] = exp_name
    except Exception as e:
        print(f"Warning: Could not parse experiment comments: {e}")
    
    return experiment_display_mapping

def get_task_csv_filename(task_name, use_subsampled=False):
    """
    Get the correct CSV filename for a task based on subsample flag.
    
    Args:
        task_name: Name of the task
        use_subsampled: If True, return _subsampled.csv filename, else .csv
    
    Returns:
        str: CSV filename (e.g., 'task_name.csv' or 'task_name_subsampled.csv')
    """
    if use_subsampled:
        return f"{task_name}_subsampled.csv"
    return f"{task_name}.csv"

def calculate_accuracy(df, mapping):
    """
    Calculate accuracy for a dataframe using the same logic as results_analyzer.
    """
    try:
        df = df.copy()
        
        # Check for required columns
        if 'model_response' not in df.columns:
            print("  Warning: 'model_response' column not found")
            return None
        
        # Extract response (primary answer for display)
        df['cleaned_response'] = df['model_response'].apply(_extract_answer)
        
        # Map label if mapping and label column exist (handles int/float labels vs string mapping keys)
        if mapping and 'label' in df.columns:
            df['mapped_label'] = df['label'].apply(lambda lbl: map_label_to_answer(lbl, mapping))
            
            # Accuracy: any candidate (| splits, \boxed{}, <answer>/<label>, last word/phrase) matches
            df['is_correct'] = df.apply(
                lambda x: is_answer_correct(x['model_response'], x['mapped_label']),
                axis=1
            )
            
            return df['is_correct'].mean() * 100
        else:
            # If no mapping, return None (can't calculate accuracy without ground truth)
            print("  Warning: No mapping or 'label' column found, cannot calculate accuracy")
            return None
    except Exception as e:
        print(f"  Error calculating accuracy: {e}")
        import traceback
        traceback.print_exc()
        return None

def generate_experiment_comparison_plots(results_path=None, base_path='/home/dcunhrya/vista_bench', config_path=None):
    """
    Generate comparison plots for experiments across models for each task.

    Args:
        results_path: Base path to search for result CSV files (overridden by config paths.results_dir if present)
        base_path: Base path to load task registry for mappings
        config_path: Optional path to YAML config file to check for subsample flag and filter tasks/models
    """
    if config_path is None:
        config_path = '/home/dcunhrya/vista_eval/configs/all_tasks.yaml'

    # Load config first to get paths.results_dir
    config = None
    use_subsampled = False
    valid_tasks = None
    valid_models = None
    valid_experiments = None
    experiment_display_mapping = {}

    if config_path:
        try:
            import yaml
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
                use_subsampled = config.get('subsample', False)
                # Use results_dir from config if present, else fall back to results_path arg
                config_results_dir = config.get('paths', {}).get('results_dir')
                if config_results_dir is not None:
                    results_path = config_results_dir
                if results_path is None:
                    results_path = '/home/dcunhrya/results'  # final fallback

                # Extract valid tasks from config
                tasks_list = config.get('tasks', [])
                if tasks_list:
                    valid_tasks = set(tasks_list)
                    print(f"Loaded {len(valid_tasks)} tasks from config: {sorted(valid_tasks)}")
                
                # Extract valid models from config
                models_list = config.get('models', [])
                if models_list:
                    # Convert model names to file format (matching run_bq.py logic)
                    valid_models = set()
                    for model in models_list:
                        if isinstance(model, dict) and 'name' in model:
                            model_name = model['name']
                            # Match run_bq.py: model_name.split('/')[-1].replace('/', '_')
                            # This takes the last part after splitting by '/'
                            file_model_name = model_name.split('/')[-1].replace('/', '_')
                            valid_models.add(file_model_name)
                            # Also try the full name with slashes replaced (in case it's stored differently)
                            valid_models.add(model_name.replace('/', '_'))
                            # Also try just the last part without any replacement (fallback)
                            if '/' in model_name:
                                valid_models.add(model_name.split('/')[-1])
                    if valid_models:
                        print(f"Loaded {len(valid_models)} model patterns from config: {sorted(valid_models)}")
                
                # Extract valid experiments from config
                experiments_list = config.get('experiments', [])
                if experiments_list:
                    valid_experiments = set(experiments_list)
                    print(f"Loaded {len(valid_experiments)} experiments from config: {sorted(valid_experiments)}")
                
                # Extract experiment display names from comments
                experiment_display_mapping = parse_experiment_comments(config_path)
        except Exception as e:
            print(f"Warning: Could not load config from {config_path}: {e}")

    if results_path is None:
        results_path = '/home/dcunhrya/results'
    results_base = Path(results_path)
    base_path_obj = Path(base_path)

    if use_subsampled:
        print("Note: Using subsampled CSV files (subsample flag is true)")
    
    # Load task registry for mappings
    tasks_json = base_path_obj / 'tasks' / 'valid_tasks.json'
    task_registry = {}
    if tasks_json.exists():
        import json
        with open(tasks_json, 'r') as f:
            tasks_list = json.load(f)
            task_registry = {t['task_name']: t for t in tasks_list}
    
    # Find all experiment result files
    print("Scanning for experiment result files...")
    all_result_files = list(results_base.rglob("*_results_*.csv"))
    
    # Filter by tasks, models, and experiments from config
    result_files = []
    if valid_tasks or valid_models or valid_experiments:
        for file_path in all_result_files:
            # Extract path components: results / source_folder / task_name / model_name / file
            relative_parts = file_path.relative_to(results_base).parts
            
            if len(relative_parts) < 4:
                continue
            
            task_name = relative_parts[1]
            model_name = relative_parts[2]
            filename = relative_parts[3]
            
            # Extract experiment name from filename
            experiment = extract_experiment_from_filename(filename)
            
            # Check if task matches config
            if valid_tasks and task_name not in valid_tasks:
                continue
            
            # Check if model matches config
            if valid_models:
                model_matches = False
                # First try exact match
                if model_name in valid_models:
                    model_matches = True
                else:
                    # Also check if model_name ends with any of the valid patterns
                    # (for cases where full path is stored like "OpenGVLab_InternVL3_5-8B")
                    for valid_model in valid_models:
                        # Check if model_name ends with the valid_model (with or without separator)
                        if model_name == valid_model or model_name.endswith('_' + valid_model) or model_name.endswith('/' + valid_model):
                            model_matches = True
                            break
                
                if not model_matches:
                    continue
            
            # Check if experiment matches config
            if valid_experiments and experiment not in valid_experiments:
                continue
            
            result_files.append(file_path)
        
        print(f"Filtered to {len(result_files)} files matching config (out of {len(all_result_files)} total).")
    else:
        result_files = all_result_files
        print(f"Found {len(result_files)} experiment result file(s) (no filtering applied)")
    
    if not result_files:
        print("No experiment result files found matching config criteria.")
        return
    
    # Organize data by (source, task, model, experiment)
    data_dict = {}
    
    print("\nProcessing files...")
    for file_path in result_files:
        # Extract path components: results / source_folder / task_name / model_name / file
        relative_parts = file_path.relative_to(results_base).parts
        
        if len(relative_parts) < 4:
            continue
        
        source_folder = relative_parts[0]
        task_name = relative_parts[1]
        model_name = relative_parts[2]
        filename = relative_parts[3]
        
        # Extract experiment name from filename
        experiment = extract_experiment_from_filename(filename)
        
        # Get mapping for this task
        mapping = {}
        if task_name in task_registry:
            mapping = task_registry[task_name].get('mapping', {})
        
        try:
            df = pd.read_csv(file_path)
            
            if df.empty:
                print(f"  Warning: Empty file {file_path}")
                continue
            
            # Calculate accuracy
            accuracy = calculate_accuracy(df, mapping)
            
            if accuracy is not None:
                key = (source_folder, task_name)
                if key not in data_dict:
                    data_dict[key] = []
                
                # Get display name for experiment (use comment if available, otherwise use experiment name)
                experiment_display = experiment_display_mapping.get(experiment, experiment)
                
                data_dict[key].append({
                    'Source': source_folder,
                    'Task': task_name,
                    'Model': model_name,
                    'Experiment': experiment_display,
                    'Accuracy': accuracy
                })
                # print(f"  ✓ {task_name} | {model_name} | {experiment}: {accuracy:.2f}%")
            else:
                print(f"  ✗ Skipping {file_path.name}: Could not calculate accuracy")
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    if not data_dict:
        print("No valid data found for plotting.")
        return
    
    # Generate plots for each (source, task) combination
    print(f"\nGenerating plots for {len(data_dict)} task(s)...")
    
    for (source_folder, task_name), data_list in data_dict.items():
        df = pd.DataFrame(data_list)
        
        # Check if we have multiple experiments and models
        unique_experiments = df['Experiment'].unique()
        unique_models = df['Model'].unique()
        
        if len(unique_experiments) < 2:
            print(f"Skipping {task_name}: Need at least 2 experiments (found {len(unique_experiments)})")
            continue
        
        if len(unique_models) < 1:
            print(f"Skipping {task_name}: No models found")
            continue
        
        # Create plot
        plt.figure(figsize=(max(12, len(unique_experiments) * 2), 8))
        sns.set_style("whitegrid")
        
        # Create bar plot with experiments on x-axis, models as hue
        plot = sns.barplot(
            data=df,
            x='Experiment',
            y='Accuracy',
            hue='Model',
            palette='deep',
            order=sorted(unique_experiments)
        )
        
        plt.title(f'Experiment Comparison: {task_name}\n({source_folder})', fontsize=14, fontweight='bold')
        plt.ylabel('Accuracy (%)', fontsize=12)
        plt.xlabel('Experiment', fontsize=12)
        plt.ylim(0, 100)
        plt.legend(title='Model', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.xticks(rotation=45, ha='right')
        
        # Add value labels on top of bars
        # for container in plot.containers:
        #     plot.bar_label(container, fmt='%.1f', label_type='edge', padding=3)
        
        plt.tight_layout()
        
        # Save plot - organize by source_folder (overall task name) in figures directory
        # Structure: figures/{source_folder}/{task_name}_experiment_comparison.pdf
        base_figures_dir = Path('/home/dcunhrya/vista_eval/figures')
        save_dir = base_figures_dir / source_folder
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Sanitize filename (only task_name, not source_folder since it's in the path)
        safe_task_name = task_name.replace('/', '_').replace(' ', '_')
        save_path = save_dir / f"{safe_task_name}_experiment_comparison.pdf"
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  Plot saved: {save_path}")
        plt.close()
        
        # Also create a summary table
        summary_df = df.pivot_table(
            index='Model',
            columns='Experiment',
            values='Accuracy',
            aggfunc='mean'
        )
        print(f"\n  Summary for {task_name}:")
        print(summary_df.to_string())
        print()

if __name__ == "__main__":
    # Default config path
    default_config = '/home/dcunhrya/vista_eval/configs/all_tasks.yaml'
    generate_experiment_comparison_plots(config_path=default_config)
