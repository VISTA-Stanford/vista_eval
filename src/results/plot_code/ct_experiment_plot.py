import os
import re
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from results.results_analyzer import DynamicResultsAnalyzer

def _extract_answer(text):
    """
    Extracts the core answer from the model's raw output.
    Prioritizes <answer> tags, then Pipe |, then fallback.
    """
    text = str(text).strip()
    
    # 1. Regex for <answer> tags (case-insensitive, handles newlines)
    tag_match = re.search(r'<answer>(.*?)</answer>', text, re.DOTALL | re.IGNORECASE)
    if tag_match:
        return tag_match.group(1).strip()
    
    # 2. Fallback to everything after the Pipe delimiter
    if '|' in text:
        return text.split('|')[-1].strip()
    
    # 3. Final fallback: Cleaned version of original text
    return text

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
    task_name_subsampled_results_experiment.csv
    Returns experiment name or 'default' if pattern doesn't match
    """
    match = re.search(r'_subsampled_results_(.+)\.csv$', filename)
    if match:
        return match.group(1)
    return 'default'

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
        
        # Extract response
        df['cleaned_response'] = df['model_response'].apply(_extract_answer)
        
        # Map label if mapping and label column exist
        if mapping and 'label' in df.columns:
            df['mapped_label'] = df['label'].astype(str).map(mapping)
            
            # Calculate accuracy
            df['is_correct'] = df.apply(
                lambda x: str(x['cleaned_response']).strip().lower() == str(x['mapped_label']).strip().lower() 
                if pd.notna(x['mapped_label']) else False, 
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

def generate_experiment_comparison_plots(results_path='/home/dcunhrya/results', base_path='/home/dcunhrya/vista_bench'):
    """
    Generate comparison plots for experiments across models for each task.
    
    Args:
        results_path: Base path to search for result CSV files
        base_path: Base path to load task registry for mappings
    """
    results_base = Path(results_path)
    base_path_obj = Path(base_path)
    
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
    result_files = list(results_base.rglob("*_subsampled_results_*.csv"))
    
    if not result_files:
        print("No experiment result files found.")
        return
    
    print(f"Found {len(result_files)} experiment result file(s)")
    
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
                
                data_dict[key].append({
                    'Source': source_folder,
                    'Task': task_name,
                    'Model': model_name,
                    'Experiment': experiment,
                    'Accuracy': accuracy
                })
                print(f"  ✓ {task_name} | {model_name} | {experiment}: {accuracy:.2f}%")
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
        for container in plot.containers:
            plot.bar_label(container, fmt='%.1f', label_type='edge', padding=3)
        
        plt.tight_layout()
        
        # Save plot
        save_dir = Path('/home/dcunhrya/vista_eval/figures/experiment_comparisons')
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Sanitize filename
        safe_task_name = task_name.replace('/', '_').replace(' ', '_')
        safe_source = source_folder.replace('/', '_').replace(' ', '_')
        save_path = save_dir / f"{safe_source}_{safe_task_name}_experiment_comparison.pdf"
        
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
    generate_experiment_comparison_plots()
