import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from results.results_analyzer import DynamicResultsAnalyzer

def generate_filtered_comparison_plots(results_path='/home/dcunhrya/results', min_models=2):
    """
    Generate comparison plots but only for subtasks that have results from more than min_models.
    
    Args:
        results_path: Base path to search for results_analyzed.csv files
        min_models: Minimum number of models required to plot a subtask (default: 2, so > 2 means 3+)
    """
    results_base = Path(results_path)
    all_data = []

    # 1. Collect all analyzed data
    print("Gathering data from analyzed results...")
    for analyzed_file in results_base.rglob("results_analyzed.csv"):
        # Path structure: results / source_folder / task_name / model_name / file
        parts = analyzed_file.parts
        relative_parts = analyzed_file.relative_to(results_base).parts
        
        if len(relative_parts) < 3:
            continue
            
        source_folder = relative_parts[0]
        task_name = relative_parts[1]
        model_name = relative_parts[2]

        try:
            df = pd.read_csv(analyzed_file)
            
            # Calculate accuracy for this specific subtask/model
            if 'is_correct' in df.columns:
                accuracy = df['is_correct'].mean() * 100
            else:
                # Fallback if is_correct wasn't calculated
                accuracy = (df['cleaned_response'].astype(str).str.lower() == 
                            df['mapped_label'].astype(str).str.lower()).mean() * 100
            
            all_data.append({
                'Source': source_folder,
                'Subtask': task_name,
                'Model': model_name,
                'Accuracy': accuracy
            })
        except Exception as e:
            print(f"Error processing {analyzed_file}: {e}")
            continue

    if not all_data:
        print("No analyzed data found. Run the analyzer first.")
        return

    full_df = pd.DataFrame(all_data)

    # 2. Filter subtasks: only keep those with more than min_models
    print(f"\nFiltering subtasks: keeping only those with > {min_models} models...")
    
    # Count models per subtask per source
    filtered_data = []
    for source in full_df['Source'].unique():
        source_df = full_df[full_df['Source'] == source]
        
        # Count unique models per subtask
        subtask_model_counts = source_df.groupby('Subtask')['Model'].nunique()
        
        # Filter: only subtasks with > min_models
        valid_subtasks = subtask_model_counts[subtask_model_counts > min_models].index
        
        if len(valid_subtasks) > 0:
            print(f"  {source}: {len(valid_subtasks)}/{len(subtask_model_counts)} subtasks have > {min_models} models")
            filtered_source_df = source_df[source_df['Subtask'].isin(valid_subtasks)]
            filtered_data.append(filtered_source_df)
        else:
            print(f"  {source}: No subtasks with > {min_models} models (skipping)")
    
    if not filtered_data:
        print(f"\nNo subtasks found with > {min_models} models. Nothing to plot.")
        return
    
    filtered_df = pd.concat(filtered_data, ignore_index=True)

    # 3. Generate one plot per 'task_source_csv' (Source)
    for source in filtered_df['Source'].unique():
        source_df = filtered_df[filtered_df['Source'] == source]
        
        # Sort subtasks for better visualization
        source_df = source_df.sort_values('Subtask')
        
        plt.figure(figsize=(24, 12))
        sns.set_style("whitegrid")
        
        plot = sns.barplot(
            data=source_df, 
            x='Subtask', 
            y='Accuracy', 
            hue='Model',
            palette='deep'
        )

        plt.title(f'Model Comparison: {source} (Subtasks with > {min_models} models)', fontsize=15)
        plt.ylabel('Accuracy (%)')
        plt.xlabel('Subtasks')
        plt.ylim(0, 100)
        plt.legend(title='VLM Model', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.xticks(rotation=90)
        
        # Add value labels on top of bars
        # for p in plot.patches:
        #     if not pd.isna(p.get_height()):
        #         plot.annotate(format(p.get_height(), '.1f'), 
        #                (p.get_x() + p.get_width() / 2., p.get_height()), 
        #                ha = 'center', va = 'center', 
        #                xytext = (0, 9), 
        #                textcoords = 'offset points')

        plt.tight_layout()
        
        # Save plot
        save_path = os.path.join('/home/dcunhrya/vista_eval/figures/eval_results', f"{source}_filtered_comparison_plot.pdf")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"Plot saved for {source} at {save_path}")
        plt.close()

if __name__ == "__main__":
    analyzer = DynamicResultsAnalyzer()
    analyzer.analyze_all_models()
    generate_filtered_comparison_plots(min_models=2)  # Only plot subtasks with > 2 models (i.e., 3+)
