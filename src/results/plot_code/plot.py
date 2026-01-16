import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from results.results_analyzer import DynamicResultsAnalyzer

def generate_comparison_plots(results_path='/home/dcunhrya/results'):
    results_base = Path(results_path)
    all_data = []

    # 1. Collect all analyzed data
    print("Gathering data from analyzed results...")
    for analyzed_file in results_base.rglob("results_analyzed.csv"):
        # Path structure: results / source_folder / task_name / model_name / file
        parts = analyzed_file.parts
        relative_parts = analyzed_file.relative_to(results_base).parts
        
        source_folder = relative_parts[0]
        task_name = relative_parts[1]
        model_name = relative_parts[2]

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

    if not all_data:
        print("No analyzed data found. Run the analyzer first.")
        return

    full_df = pd.DataFrame(all_data)

    # 2. Generate one plot per 'task_source_csv' (Source)
    for source in full_df['Source'].unique():
        source_df = full_df[full_df['Source'] == source]
        
        plt.figure(figsize=(50, 10))
        sns.set_style("whitegrid")
        
        plot = sns.barplot(
            data=source_df, 
            x='Subtask', 
            y='Accuracy', 
            hue='Model',
            palette='viridis'
        )

        plt.title(f'Model Comparison: {source}', fontsize=15)
        plt.ylabel('Accuracy (%)')
        plt.xlabel('Subtasks')
        plt.ylim(0, 100)
        plt.legend(title='VLM Model', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.xticks(rotation=45)
        
        # Add value labels on top of bars
        for p in plot.patches:
            plot.annotate(format(p.get_height(), '.1f'), 
                   (p.get_x() + p.get_width() / 2., p.get_height()), 
                   ha = 'center', va = 'center', 
                   xytext = (0, 9), 
                   textcoords = 'offset points')

        plt.tight_layout()
        
        # Save plot in the source folder
        # save_path = results_base / source / f"{source}_comparison_plot.png"
        save_path = os.path.join('/home/dcunhrya/vista_eval/figures', f"{source}_comparison_plot.pdf")
        plt.savefig(save_path)
        print(f"Plot saved for {source} at {save_path}")
        plt.close()

if __name__ == "__main__":
    analyzer = DynamicResultsAnalyzer()
    analyzer.analyze_all_models()
    generate_comparison_plots()