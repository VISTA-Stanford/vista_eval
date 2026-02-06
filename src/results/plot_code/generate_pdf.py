import os
import json
import pandas as pd
import yaml
import re
import numpy as np
from pathlib import Path
from collections import defaultdict
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
from reportlab.lib.enums import TA_LEFT, TA_CENTER

from results.results_analyzer import is_answer_correct, map_label_to_answer

# Experiments that include CT scan slices (no_image excluded)
CT_EXPERIMENTS = [
    'axial_1_image',      # 1 middle axial slice
    'all_image',          # 3 slices: sagittal, coronal, axial middle
    'axial_all_image',    # 10 axial slices
    # 'sagittal_all_image', # 10 sagittal slices
    'no_timeline',        # 100 axial slices
    'no_report',          # 50 axial slices
]


def extract_experiment_from_filename(filename):
    """Extract experiment name from filename like: task_name_results_experiment.csv"""
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


def load_config(config_path):
    """Load config and extract models and experiments."""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Extract models
        models_list = config.get('models', [])
        valid_models = []
        model_name_mapping = {}  # Map file model name to display name
        for model in models_list:
            if isinstance(model, dict) and 'name' in model:
                model_name = model['name']
                # Convert to file format (matching run_bq.py logic)
                file_model_name = model_name.split('/')[-1].replace('/', '_')
                valid_models.append(file_model_name)
                model_name_mapping[file_model_name] = model_name
        
        # Extract experiments
        valid_experiments = config.get('experiments', [])
        
        # Extract experiment display names from comments
        experiment_display_mapping = parse_experiment_comments(config_path)
        
        # Extract tasks
        valid_tasks = config.get('tasks', [])

        # Extract paths (results_dir, base_dir)
        paths = config.get('paths', {})
        results_dir = paths.get('results_dir')
        base_dir = paths.get('base_dir')

        return {
            'models': valid_models,
            'model_name_mapping': model_name_mapping,
            'experiments': valid_experiments,
            'experiment_display_mapping': experiment_display_mapping,
            'tasks': valid_tasks,
            'results_dir': results_dir,
            'base_dir': base_dir
        }
    except Exception as e:
        print(f"Error loading config: {e}")
        return None

def get_task_mapping(base_path, task_name):
    """Get mapping for a task from valid_tasks.json."""
    tasks_json = Path(base_path) / 'tasks' / 'valid_tasks.json'
    if tasks_json.exists():
        with open(tasks_json, 'r') as f:
            tasks_list = json.load(f)
            for task in tasks_list:
                if task.get('task_name') == task_name:
                    return task.get('mapping', {})
    return {}


def _local_path_to_nifti_path(local_path_str, download_base=None, prefix=None):
    """Convert local_path (gs://... or path) to local NIfTI file path."""
    if download_base is None:
        download_base = Path('/home/dcunhrya/downloaded_ct_scans')
    if prefix is None:
        prefix = 'chaudhari_lab/ct_data/ct_scans/vista/nov25'
    parts = local_path_str.split('/')
    filename_no_ext = parts[-1].replace('.zip', '')
    bucket_filename = f"{parts[-2]}__{filename_no_ext}.nii.gz"
    return Path(download_base) / prefix / bucket_filename


def _find_sample_nifti_path(base_path, valid_tasks, use_subsampled=True):
    """Find first existing NIfTI path from task CSVs."""
    tasks_json = Path(base_path) / 'tasks' / 'valid_tasks.json'
    if not tasks_json.exists():
        return None
    with open(tasks_json, 'r') as f:
        tasks_list = json.load(f)
    suffix = '_subsampled' if use_subsampled else ''
    for task in tasks_list:
        task_name = task.get('task_name')
        if valid_tasks and task_name not in valid_tasks:
            continue
        source_csv = task.get('task_source_csv')
        if not source_csv:
            continue
        csv_name = f"{task_name}{suffix}.csv"
        csv_path = Path(base_path) / source_csv / csv_name
        if not csv_path.exists():
            continue
        try:
            df = pd.read_csv(csv_path, nrows=500)
            local_path_col = next((c for c in df.columns if c.lower() == 'local_path'), None)
            if not local_path_col:
                continue
            for _, row in df.iterrows():
                local_path_str = row[local_path_col]
                if pd.notna(local_path_str) and isinstance(local_path_str, str):
                    nifti_path = _local_path_to_nifti_path(local_path_str)
                    if nifti_path.exists():
                        return nifti_path
        except Exception as e:
            continue
    return None


def _extract_slices_for_experiment(img_data, experiment):
    """Extract slice arrays for an experiment (matches vqa_dataset logic)."""
    from vista_run.utils.utils_inference import normalize_slice

    slices = []
    if experiment == 'axial_1_image':
        if len(img_data.shape) > 2:
            idx = img_data.shape[2] // 2
            s = img_data[:, :, idx]
            slices.append(normalize_slice(s))
    elif experiment == 'all_image':
        if len(img_data.shape) >= 3:
            # sagittal, coronal, axial middle
            sx = img_data.shape[0] // 2
            sy = img_data.shape[1] // 2
            sz = img_data.shape[2] // 2
            slices.append(normalize_slice(img_data[sx, :, :]))
            slices.append(normalize_slice(img_data[:, sy, :]))
            slices.append(normalize_slice(img_data[:, :, sz]))
    elif experiment == 'axial_all_image':
        if len(img_data.shape) > 2:
            depth = img_data.shape[2]
            for i in range(10):
                pos = i * 0.1
                idx = min(int(pos * (depth - 1)), depth - 1)
                slices.append(normalize_slice(img_data[:, :, idx]))
    elif experiment == 'sagittal_all_image':
        if len(img_data.shape) > 0:
            width = img_data.shape[0]
            for i in range(10):
                pos = i * 0.1
                idx = min(int(pos * (width - 1)), width - 1)
                slices.append(normalize_slice(img_data[idx, :, :]))
    elif experiment == 'no_timeline':
        if len(img_data.shape) > 2:
            depth = img_data.shape[2]
            for i in range(100):
                pos = i / 99.0 if 99 else 0.0
                idx = min(int(pos * (depth - 1)), depth - 1)
                slices.append(normalize_slice(img_data[:, :, idx]))
    elif experiment == 'no_report':
        if len(img_data.shape) > 2:
            depth = img_data.shape[2]
            for i in range(50):
                pos = i / 49.0 if 49 else 0.0
                idx = min(int(pos * (depth - 1)), depth - 1)
                slices.append(normalize_slice(img_data[:, :, idx]))
    return slices


def generate_ct_slice_pdfs(base_path='/home/dcunhrya/vista_bench',
                          config_path='/home/dcunhrya/vista_eval/configs/all_tasks.yaml',
                          output_dir='figures/ct_example'):
    """
    For each CT-inclusive experiment, generate a PDF with slices (small, fit on single page).
    """
    try:
        import nibabel as nib
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_pdf import PdfPages
    except ImportError as e:
        print(f"Skipping CT slice PDFs (missing dependency: {e})")
        return

    config = load_config(config_path)
    if not config:
        return
    valid_tasks = config.get('tasks', [])
    use_subsampled = config.get('subsample', True)

    nifti_path = _find_sample_nifti_path(base_path, valid_tasks, use_subsampled)
    if not nifti_path:
        print("No existing CT scan found in task CSVs. Skipping CT slice PDFs.")
        return

    try:
        img_obj = nib.load(str(nifti_path))
        img_data = img_obj.get_fdata()
    except Exception as e:
        print(f"Failed to load NIfTI {nifti_path}: {e}")
        return

    output_path_obj = Path(output_dir)
    output_path_obj.mkdir(parents=True, exist_ok=True)
    save_path = output_path_obj / 'ct_slices.pdf'

    with PdfPages(str(save_path)) as pdf:
        for experiment in CT_EXPERIMENTS:
            slices = _extract_slices_for_experiment(img_data, experiment)
            if not slices:
                continue

            n = len(slices)
            # Choose grid dimensions to fit on single letter page (small subplots)
            if n <= 3:
                ncols, nrows = n, 1
            elif n <= 10:
                ncols, nrows = 5, 2
            elif n <= 50:
                ncols, nrows = 10, 5
            else:
                ncols, nrows = 10, 10

            fig, axes = plt.subplots(nrows, ncols, figsize=(11, 8.5))
            if nrows == 1 and ncols == 1:
                axes = np.array([[axes]])
            elif nrows == 1 or ncols == 1:
                axes = np.atleast_1d(axes).reshape(nrows, ncols)

            for i, ax in enumerate(axes.flat):
                if i < n:
                    ax.imshow(slices[i], cmap='gray', aspect='auto')
                    ax.set_title(f'{i+1}', fontsize=6)
                ax.axis('off')

            fig.suptitle(f'CT Slices: {experiment} ({n} slices)', fontsize=12, fontweight='bold', y=0.98)
            plt.tight_layout(rect=[0, 0, 1, 0.96])
            pdf.savefig(fig, dpi=150, bbox_inches='tight')
            plt.close(fig)

    print(f"  CT slice PDF: {save_path}")

def generate_questions_pdf(results_path=None,
                          base_path='/home/dcunhrya/vista_bench',
                          config_path='/home/dcunhrya/vista_eval/configs/all_tasks.yaml',
                          output_path='figures/questions_responses.pdf'):
    """
    Generate PDF with first question from each subtask, showing all model responses across experiments.
    Structure: Task -> Subtask -> Question + Correct Answer -> Model -> Experiment responses

    Args:
        results_path: Base path to search for result CSV files (overridden by config paths.results_dir if present)
        base_path: Base path to load task registry for mappings
        config_path: Path to YAML config file
        output_path: Output PDF path
    """
    # Load config first to get paths.results_dir
    config = load_config(config_path)
    if not config:
        print("Failed to load config. Exiting.")
        return

    # Use results_dir from config if present, else fall back to results_path arg
    config_results_dir = config.get('results_dir')
    if config_results_dir is not None:
        results_path = config_results_dir
    if results_path is None:
        results_path = '/home/dcunhrya/results'

    # Use base_dir from config if present, else fall back to base_path arg
    config_base_dir = config.get('base_dir')
    if config_base_dir is not None:
        base_path = config_base_dir

    results_base = Path(results_path)
    base_path_obj = Path(base_path)

    valid_models = config['models']
    model_name_mapping = config['model_name_mapping']
    valid_experiments = config['experiments']
    experiment_display_mapping = config.get('experiment_display_mapping', {})
    valid_tasks = config['tasks']
    
    print(f"Loaded {len(valid_models)} models: {valid_models}")
    print(f"Loaded {len(valid_experiments)} experiments: {valid_experiments}")
    print(f"Loaded {len(valid_tasks)} tasks: {valid_tasks}")
    
    # Find all result files
    print("\nScanning for result files...")
    all_result_files = list(results_base.rglob("*_results_*.csv"))
    
    # Organize data: [source_folder][task_name][model_name][experiment] = first row data
    structured_data = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    # Per-CSV means of unique_events per subtask: [source_folder][task_name] = [mean1, mean2, ...]
    # Each results CSV contributes its own mean(unique_events); we average those for the subtask
    unique_events_means = defaultdict(lambda: defaultdict(list))

    for file_path in all_result_files:
        relative_parts = file_path.relative_to(results_base).parts
        
        if len(relative_parts) < 4:
            continue
        
        source_folder = relative_parts[0]
        task_name = relative_parts[1]
        model_name = relative_parts[2]
        filename = relative_parts[3]
        
        # Filter by task
        if valid_tasks and task_name not in valid_tasks:
            continue
        
        # Filter by model
        model_matches = False
        if model_name in valid_models:
            model_matches = True
        else:
            # Check if model_name ends with any valid model pattern
            for valid_model in valid_models:
                if model_name == valid_model or model_name.endswith('_' + valid_model) or model_name.endswith('/' + valid_model):
                    model_matches = True
                    break
        
        if not model_matches:
            continue
        
        # Extract experiment from filename
        experiment = extract_experiment_from_filename(filename)
        
        # Filter by experiment
        if valid_experiments and experiment not in valid_experiments:
            continue
        
        try:
            df = pd.read_csv(file_path)
            if df.empty:
                continue
            
            # Get the first row (first question)
            first_row = df.iloc[0]
            
            # Check if is_correct column exists, otherwise calculate it
            is_correct = first_row.get('is_correct', None)
            if pd.isna(is_correct) or is_correct is None or 'is_correct' not in df.columns:
                # Calculate is_correct if not present (checks all candidates: | splits, \boxed{}, <answer>/<label>, last word/phrase)
                mapping = get_task_mapping(base_path_obj, task_name)
                label = first_row.get('label', 'N/A')
                mapped_label = map_label_to_answer(label, mapping)
                model_response = str(first_row.get('model_response', ''))
                is_correct = is_answer_correct(model_response, mapped_label)
            else:
                is_correct = bool(is_correct)
            
            structured_data[source_folder][task_name][model_name][experiment] = {
                'question': str(first_row.get('question', 'N/A')),
                'model_response': str(first_row.get('model_response', 'N/A')),
                'label': str(first_row.get('label', 'N/A')),
                'index': first_row.get('index', 'N/A'),
                'is_correct': is_correct
            }

            # For this specific results CSV: compute mean of unique_events within that CSV
            if 'unique_events' in df.columns:
                vals = pd.to_numeric(df['unique_events'], errors='coerce').dropna()
                if len(vals) > 0:
                    csv_mean = float(vals.mean())
                    unique_events_means[source_folder][task_name].append(csv_mean)

        except Exception as e:
            print(f"Error reading {file_path}: {e}")
    
    # Get task mappings for correct answers
    task_mappings = {}
    for source_folder in structured_data.keys():
        for task_name in structured_data[source_folder].keys():
            if task_name not in task_mappings:
                task_mappings[task_name] = get_task_mapping(base_path_obj, task_name)
    
    # Initialize PDF
    output_path_obj = Path(output_path)
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)
    
    doc = SimpleDocTemplate(str(output_path_obj), pagesize=letter, 
                           leftMargin=72, rightMargin=72, topMargin=72, bottomMargin=72)
    story = []
    styles = getSampleStyleSheet()
    
    # Define Styles
    title_style = ParagraphStyle('Title', parent=styles['Heading1'], 
                                 alignment=TA_CENTER, spaceAfter=20, fontSize=16)
    source_style = ParagraphStyle('Source', parent=styles['Heading1'], 
                                  textColor='#2980b9', spaceBefore=20, fontSize=14)
    task_style = ParagraphStyle('Task', parent=styles['Heading2'], 
                                textColor='#8e44ad', leftIndent=10, fontSize=12)
    question_header_style = ParagraphStyle('QHeader', parent=styles['Heading3'], 
                                          textColor='#2c3e50', leftIndent=20, fontSize=11)
    question_text_style = ParagraphStyle('QText', parent=styles['BodyText'], 
                                        leftIndent=25, rightIndent=25, fontSize=10)
    correct_answer_style = ParagraphStyle('CorrectAnswer', parent=styles['BodyText'], 
                                         leftIndent=25, rightIndent=25, fontSize=10, 
                                         textColor='#27ae60', spaceAfter=10)
    model_header_style = ParagraphStyle('ModelHeader', parent=styles['Heading3'], 
                                        textColor='#e67e22', leftIndent=30, fontSize=11, 
                                        spaceBefore=10)
    experiment_style = ParagraphStyle('Experiment', parent=styles['BodyText'], 
                                      leftIndent=50, rightIndent=25, fontSize=9, 
                                      backColor='#f8f9fa', spaceBefore=3, spaceAfter=3)
    avg_unique_events_style = ParagraphStyle('AvgUniqueEvents', parent=styles['BodyText'],
                                             leftIndent=20, rightIndent=25, fontSize=10,
                                             textColor='#7f8c8d', spaceAfter=5)

    story.append(Paragraph("VLM Multi-Model Comparison Report", title_style))
    story.append(Spacer(1, 0.2*inch))
    
    # Hierarchical traversal: Source -> Task -> Models -> Experiments
    for source_folder in sorted(structured_data.keys()):
        story.append(Paragraph(f"Dataset: {source_folder}", source_style))
        story.append(Spacer(1, 0.1*inch))
        
        for task_name in sorted(structured_data[source_folder].keys()):
            story.append(Paragraph(f"Subtask: {task_name}", task_style))
            story.append(Spacer(1, 0.05*inch))

            # Average unique events for this subtask (mean of per-CSV means)
            means_list = unique_events_means[source_folder][task_name]
            if means_list:
                avg_unique_events = sum(means_list) / len(means_list)
                story.append(Paragraph(
                    f"<b>Average unique events:</b> {avg_unique_events:.2f}",
                    avg_unique_events_style
                ))
                story.append(Spacer(1, 0.05*inch))

            # Get the first question and correct answer from any model/experiment
            first_data = None
            for model_name in structured_data[source_folder][task_name].keys():
                for experiment in structured_data[source_folder][task_name][model_name].keys():
                    first_data = structured_data[source_folder][task_name][model_name][experiment]
                    break
                if first_data:
                    break
            
            if not first_data:
                continue
            
            # Get mapping for this task
            mapping = task_mappings.get(task_name, {})
            label = first_data.get('label', 'N/A')
            mapped_label = map_label_to_answer(label, mapping)
            
            # Print question and correct answer
            q_text = first_data.get('question', 'N/A').replace('<', '&lt;').replace('>', '&gt;')
            
            story.append(Paragraph("<b>Question:</b>", question_header_style))
            story.append(Paragraph(q_text, question_text_style))
            story.append(Paragraph(f"<b>Correct Answer:</b> {mapped_label}", correct_answer_style))
            story.append(Spacer(1, 0.1*inch))
            
            # For each model, print all experiment responses
            for model_name in sorted(structured_data[source_folder][task_name].keys()):
                display_model_name = model_name_mapping.get(model_name, model_name)
                story.append(Paragraph(f"<b>Model: {display_model_name}</b>", model_header_style))
                
                # Print responses for each experiment in order
                for experiment in sorted(valid_experiments):
                    if experiment in structured_data[source_folder][task_name][model_name]:
                        data = structured_data[source_folder][task_name][model_name][experiment]
                        response = data.get('model_response', 'N/A').replace('<', '&lt;').replace('>', '&gt;')
                        is_correct = data.get('is_correct', None)
                        
                        # Get display name for experiment (use comment if available, otherwise use experiment name)
                        experiment_display = experiment_display_mapping.get(experiment, experiment)
                        
                        # Add is_correct indicator
                        if is_correct is not None:
                            if is_correct:
                                correct_indicator = "<font color='#27ae60'><b>[CORRECT]</b></font>"
                            else:
                                correct_indicator = "<font color='#c0392b'><b>[INCORRECT]</b></font>"
                            story.append(Paragraph(f"<b>{experiment_display}:</b> {correct_indicator} {response}", experiment_style))
                        else:
                            story.append(Paragraph(f"<b>{experiment_display}:</b> {response}", experiment_style))
                
                story.append(Spacer(1, 0.05*inch))
            
            story.append(Spacer(1, 0.2*inch))
            
            # Page break between subtasks
            story.append(PageBreak())
    
    doc.build(story)
    print(f"\nPDF generated: {output_path}")


if __name__ == "__main__":
    config_path = '/home/rdcunha/vista_project/vista_eval_vlm/configs/all_tasks.yaml'
    base_path = '/home/rdcunha/vista_project/vista_bench'
    generate_questions_pdf(config_path=config_path)
    # print("\nGenerating CT slice PDFs for each experiment...")
    # generate_ct_slice_pdfs(base_path=base_path, config_path=config_path,
    #                       output_dir='/home/rdcunha/vista_project/vista_eval_vlm/figures/ct_example')
