import os
import pandas as pd
import yaml
import re
from pathlib import Path
from collections import defaultdict
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
from reportlab.lib.enums import TA_LEFT, TA_CENTER

from results.results_analyzer import is_answer_correct, map_label_to_answer


def extract_experiment_from_filename(filename):
    """Extract experiment name from filename like: task_name_results_experiment.csv"""
    match = re.search(r'_results_(.+)\.csv$', filename)
    if match:
        return match.group(1)
    return 'default'


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
        
        # Extract tasks
        valid_tasks = config.get('tasks', [])
        
        return {
            'models': valid_models,
            'model_name_mapping': model_name_mapping,
            'experiments': valid_experiments,
            'tasks': valid_tasks
        }
    except Exception as e:
        print(f"Error loading config: {e}")
        return None

def get_task_mapping(base_path, task_name):
    """Get mapping for a task from valid_tasks.json."""
    tasks_json = Path(base_path) / 'tasks' / 'valid_tasks.json'
    if tasks_json.exists():
        import json
        with open(tasks_json, 'r') as f:
            tasks_list = json.load(f)
            for task in tasks_list:
                if task.get('task_name') == task_name:
                    return task.get('mapping', {})
    return {}

def generate_questions_pdf(results_path='/home/dcunhrya/results', 
                          base_path='/home/dcunhrya/vista_bench',
                          config_path='/home/dcunhrya/vista_eval/configs/all_tasks.yaml',
                          output_path='figures/questions_responses.pdf'):
    """
    Generate PDF with first question from each subtask, showing all model responses across experiments.
    Structure: Task -> Subtask -> Question + Correct Answer -> Model -> Experiment responses
    """
    results_base = Path(results_path)
    base_path_obj = Path(base_path)
    
    # Load config
    config = load_config(config_path)
    if not config:
        print("Failed to load config. Exiting.")
        return
    
    valid_models = config['models']
    model_name_mapping = config['model_name_mapping']
    valid_experiments = config['experiments']
    valid_tasks = config['tasks']
    
    print(f"Loaded {len(valid_models)} models: {valid_models}")
    print(f"Loaded {len(valid_experiments)} experiments: {valid_experiments}")
    print(f"Loaded {len(valid_tasks)} tasks: {valid_tasks}")
    
    # Find all result files
    print("\nScanning for result files...")
    all_result_files = list(results_base.rglob("*_results_*.csv"))
    
    # Organize data: [source_folder][task_name][model_name][experiment] = first row data
    structured_data = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    
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
    
    story.append(Paragraph("VLM Multi-Model Comparison Report", title_style))
    story.append(Spacer(1, 0.2*inch))
    
    # Hierarchical traversal: Source -> Task -> Models -> Experiments
    for source_folder in sorted(structured_data.keys()):
        story.append(Paragraph(f"Dataset: {source_folder}", source_style))
        story.append(Spacer(1, 0.1*inch))
        
        for task_name in sorted(structured_data[source_folder].keys()):
            story.append(Paragraph(f"Subtask: {task_name}", task_style))
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
                        
                        # Add is_correct indicator
                        if is_correct is not None:
                            if is_correct:
                                correct_indicator = "<font color='#27ae60'><b>[CORRECT]</b></font>"
                            else:
                                correct_indicator = "<font color='#c0392b'><b>[INCORRECT]</b></font>"
                            story.append(Paragraph(f"<b>{experiment}:</b> {correct_indicator} {response}", experiment_style))
                        else:
                            story.append(Paragraph(f"<b>{experiment}:</b> {response}", experiment_style))
                
                story.append(Spacer(1, 0.05*inch))
            
            story.append(Spacer(1, 0.2*inch))
            
            # Page break between subtasks
            story.append(PageBreak())
    
    doc.build(story)
    print(f"\nPDF generated: {output_path}")

if __name__ == "__main__":
    generate_questions_pdf()
