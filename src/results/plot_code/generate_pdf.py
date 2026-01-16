import os
import pandas as pd
from pathlib import Path
from collections import defaultdict
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
from reportlab.lib.enums import TA_CENTER

def generate_questions_pdf(results_path='/home/dcunhrya/results', output_path='figures/questions_responses.pdf'):
    results_base = Path(results_path)
    # Nested dictionary: [source][task][index] = list of model responses
    structured_data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    
    print("Gathering data from analyzed results...")
    for analyzed_file in results_base.rglob("results_analyzed.csv"):
        relative_parts = analyzed_file.relative_to(results_base).parts
        if len(relative_parts) < 3: continue
            
        source_folder = relative_parts[0]
        task_name = relative_parts[1]
        model_name = relative_parts[2]
        
        try:
            df = pd.read_csv(analyzed_file)
            for _, row in df.iterrows():
                entry_idx = row.get('index', 'N/A')
                structured_data[source_folder][task_name][entry_idx].append({
                    'model': model_name,
                    'question': str(row.get('question', 'N/A')),
                    'response': str(row.get('model_response', 'N/A')),
                    'mapped_label': str(row.get('mapped_label', 'N/A')),
                    'is_correct': row.get('is_correct', False)
                })
        except Exception as e:
            print(f"Error reading {analyzed_file}: {e}")

    # Initialize PDF
    doc = SimpleDocTemplate(output_path, pagesize=letter, margin=72)
    story = []
    styles = getSampleStyleSheet()

    # Define Styles
    title_style = ParagraphStyle('Title', parent=styles['Heading1'], alignment=TA_CENTER, spaceAfter=20)
    source_style = ParagraphStyle('Source', parent=styles['Heading1'], textColor='#2980b9', spaceBefore=20)
    task_style = ParagraphStyle('Task', parent=styles['Heading2'], textColor='#8e44ad', leftIndent=10)
    question_header_style = ParagraphStyle('QHeader', parent=styles['Heading3'], textColor='#2c3e50', leftIndent=20)
    question_text_style = ParagraphStyle('QText', parent=styles['BodyText'], leftIndent=25, rightIndent=25, italic=True)
    
    model_resp_style = ParagraphStyle('ModelResp', parent=styles['BodyText'], fontSize=9, leftIndent=40, backColor='#f8f9fa', spaceBefore=5)
    correct_tag = "<font color='#27ae60'>[CORRECT]</font>"
    incorrect_tag = "<font color='#c0392b'>[INCORRECT]</font>"

    story.append(Paragraph("VLM Multi-Model Comparison Report", title_style))

    # Hierarchical traversal: Source -> Task -> Question Index
    for source in sorted(structured_data.keys()):
        story.append(Paragraph(f"Dataset: {source}", source_style))
        
        for task in sorted(structured_data[source].keys()):
            story.append(Paragraph(f"Subtask: {task}", task_style))
            story.append(Spacer(1, 0.1*inch))
            
            for idx in sorted(structured_data[source][task].keys()):
                responses = structured_data[source][task][idx]
                if not responses: continue

                # 1. Print the Question once (taking the first model's version of it)
                q_text = responses[0]['question'].replace('<', '&lt;').replace('>', '&gt;')
                expected = responses[0]['mapped_label']
                
                story.append(Paragraph(f"Question Index: {idx}", question_header_style))
                story.append(Paragraph(q_text, question_text_style))
                story.append(Paragraph(f"<b>Expected Answer:</b> {expected}", question_text_style))
                story.append(Spacer(1, 0.05*inch))

                # 2. Print each model's response for this specific question
                for resp in responses:
                    status = correct_tag if resp['is_correct'] else incorrect_tag
                    r_text = resp['response'].replace('<', '&lt;').replace('>', '&gt;')
                    if len(r_text) > 5000: r_text = r_text[:5000] + "... [TRUNCATED]"
                    
                    model_line = f"<b>{resp['model']}</b> {status}<br/>{r_text}"
                    story.append(Paragraph(model_line, model_resp_style))
                
                story.append(Spacer(1, 0.2*inch))
                
                # Page break logic to prevent massive tasks from clumping
                if len(story) > 50: # Crude check for flowable objects
                    story.append(PageBreak())

    doc.build(story)
    print(f"Grouped PDF generated: {output_path}")

if __name__ == "__main__":
    generate_questions_pdf()