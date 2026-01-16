import os
import json
import re
import pandas as pd
from pathlib import Path

class DynamicResultsAnalyzer:
    def __init__(self, base_path='/home/dcunhrya/vista_bench', 
                 results_path='/home/dcunhrya/results'):
        self.base_path = Path(base_path)
        self.results_path = Path(results_path)
        
        # Load tasks once into a dictionary for fast lookup by task_name
        # Note: adjust path if your 'tasks' folder name varies (e.g., 'configs')
        tasks_json = self.base_path / 'tasks' / 'valid_tasks.json'
            
        with open(tasks_json, 'r') as f:
            tasks_list = json.load(f)
            self.task_registry = {t['task_name']: t for t in tasks_list}

    def _extract_answer(self, text):
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

    def _clean_question(self, text):
        """
        Filters the question to keep only the part before 'OPTIONS'.
        """
        if pd.isna(text):
            return ""
        # Split on 'OPTIONS' (case-insensitive check is safer)
        # Using regex split to handle 'OPTIONS', 'Options', etc.
        parts = re.split(r'\bOPTIONS\b', str(text), flags=re.IGNORECASE)
        return parts[0].strip()

    def analyze_all_models(self):
        print(f"Scanning results in: {self.results_path}")
        result_files = list(self.results_path.rglob("*_results.csv"))
        
        if not result_files:
            print("No result files found. Check your paths.")
            return

        for file_path in result_files:
            if file_path.name == "results_analyzed.csv":
                continue
                
            # Pattern matching for task names
            task_name = file_path.name.replace("_subsampled_results.csv", "").replace("_results.csv", "")
            
            if task_name not in self.task_registry:
                continue

            task_info = self.task_registry[task_name]
            mapping = task_info.get('mapping', {})
            
            print(f"Processing Model: {file_path.parent.name} | Task: {task_name}")
            self._process_file(file_path, mapping)

    def _process_file(self, file_path, mapping):
        try:
            df = pd.read_csv(file_path)
            
            # 1. Filter Question: Remove everything from 'OPTIONS' onwards
            df['question'] = df['question'].apply(self._clean_question)

            # 2. Extract Response: Handle <answer> tags or Pipe |
            df['cleaned_response'] = df['model_response'].apply(self._extract_answer)

            # 3. Map Label: Numeric key -> String label
            df['mapped_label'] = df['label'].astype(str).map(mapping)

            # 4. Accuracy Check: Case-insensitive comparison
            df['is_correct'] = df.apply(
                lambda x: str(x['cleaned_response']).strip().lower() == str(x['mapped_label']).strip().lower() 
                if pd.notna(x['mapped_label']) else False, 
                axis=1
            )

            # 5. Save results_analyzed.csv
            cols_to_keep = ['index', 'question', 'label', 'mapped_label', 'model_response', 'cleaned_response', 'is_correct']
            existing_cols = [c for c in cols_to_keep if c in df.columns]
            
            out_path = file_path.parent / "results_analyzed.csv"
            df[existing_cols].to_csv(out_path, index=False)
            
        except Exception as e:
            print(f"Error processing {file_path}: {e}")

if __name__ == "__main__":
    analyzer = DynamicResultsAnalyzer()
    analyzer.analyze_all_models()