import os
import yaml
import json
import pandas as pd
from pathlib import Path

class TaskOrchestrator:
    def __init__(self, config_path='config.yaml'):
        # Load YAML configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
        self.base_path = Path(self.config['paths']['base_dir'])
        self.results_path = Path(self.config['paths']['results_dir'])
        self.model_name = self.config['model']['name']
        
        # Load JSONs using paths from YAML
        with open(self.base_path / self.config['paths']['valid_tasks'], 'r') as f:
            self.tasks_def = json.load(f)
        with open(self.base_path / self.config['paths']['prompts'], 'r') as f:
            self.prompts_def = json.load(f)

    def run_all_tasks(self):
        """Iterates through every task defined in valid_tasks.json"""
        all_task_names = [t['task_name'] for t in self.tasks_def]
        print(f"Found {len(all_task_names)} tasks in registry. Starting batch inference...")
        
        for name in all_task_names:
            self.run_task(name)

    def run_task(self, task_name):
        """Runs inference for a specific task name"""
        print(f"--- Processing Task: {task_name} ---")
        task_info = next((t for t in self.tasks_def if t['task_name'] == task_name), None)
        
        if not task_info:
            print(f"Error: Task '{task_name}' not found in valid_tasks.json")
            return

        csv_path = self.base_path / 'data' / task_info['task_source_csv'] / f"{task_name}.csv"
        if not csv_path.exists():
            print(f"File not found: {csv_path}")
            return
            
        df = pd.read_csv(csv_path)
        
        timeline_col = next((c for c in df.columns if 'patient_string' in c.lower()), None)
        if not timeline_col:
            print(f"Column 'PATIENT_TIMELINE' missing in {task_name}")
            return

        # 3. Inference Loop
        results = []
        for _, row in df.iterrows():
            prompt = self.prompts_def[task_name].replace('[PATIENT_TIMELINE]', str(row[timeline_col]))
            
            # --- MODEL INFERENCE ---
            # response = vlm_pipeline.predict(prompt)
            response = "Output text" 
            
            # 4. Filter columns and add response
            res_row = row.drop(labels=[timeline_col]).to_dict()
            res_row['model_response'] = response
            results.append(res_row)

        self._save(task_name, task_info['task_source_csv'], results)
    def _save(self, task_name, source_csv, data):
        save_dir = self.results_path / source_csv / task_name / self.model_name
        save_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(data).to_csv(save_dir / "results.csv", index=False)