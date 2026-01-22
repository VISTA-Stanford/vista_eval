# import os
# import yaml
# import json
# import pandas as pd
# from pathlib import Path

# class TaskOrchestrator:
#     def __init__(self, config_path='config.yaml'):
#         # Load YAML configuration
#         with open(config_path, 'r') as f:
#             self.config = yaml.safe_load(f)
            
#         self.base_path = Path(self.config['paths']['base_dir'])
#         self.results_path = Path(self.config['paths']['results_dir'])
#         self.model_name = self.config['model']['name']
        
#         # Load JSONs using paths from YAML
#         with open(self.base_path / self.config['paths']['valid_tasks'], 'r') as f:
#             self.tasks_def = json.load(f)
#         with open(self.base_path / self.config['paths']['prompts'], 'r') as f:
#             self.prompts_def = json.load(f)

#     def run_all_tasks(self):
#         """Iterates through every task defined in valid_tasks.json"""
#         all_task_names = [t['task_name'] for t in self.tasks_def]
#         print(f"Found {len(all_task_names)} tasks in registry. Starting batch inference...")
        
#         for name in all_task_names:
#             self.run_task(name)

#     def run_task(self, task_name):
#         """Runs inference for a specific task name"""
#         print(f"--- Processing Task: {task_name} ---")
#         task_info = next((t for t in self.tasks_def if t['task_name'] == task_name), None)
        
#         if not task_info:
#             print(f"Error: Task '{task_name}' not found in valid_tasks.json")
#             return

#         csv_path = self.base_path / 'data' / task_info['task_source_csv'] / f"{task_name}.csv"
#         if not csv_path.exists():
#             print(f"File not found: {csv_path}")
#             return
            
#         df = pd.read_csv(csv_path)
        
#         timeline_col = next((c for c in df.columns if 'patient_string' in c.lower()), None)
#         if not timeline_col:
#             print(f"Column 'PATIENT_TIMELINE' missing in {task_name}")
#             return

#         # 3. Inference Loop
#         results = []
#         for _, row in df.iterrows():
#             prompt = self.prompts_def[task_name].replace('[PATIENT_TIMELINE]', str(row[timeline_col]))
            
#             # --- MODEL INFERENCE ---
#             # response = vlm_pipeline.predict(prompt)
#             response = "Output text" 
            
#             # 4. Filter columns and add response
#             res_row = row.drop(labels=[timeline_col]).to_dict()
#             res_row['model_response'] = response
#             results.append(res_row)

#         self._save(task_name, task_info['task_source_csv'], results)
#     def _save(self, task_name, source_csv, data):
#         save_dir = self.results_path / source_csv / task_name / self.model_name
#         save_dir.mkdir(parents=True, exist_ok=True)
#         pd.DataFrame(data).to_csv(save_dir / "results.csv", index=False)

import os
import yaml
import json
import pandas as pd
from pathlib import Path
from google.cloud import bigquery

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

        # Initialize BigQuery Client
        # Ensure your 'config.yaml' has a 'project' field or set it manually here
        self.project_id = self.config.get('project_id', 'som-nero-plevriti-deidbdf')
        self.bq_client = bigquery.Client(project=self.project_id)

    def run_all_tasks(self):
        """Iterates through every task defined in valid_tasks.json"""
        all_task_names = [t['task_name'] for t in self.tasks_def]
        print(f"Found {len(all_task_names)} tasks in registry. Starting batch inference...")
        
        for name in all_task_names:
            self.run_task(name)

    def run_task(self, task_name):
        """Runs inference for a specific task name via BigQuery"""
        print(f"--- Processing Task: {task_name} ---")
        
        # 1. Get Task Info
        task_info = next((t for t in self.tasks_def if t['task_name'] == task_name), None)
        if not task_info:
            print(f"Error: Task '{task_name}' not found in valid_tasks.json")
            return

        # 2. Construct BigQuery Reference
        # task_source_csv now acts as the Dataset Name (e.g., 'vista_bench_v1_1')
        dataset_id = task_info['task_source_csv'] 
        
        # The user specified the table structure: som-nero-plevriti-deidbdf.Datasets.vista_bench_v1_1.Tables
        # Adjust the SQL below if your project/dataset hierarchy is different.
        # Assuming table ID format: `project.dataset.table`
        # If task_source_csv is just the dataset name 'vista_bench_v1_1', we use that.
        full_table_id = f"{self.project_id}.Datasets.{dataset_id}"

        # 3. Query BigQuery
        # We query for rows where 'task' matches the current task_name
        query = f"""
            SELECT *
            FROM `{full_table_id}`
            WHERE task = @task_name
        """
        
        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("task_name", "STRING", task_name)
            ]
        )

        try:
            df = self.bq_client.query(query, job_config=job_config).to_dataframe()
        except Exception as e:
            print(f"BigQuery Error for {task_name}: {e}")
            return

        if df.empty:
            print(f"No data found for task '{task_name}' in table {full_table_id}")
            return
            
        # 4. Identify Timeline Column (Case-insensitive)
        timeline_col = next((c for c in df.columns if 'patient_string' in c.lower()), None)
        if not timeline_col:
            print(f"Column 'PATIENT_TIMELINE' (patient_string) missing in data for {task_name}")
            return

        print(f"  Loaded {len(df)} rows from BigQuery.")

        # 5. Inference Loop
        results = []
        for _, row in df.iterrows():
            # Get the correct prompt template
            base_prompt = self.prompts_def.get(task_name)
            if not base_prompt:
                print(f"Warning: No prompt found for {task_name}, skipping.")
                break

            prompt = base_prompt.replace('[PATIENT_TIMELINE]', str(row[timeline_col]))
            
            # --- MODEL INFERENCE PLACEHOLDER ---
            # response = vlm_pipeline.predict(prompt)
            response = "Output text" 
            
            # 6. Filter columns and add response
            # We convert row to dict but drop the heavy timeline column to save space
            res_row = row.drop(labels=[timeline_col]).to_dict()
            res_row['model_response'] = response
            results.append(res_row)

        self._save(task_name, dataset_id, results)

    def _save(self, task_name, source_dataset, data):
        # We save under the dataset name folder structure
        save_dir = self.results_path / source_dataset / task_name / self.model_name
        save_dir.mkdir(parents=True, exist_ok=True)
        
        out_file = save_dir / "results.csv"
        pd.DataFrame(data).to_csv(out_file, index=False)
        print(f"  Saved results to {out_file}")