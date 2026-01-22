import os
import argparse
import json
import yaml
import torch
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import DataLoader
from google.cloud import bigquery  # NEW IMPORT

from vqa_dataset import PromptDataset, prompt_collate
from models import load_model_adapter

class TaskOrchestrator:
    def __init__(self, config_path, model_type, model_name):
        # 1. Load YAML Config
        with open(config_path, "r") as f:
            self.cfg = yaml.safe_load(f)
        
        self.base_path = Path(self.cfg['paths']['base_dir'])
        self.results_base = Path(self.cfg['paths']['results_dir'])
        self.model_type = model_type
        self.model_name = model_name
        self.file_model_name = model_name.split('/')[-1].replace('/', '_')
        
        # 2. Set Environments
        self._set_envs(self.cfg['runtime']['cache_dir'])
        
        # 3. Load Task Registries
        valid_task_path = os.path.join(self.base_path, self.cfg['paths']['valid_tasks'])
        prompts_path = os.path.join(self.base_path, self.cfg['paths']['prompts'])

        with open(valid_task_path, 'r') as f:
            self.valid_tasks = json.load(f)
        with open(prompts_path, 'r') as f:
            self.prompts_map = json.load(f)

        # 4. Initialize BigQuery Client (NEW)
        # We default to the specific project mentioned. 
        self.project_id = "som-nero-plevriti-deidbdf"
        self.bq_client = bigquery.Client(project=self.project_id)

        # 5. Initialize Model
        self.adapter = load_model_adapter(
            self.model_type, 
            self.model_name, 
            self.cfg['model'].get("device", "auto"), 
            self.cfg['runtime']['cache_dir']
        )
        self.model, self.processor = self.adapter.load()

    def _set_envs(self, model_dir):
        os.environ.update({
            "HF_HOME": model_dir,
            "TRANSFORMERS_CACHE": model_dir,
            "VLLM_CACHE_ROOT": model_dir,
            "TOKENIZERS_PARALLELISM": "false"
        })

    def run_inference(self, task_names=None):
        # Determine which tasks to run
        tasks_to_run = self.valid_tasks
        if task_names:
            tasks_to_run = [t for t in self.valid_tasks if t['task_name'] in task_names]

        for task_info in tasks_to_run:
            self._process_single_task(task_info)

    def _process_single_task(self, task_info):
        task_name = task_info['task_name']
        source_csv = task_info['task_source_csv'] # Acts as Table ID (e.g., 'oncology')
        print(f"\n>>> Starting Task: {task_name}")

        # --- BIGQUERY LOADING START ---
        # Construct Table ID: project.dataset.table
        dataset_id = "vista_bench_v1_1"
        full_table_id = f"{self.project_id}.{dataset_id}.{source_csv}"
        
        print(f"    Querying BigQuery table: {full_table_id} for task='{task_name}'...")
        
        query = f"""
            SELECT *
            FROM `{full_table_id}`
            WHERE task = @task_name
        """
        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("task", "STRING", task_name)
            ]
        )

        try:
            df = self.bq_client.query(query, job_config=job_config).to_dataframe()
        except Exception as e:
            print(f"!!! BigQuery Error for {task_name}: {e}")
            return

        if df.empty:
            print(f"!!! No data found for task '{task_name}' in BigQuery.")
            return
            
        print(f"    Loaded {len(df)} rows from BigQuery.")
        # --- BIGQUERY LOADING END ---
        
        # Ensure unique index for tracking
        if 'index' not in df.columns:
            df['index'] = df.index

        # Dynamic Column Detection (works for BQ dataframe too)
        timeline_col = next((c for c in df.columns if 'patient_string' in c.lower()), None)
        
        if not timeline_col:
            print(f"!!! Error: Column 'patient_string' (or similar) not found in BQ results.")
            return

        # Define OOM Protection Truncator
        def truncate_timeline(text, max_chars=5000):
            if len(str(text)) > max_chars:
                return str(text)[:max_chars] + "... [TRUNCATED]"
            return str(text)

        # 2. Setup Resume Logic
        # We keep the local folder structure for results
        save_dir = self.results_base / source_csv / task_name / self.file_model_name
        save_dir.mkdir(parents=True, exist_ok=True)
        out_file = save_dir / f"{task_name}_results.csv"

        existing_indices = set()
        if out_file.exists():
            try:
                existing_df = pd.read_csv(out_file)
                if 'index' in existing_df.columns:
                    existing_indices = set(existing_df['index'].tolist())
                    print(f"Resuming: Found {len(existing_indices)} existing records.")
            except Exception as e:
                print(f"Could not read existing file, starting fresh: {e}")

        # 3. Prepare Prompt and Dataset
        df[timeline_col] = df[timeline_col].apply(truncate_timeline)
        base_prompt_template = self.prompts_map.get(task_name, "[PATIENT_TIMELINE]")
        df['dynamic_prompt'] = df[timeline_col].apply(lambda x: base_prompt_template.replace('[PATIENT_TIMELINE]', str(x)))

        dataset = PromptDataset(df=df, prompt_col='dynamic_prompt') 
        loader = DataLoader(
            dataset,
            batch_size=self.cfg['runtime']['batch_size'],
            shuffle=False,
            collate_fn=prompt_collate,
            num_workers=4
        )

        # 4. Inference Loop
        results_buffer = []
        batch_counter = 0

        for batch in tqdm(loader, desc=f"Inference {task_name}"):
            # Skip items already processed
            new_items = [item for item in batch if item['raw_row']['index'] not in existing_indices]
            
            if not new_items:
                continue

            try:
                if 'gemma' in self.model_type:
                    all_inputs = []
                    for item in new_items:
                        single_msg = self.adapter.create_template(item)
                        single_inp = self.adapter.prepare_inputs([single_msg], self.processor, self.model)
                        all_inputs.append(single_inp)
                    batched_inputs = self.adapter.stack_inputs(all_inputs, self.model)
                    outputs = self.adapter.infer(self.model, self.processor, batched_inputs, self.cfg['runtime']["max_new_tokens"])
                else:
                    messages = [self.adapter.create_template(item) for item in new_items]
                    inputs = self.adapter.prepare_inputs(messages, self.processor, self.model)
                    outputs = self.adapter.infer(self.model, self.processor, inputs, self.cfg['runtime']["max_new_tokens"])
                
                # Process outputs
                for item, out_text in zip(new_items, outputs):
                    # We drop the heavy timeline column before saving to CSV
                    res_row = item['raw_row'].drop(labels=[timeline_col]).to_dict()
                    res_row['model_response'] = out_text
                    results_buffer.append(res_row)
                
                batch_counter += 1

                # Flush to disk every 20 batches
                if batch_counter % 20 == 0:
                    self._append_to_csv(out_file, results_buffer)
                    results_buffer = [] 

                # Clear CUDA cache periodically
                if batch_counter % 5 == 0:
                    torch.cuda.empty_cache()

            except Exception as e:
                print(f"Error in batch: {e}")
                continue

        # 6. Final Save for remaining items
        if results_buffer:
            self._append_to_csv(out_file, results_buffer)

    def _append_to_csv(self, file_path, data_list):
        """Helper to append data to CSV without rewriting the whole file"""
        new_df = pd.DataFrame(data_list)
        # If file doesn't exist, write with header; else append without header
        header = not file_path.exists()
        new_df.to_csv(file_path, mode='a', index=False, header=header)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--type", type=str, required=True)
    parser.add_argument("--name", type=str, required=True)
    parser.add_argument("--tasks", nargs='*', default=None)
    args = parser.parse_args()

    orchestrator = TaskOrchestrator(args.config, args.type, args.name)
    # Use command line tasks if provided, otherwise use config tasks
    tasks_to_run = args.tasks if args.tasks else orchestrator.cfg.get('tasks', None)
    orchestrator.run_inference(tasks_to_run)