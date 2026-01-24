import os
import argparse
import json
import yaml
import re
import torch
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import DataLoader
from google.cloud import bigquery
from google.cloud import storage
import meds_reader
from meds_tools import patient_timeline
from meds2text.ontology import OntologyDescriptionLookupTable

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
        
        # 4.25. Initialize GCP Storage Client for NIfTI images
        self.storage_client = storage.Client(project=self.project_id)

        # 4.5. Initialize Meds Database and Ontology Lookup
        meds_db_path = self.base_path / "thoracic_cohort_meds" / "vista_thoracic_cohort_v0_db"
        ontology_path = self.base_path / "thoracic_cohort_meds" / "athena_omop_ontologies"
        
        print(f"    Initializing meds database from: {meds_db_path}")
        print(f"    Initializing ontology lookup from: {ontology_path}")
        
        self.lookup = OntologyDescriptionLookupTable()
        self.lookup.load(str(ontology_path))
        self.meds_database = meds_reader.SubjectDatabase(str(meds_db_path))

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

    def _get_patient_timeline(self, row):
        """
        Fetch patient timeline from meds database for a given row.
        
        Args:
            row: A pandas Series or dict containing subject_id, start_date, end_date
            
        Returns:
            str: Patient timeline as formatted string
        """
        # Try to find subject_id column (case-insensitive)
        subject_id = None
        for col in ['person_id','subject_id', 'SUBJECT_ID']:
            if col in row and pd.notna(row[col]):
                subject_id = row[col]
                break
        
        if subject_id is None:
            return "No subject_id found in row."
        
        # Try to find date columns (case-insensitive)
        start_date = None
        end_date = '2025-12-19'
        
        for col in ['diagnosis_date', 'start_date', 'start_time', 'START_TIME']:
            if col in row and pd.notna(row[col]):
                start_date = str(row[col])
                break
        
        # for col in ['end_date', 'END_DATE', 'endDate', 'EndDate', 'end_time', 'END_TIME']:
        #     if col in row and pd.notna(row[col]):
        #         end_date = str(row[col])
        #         break
        
        # if end_date is None:
        #     return f"No end_date found for subject_id {subject_id}."
        
        try:
            # Fetch patient timeline
            df_patient = patient_timeline.get_described_events_window(
                database=self.meds_database,
                lookup_table=self.lookup,
                subject_id=subject_id,
                start_time=start_date if start_date else None,
                end_time=end_date
            )
            
            # Convert to string format
            patient_string = patient_timeline.get_llm_event_string(df_patient, include_text=True)
            return patient_string if patient_string else "No clinical events found for this period."
            
        except Exception as e:
            return f"Error fetching timeline for subject_id {subject_id}: {str(e)}"

    def count_unique_event_dates(self, timeline_text):
        """
        Count unique event dates in the patient timeline.
        
        Args:
            timeline_text: The patient timeline string with format [YYYY-MM-DD HH:MM] | ...
            
        Returns:
            int: Number of unique event dates
        """
        if pd.isna(timeline_text) or not timeline_text:
            return 0
        
        text_str = str(timeline_text)
        
        # Pattern to match date-time markers: [YYYY-MM-DD HH:MM]
        # Extract just the date part (YYYY-MM-DD)
        pattern = r'\[(\d{4}-\d{2}-\d{2})\s+\d{2}:\d{2}\]'
        
        # Find all date matches
        dates = re.findall(pattern, text_str)
        
        # Count unique dates
        unique_dates = len(set(dates)) if dates else 0
        
        return unique_dates

    def truncate_timeline(self, text, truncation_config=None):
        """
        Truncate timeline based on configuration.
        
        Args:
            text: The timeline text to truncate
            truncation_config: Dict with keys:
                - 'mode': 'max_chars' or 'last_k_events'
                - 'max_chars': int (for max_chars mode, also used as safety limit for last_k_events)
                - 'k': int (for last_k_events mode)
        
        Returns:
            Truncated timeline string
        """
        if truncation_config is None:
            # Default: no truncation
            return str(text)
        
        text_str = str(text)
        mode = truncation_config.get('mode', 'max_chars')
        
        if mode == 'max_chars':
            max_chars = truncation_config.get('max_chars', 5000)
            if len(text_str) > max_chars:
                return text_str[:max_chars] + "... [TRUNCATED]"
            return text_str
        
        elif mode == 'last_k_events':
            initial_k = truncation_config.get('k', 10)
            # Get safety max_chars limit to prevent token overflow
            # Rough estimate: 1 token ≈ 4 characters, but can vary
            # Model max is 14588 tokens, so we use ~30000 chars to be safe (allows for prompt overhead)
            safety_max_chars = truncation_config.get('max_chars', 30000)
            
            # Pattern to match event markers: [YYYY-MM-DD HH:MM] |
            # Find all event start positions and extract dates
            # Pattern matches: [ followed by date/time, followed by ] |
            pattern = r'\[(\d{4}-\d{2}-\d{2})\s+\d{2}:\d{2}\]\s*\|'
            
            # Find all matches with their positions and dates
            matches = list(re.finditer(pattern, text_str))
            
            if len(matches) == 0:
                # No event markers found, return original text (but still apply safety limit)
                if len(text_str) > safety_max_chars:
                    return text_str[:safety_max_chars] + "... [TRUNCATED]"
                return text_str
            
            # Extract dates from matches
            # Build a list of (date, match) pairs
            date_match_pairs = []
            for match in matches:
                date = match.group(1)  # Extract the date (YYYY-MM-DD)
                date_match_pairs.append((date, match))
            
            # Find unique dates and their first occurrence
            seen_dates = set()
            unique_dates_list = []
            for date, match in date_match_pairs:
                if date not in seen_dates:
                    seen_dates.add(date)
                    unique_dates_list.append(date)
            
            # If we have fewer unique dates than requested k, just return all (with safety check)
            if len(unique_dates_list) <= initial_k:
                if len(text_str) > safety_max_chars:
                    return text_str[:safety_max_chars] + "... [TRUNCATED]"
                return text_str
            
            # Iteratively decrease k until the result fits within safety_max_chars
            current_k = initial_k
            while current_k > 0:
                # Take the last current_k unique dates
                first_date_in_last_k = unique_dates_list[-current_k]
                start_pos = None
                
                # Find the first match that belongs to the current_k-th unique date from the end
                for date, match in date_match_pairs:
                    if date == first_date_in_last_k:
                        start_pos = match.start()
                        break
                
                if start_pos is None:
                    # Should not happen, but if it does, try with fewer dates
                    current_k -= 1
                    continue
                
                truncated = text_str[start_pos:]
                
                # Check if the truncated text fits within the safety limit
                if len(truncated) <= safety_max_chars:
                    # Success! Return the truncated text
                    return truncated
                
                # Too long, decrease k and try again
                current_k -= 1
            
            # If we get here, even k=1 was too long, so apply character limit as fallback
            # Use the last unique date and truncate at character limit
            if len(unique_dates_list) > 0:
                last_date = unique_dates_list[-1]
                start_pos = None
                for date, match in date_match_pairs:
                    if date == last_date:
                        start_pos = match.start()
                        break
                
                if start_pos is not None:
                    truncated = text_str[start_pos:]
                    # Find the last event marker before the safety limit
                    safety_matches = list(re.finditer(pattern, truncated[:safety_max_chars]))
                    if safety_matches:
                        # Keep up to the last complete event before the limit
                        last_safe_match = safety_matches[-1]
                        # Find the end of this event (next newline or end of string)
                        event_end = truncated.find('\n', last_safe_match.end())
                        if event_end == -1:
                            event_end = last_safe_match.end()
                        return truncated[:event_end] + "\n... [TRUNCATED - even single date too long]"
                    else:
                        return truncated[:safety_max_chars] + "... [TRUNCATED]"
            
            # Final fallback: just truncate at character limit
            if len(text_str) > safety_max_chars:
                return text_str[:safety_max_chars] + "... [TRUNCATED]"
            return text_str
        
        else:
            # Unknown mode, return original
            return text_str

    def run_inference(self, task_names=None):
        # Determine which tasks to run
        tasks_to_run = self.valid_tasks
        if task_names:
            tasks_to_run = [t for t in self.valid_tasks if t['task_name'] in task_names]

        # Get experiments from config (default to ['no_image'] if not specified)
        experiments = self.cfg.get('experiments', ['no_image'])

        # Run each task for each experiment
        for task_info in tasks_to_run:
            for experiment in experiments:
                self._process_single_task(task_info, experiment)

    def _process_single_task(self, task_info, experiment='no_image'):
        task_name = task_info['task_name']
        source_csv = task_info['task_source_csv'] # Acts as Table ID (e.g., 'oncology')
        print(f"\n>>> Starting Task: {task_name} | Experiment: {experiment}")

        # --- BIGQUERY LOADING START ---
        # Construct Table ID: project.dataset.table
        dataset_id = "vista_bench_v1_1"
        full_table_id = f"{self.project_id}.{dataset_id}.{source_csv}"
        
        print(f"    Querying BigQuery table: {full_table_id} for task='{task_name}'...")
        
        query = f"""
            SELECT *
            FROM `{full_table_id}`
            WHERE task = @task
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
        
        # If timeline column doesn't exist, create it
        if not timeline_col:
            timeline_col = 'patient_string'
            df[timeline_col] = None
            print(f"    Creating new '{timeline_col}' column to store fetched timelines.")

        # --- COMMENTED OUT: Fetch patient timelines from meds database ---
        # This code is kept as skeleton for future use
        # print(f"    Fetching patient timelines from meds database...")
        # df[timeline_col] = df.apply(self._get_patient_timeline, axis=1)
        # print(f"    Fetched {len(df[df[timeline_col].notna()])} patient timelines.")

        # --- LOAD PATIENT TIMELINES FROM LOCAL CSV ---
        # Load local CSV file (same logic as run.py)
        csv_path = self.base_path / source_csv / f"{task_name}.csv"
        print(f"    Loading patient timelines from local CSV: {csv_path}")
        
        if not csv_path.exists():
            print(f"!!! Error: Local CSV file not found at {csv_path}")
            return
        
        try:
            csv_df = pd.read_csv(csv_path)
            print(f"    Loaded {len(csv_df)} rows from local CSV.")
            
            # Ensure person_id exists in both dataframes
            if 'person_id' not in df.columns:
                print(f"!!! Error: 'person_id' column not found in BigQuery data.")
                return
            
            if 'person_id' not in csv_df.columns:
                print(f"!!! Error: 'person_id' column not found in CSV data.")
                return
            
            # Find patient_timeline column in CSV (case-insensitive)
            csv_timeline_col = next((c for c in csv_df.columns if 'patient_string' in c.lower() or 'patient_timeline' in c.lower()), None)
            
            if csv_timeline_col is None:
                print(f"!!! Error: Patient timeline column not found in CSV.")
                return
            
            print(f"    Merging BigQuery data with CSV on 'person_id' to get patient timelines...")
            # Merge on person_id to get patient_timeline from CSV
            # Only merge the timeline column to avoid duplicate columns
            merge_df = csv_df[['person_id', csv_timeline_col]].copy()
            merge_df = merge_df.rename(columns={csv_timeline_col: timeline_col})
            
            # Drop existing timeline_col from df if it exists (we'll use CSV version)
            if timeline_col in df.columns:
                df = df.drop(columns=[timeline_col])
            
            # Merge with BigQuery data (left join to keep all BigQuery rows)
            df = df.merge(merge_df, on='person_id', how='inner')
            
            matched_count = df[timeline_col].notna().sum()
            print(f"    Matched {matched_count} out of {len(df)} rows with patient timelines from CSV.")
            
        except Exception as e:
            print(f"!!! Error loading/merging CSV: {e}")
            return

        # Count unique event dates (before truncation)
        # print(f"    Counting unique event dates...")
        df['unique_events'] = df[timeline_col].apply(self.count_unique_event_dates)

        # 2. Setup Resume Logic
        # We keep the local folder structure for results
        save_dir = self.results_base / source_csv / task_name / self.file_model_name
        save_dir.mkdir(parents=True, exist_ok=True)
        out_file = save_dir / f"{task_name}_results_{experiment}.csv"

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
        # Get truncation config from YAML
        truncation_config = self.cfg.get('timeline_truncation', None)
        df[timeline_col] = df[timeline_col].apply(lambda x: self.truncate_timeline(x, truncation_config))
        base_prompt_template = self.prompts_map.get(task_name, "[PATIENT_TIMELINE]")
        df['dynamic_prompt'] = df[timeline_col].apply(lambda x: base_prompt_template.replace('[PATIENT_TIMELINE]', str(x)))

        dataset = PromptDataset(df=df, prompt_col='dynamic_prompt', experiment=experiment, storage_client=self.storage_client) 
        loader = DataLoader(
            dataset,
            batch_size=self.cfg['runtime']['batch_size'],
            shuffle=False,
            prefetch_factor=2,
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
                    
                    # Check if image was actually used
                    image = item.get('image', None)
                    if image is not None:
                        # Handle both single image and list of images
                        if isinstance(image, list):
                            res_row['used_image'] = 1 if len(image) > 0 else 0
                        else:
                            res_row['used_image'] = 1
                    else:
                        res_row['used_image'] = 0
                    
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