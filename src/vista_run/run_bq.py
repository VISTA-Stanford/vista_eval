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
# import meds_reader
# from meds_tools import patient_timeline
# from meds2text.ontology import OntologyDescriptionLookupTable

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
        # meds_db_path = self.base_path / "thoracic_cohort_meds" / "vista_thoracic_cohort_v0_db"
        # ontology_path = self.base_path / "thoracic_cohort_meds" / "athena_omop_ontologies"
        
        # print(f"    Initializing meds database from: {meds_db_path}")
        # print(f"    Initializing ontology lookup from: {ontology_path}")
        
        # self.lookup = OntologyDescriptionLookupTable()
        # self.lookup.load(str(ontology_path))
        # self.meds_database = meds_reader.SubjectDatabase(str(meds_db_path))

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
        
        # try:
        #     # Fetch patient timeline
        #     df_patient = patient_timeline.get_described_events_window(
        #         database=self.meds_database,
        #         lookup_table=self.lookup,
        #         subject_id=subject_id,
        #         start_time=start_date if start_date else None,
        #         end_time=end_date
        #     )
            
        #     # Convert to string format
        #     patient_string = patient_timeline.get_llm_event_string(df_patient, include_text=True)
        #     return patient_string if patient_string else "No clinical events found for this period."
            
        # except Exception as e:
        #     return f"Error fetching timeline for subject_id {subject_id}: {str(e)}"

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

    def _get_first_4_rows(self, text):
        """
        Return the first 4 lines of the patient timeline.
        Used so every experiment (including no_timeline) includes the first 4 rows.
        """
        if pd.isna(text) or text is None:
            return ""
        text_str = str(text).strip()
        if not text_str:
            return ""
        lines = text_str.split("\n")
        return "\n".join(lines[:4])

    def truncate_timeline(self, text, truncation_config=None):
        """
        Truncate timeline based on configuration.
        Always preserves the first 4 rows (first unique event) before applying truncation.
        
        Args:
            text: The timeline text to truncate
            truncation_config: Dict with keys:
                - 'mode': 'max_chars' or 'last_k_events'
                - 'max_chars': int (for max_chars mode, also used as safety limit for last_k_events)
                - 'k': int (for last_k_events mode)
        
        Returns:
            Truncated timeline string
        """
        text_str = str(text)
        
        # Split into lines and preserve first 4 rows
        lines = text_str.split('\n')
        if len(lines) <= 4:
            # If timeline has 4 or fewer lines, return as-is
            return text_str
        
        # Keep first 4 rows (first unique event)
        # first_4_rows = '\n'.join(lines[:4])
        # remaining_text = '\n'.join(lines[4:])
        first_4_rows = ''
        remaining_text = text_str
        
        if truncation_config is None:
            # Default: no truncation, but still preserve first 4 rows
            return first_4_rows + '\n' + remaining_text
        
        mode = truncation_config.get('mode', 'max_chars')
        
        if mode == 'max_chars':
            max_chars = truncation_config.get('max_chars', 5000)
            # Reserve space for first 4 rows + separator
            first_4_chars = len(first_4_rows)
            remaining_max_chars = max_chars - first_4_chars - len('\n')
            
            if remaining_max_chars <= 0:
                # If first 4 rows already exceed max_chars, just return them
                return first_4_rows
            
            if len(remaining_text) > remaining_max_chars:
                truncated_remaining = remaining_text[:remaining_max_chars] + "... [TRUNCATED]"
            else:
                truncated_remaining = remaining_text
            
            return first_4_rows + '\n' + truncated_remaining
        
        elif mode == 'last_k_events':
            initial_k = truncation_config.get('k', 10)
            safety_max_chars = truncation_config.get('max_chars', 180000)
            
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
                
                # Always include first 4 rows + truncated rest (no duplication)
                rest_start = max(start_pos, len(first_4_rows) + 1)
                truncated = text_str[rest_start:]
                
                # Check if the truncated text fits within the safety limit
                if len(truncated) <= safety_max_chars:
                    # Success! Return first 4 rows + truncated text
                    return first_4_rows + "\n" + truncated
                
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
                    rest_start = max(start_pos, len(first_4_rows) + 1)
                    truncated = text_str[rest_start:]
                    # Find the last event marker before the safety limit
                    safety_matches = list(re.finditer(pattern, truncated[:safety_max_chars]))
                    if safety_matches:
                        # Keep up to the last complete event before the limit
                        last_safe_match = safety_matches[-1]
                        # Find the end of this event (next newline or end of string)
                        event_end = truncated.find('\n', last_safe_match.end())
                        if event_end == -1:
                            event_end = last_safe_match.end()
                        return first_4_rows + "\n" + truncated[:event_end] + "\n... [TRUNCATED - even single date too long]"
                    else:
                        return first_4_rows + "\n" + truncated[:safety_max_chars] + "... [TRUNCATED]"
            
            # Final fallback: just truncate at character limit (still prepend first 4 rows)
            rest_max = safety_max_chars - len(first_4_rows) - 1
            if len(remaining_text) > rest_max:
                return first_4_rows + "\n" + remaining_text[:rest_max] + "... [TRUNCATED]"
            return first_4_rows + "\n" + remaining_text
        
        else:
            # Unknown mode, return first 4 rows + remaining text
            return first_4_rows + '\n' + remaining_text

    def run_inference(self, task_names=None):
        # Determine which tasks to run
        tasks_to_run = self.valid_tasks
        if task_names:
            tasks_to_run = [t for t in self.valid_tasks if t['task_name'] in task_names]

        # Get experiments from config (default to ['no_image'] if not specified)
        experiments = self.cfg.get('experiments', ['no_image'])

        # For each task: load data (different CSV for no_report/report), then run all experiments
        for task_info in tasks_to_run:
            task_name = task_info['task_name']
            # no_report and report use _subsampled_no_img_report.csv; other experiments use normal CSV
            needs_report_csv = 'no_report' in experiments or 'report' in experiments
            needs_normal = any(e not in ('no_report', 'report') for e in experiments)
            loaded_normal = self._load_task_data(task_info, use_no_report_csv=False) if needs_normal else None
            loaded_no_report = self._load_task_data(task_info, use_no_report_csv=True) if needs_report_csv else None
            for experiment in experiments:
                print(f"\n>>> Starting Task: {task_name} | Experiment: {experiment}")
                if experiment in ('no_report', 'report'):
                    loaded = loaded_no_report
                else:
                    loaded = loaded_normal
                if loaded is None:
                    continue
                df, timeline_col, source_csv = loaded
                self._process_single_task_with_data(task_info, experiment, df, timeline_col)

    def _load_task_data(self, task_info, use_no_report_csv=False):
        """
        Query BigQuery once per task and merge with local CSV timelines.
        Returns (df, timeline_col, source_csv) or None on failure.
        Same data is reused for all experiments.
        When use_no_report_csv=True, loads from *_subsampled_no_img_report.csv (for no_report and report experiments).
        """
        task_name = task_info['task_name']
        source_csv = task_info['task_source_csv']

        # --- LOAD FROM LOCAL CACHE OR BIGQUERY ---
        local_bq_path = self.base_path / "bigquery_data_2_3" / source_csv
        if local_bq_path.exists():
            print(f"\n>>> Loading data for task: {task_name} (from local cache)")
            print(f"    Reading {local_bq_path}...")
            try:
                df_all = pd.read_csv(local_bq_path)
                df = df_all[df_all["task"] == task_name].copy()
            except Exception as e:
                print(f"!!! Error reading local BigQuery data for {task_name}: {e}")
                return None
            if df.empty:
                print(f"!!! No data found for task '{task_name}' in local file.")
                return None
            print(f"    Loaded {len(df)} rows from local cache.")
        else:
            dataset_id = "vista_bench_v1_1"
            full_table_id = f"{self.project_id}.{dataset_id}.{source_csv}"
            print(f"\n>>> Loading data for task: {task_name} (one BigQuery query for all experiments)")
            print(f"    Querying BigQuery table: {full_table_id}...")

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
                return None
            if df.empty:
                print(f"!!! No data found for task '{task_name}' in BigQuery.")
                return None
            print(f"    Loaded {len(df)} rows from BigQuery.")

        if 'index' not in df.columns:
            df['index'] = df.index

        timeline_col = next((c for c in df.columns if 'patient_string' in c.lower()), None)
        if not timeline_col:
            timeline_col = 'patient_string'
            df[timeline_col] = None

        # --- LOAD PATIENT TIMELINES FROM LOCAL CSV ---
        use_subsampled = self.cfg.get('subsample', False)
        if use_no_report_csv and use_subsampled:
            csv_filename = f"{task_name}_subsampled_no_img_report.csv"
        else:
            csv_filename = f"{task_name}_subsampled.csv" if use_subsampled else f"{task_name}.csv"
        csv_path = self.base_path / source_csv / csv_filename
        print(f"    Loading patient timelines from local CSV: {csv_path}")
        if not csv_path.exists():
            print(f"!!! Error: Local CSV file not found at {csv_path}")
            return None
        try:
            csv_df = pd.read_csv(csv_path)
            print(f"    Loaded {len(csv_df)} rows from local CSV.")
            if 'person_id' not in df.columns:
                print(f"!!! Error: 'person_id' column not found in BigQuery data.")
                return None
            if 'person_id' not in csv_df.columns:
                print(f"!!! Error: 'person_id' column not found in CSV data.")
                return None
            csv_timeline_col = next((c for c in csv_df.columns if 'patient_string' in c.lower() or 'patient_timeline' in c.lower()), None)
            if csv_timeline_col is None:
                print(f"!!! Error: Patient timeline column not found in CSV.")
                return None
            print(f"    Merging BigQuery data with CSV on 'person_id'...")
            merge_cols = ['person_id', csv_timeline_col]
            if use_no_report_csv and 'report' in csv_df.columns:
                merge_cols.append('report')
            merge_df = csv_df[merge_cols].copy()
            merge_df = merge_df.rename(columns={csv_timeline_col: timeline_col})
            if timeline_col in df.columns:
                df = df.drop(columns=[timeline_col])
            if 'report' in df.columns:
                df = df.drop(columns=['report'])
            df = df.merge(merge_df, on='person_id', how='inner')
            matched_count = df[timeline_col].notna().sum()
            print(f"    Matched {matched_count} out of {len(df)} rows with patient timelines.")
        except Exception as e:
            print(f"!!! Error loading/merging CSV: {e}")
            return None

        df['unique_events'] = df[timeline_col].apply(self.count_unique_event_dates)
        truncation_config = self.cfg.get('timeline_truncation', None)
        df[timeline_col] = df[timeline_col].apply(lambda x: self.truncate_timeline(x, truncation_config))
        return (df, timeline_col, source_csv)

    def _process_single_task_with_data(self, task_info, experiment, df, timeline_col):
        """Run inference for one (task, experiment) using already-loaded task data."""
        task_name = task_info['task_name']
        source_csv = task_info['task_source_csv']

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

        # 3. Prepare Prompt and Dataset (experiment-specific; use copy so shared df is unchanged)
        # Every experiment (including no_timeline) includes the first 4 rows of the patient timeline.
        base_prompt_template = self.prompts_map.get(task_name, "[PATIENT_TIMELINE]")
        df_exp = df.copy()
        if experiment == 'no_timeline':
            # no_timeline: only first 4 rows of timeline (no additional timeline)
            df_exp['dynamic_prompt'] = df_exp[timeline_col].apply(
                lambda x: base_prompt_template.replace("[PATIENT_TIMELINE]", self._get_first_4_rows(x))
            )
        elif experiment == 'report':
            # report: timeline + "Radiology Report:" + report column (no images; uses _subsampled_no_img_report CSV)
            def timeline_with_report(row):
                timeline = str(row[timeline_col]) if pd.notna(row[timeline_col]) else ""
                report = str(row['report']) if 'report' in row and pd.notna(row.get('report')) else ""
                combined = f"{timeline}\nRadiology Report: {report}".rstrip()
                return base_prompt_template.replace('[PATIENT_TIMELINE]', combined)
            df_exp['dynamic_prompt'] = df_exp.apply(timeline_with_report, axis=1)
        else:
            df_exp['dynamic_prompt'] = df_exp[timeline_col].apply(lambda x: base_prompt_template.replace('[PATIENT_TIMELINE]', str(x)))

        ct_dir = self.cfg.get('paths', {}).get('ct_dir')
        dataset = PromptDataset(df=df_exp, prompt_col='dynamic_prompt', experiment=experiment, storage_client=self.storage_client, model_type=self.model_type, ct_dir=ct_dir) 
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

        for batch in tqdm(loader, desc=f"Inference {task_name} | {experiment}"):
            # Skip items already processed
            new_items = [item for item in batch if item['raw_row']['index'] not in existing_indices]
            
            if not new_items:
                continue

            try:
                # Track maximum input token length for this batch
                max_input_tokens = 0
                # Token counting helper (works across adapters).
                # Priority:
                # - tensor dicts with input_ids (HF-style)
                # - dicts with prompt (vLLM-style)
                # - raw strings / (text, image) tuples (LMDeploy InternVL)
                # Falls back to whitespace tokenization if no tokenizer is available.
                tokenizer = None
                if getattr(self, "processor", None) is not None:
                    tokenizer = getattr(self.processor, "tokenizer", None)
                if tokenizer is None:
                    tokenizer = getattr(self.model, "tokenizer", None)

                def _count_text_tokens(text: str) -> int:
                    if text is None:
                        return 0
                    text = str(text)
                    if tokenizer is not None and hasattr(tokenizer, "encode"):
                        try:
                            # HuggingFace-style
                            return len(tokenizer.encode(text, add_special_tokens=False))
                        except TypeError:
                            # Some tokenizers don't accept add_special_tokens
                            return len(tokenizer.encode(text))
                        except Exception:
                            pass
                    # Fallback (approximate)
                    return len(text.split())

                def _count_input_tokens(inp) -> int:
                    # HF-style dict
                    if isinstance(inp, dict):
                        ids = inp.get("input_ids", None)
                        if ids is not None and hasattr(ids, "shape"):
                            try:
                                # [B, T] or [T]
                                if len(ids.shape) >= 2:
                                    return int(ids.shape[-1])
                                return int(ids.shape[0])
                            except Exception:
                                pass
                        if "prompt" in inp:
                            return _count_text_tokens(inp["prompt"])
                        if "text" in inp:
                            return _count_text_tokens(inp["text"])
                        return 0
                    # LMDeploy InternVL: either "text" or (text, image/images)
                    if isinstance(inp, tuple) and len(inp) >= 1:
                        return _count_text_tokens(inp[0])
                    if isinstance(inp, str):
                        return _count_text_tokens(inp)
                    return 0

                if 'gemma' in self.model_type:
                    # For Gemma models, we need to handle different scenarios:
                    # - Text but no images
                    # - Single image and text
                    # - Multiple images and text
                    # - Multiple images but no text
                    # 
                    # Check if we can batch items together or need to process individually
                    # Process individually if:
                    # 1. Any item has multiple images (list with > 1 image)
                    # 2. Mixed batch (some have images, some don't, and we have multiple images)
                    # Otherwise, try to batch
                    
                    has_multiple_images = False
                    has_single_images = False
                    has_no_images = False
                    
                    for item in new_items:
                        image = item.get('image', None)
                        if image is not None:
                            if isinstance(image, list):
                                if len(image) > 1:
                                    has_multiple_images = True
                                elif len(image) == 1:
                                    has_single_images = True
                            else:
                                has_single_images = True
                        else:
                            has_no_images = True
                    
                    # Determine if we should process individually or batch
                    # Process individually if:
                    # - Any item has multiple images (different structure)
                    # - Mixed batch with multiple images (complex stacking)
                    should_process_individually = (
                        has_multiple_images or
                        (has_multiple_images and (has_single_images or has_no_images))
                    )
                    
                    if should_process_individually:
                        # Process items individually to handle different structures
                        outputs = []
                        for item in new_items:
                            try:
                                # Validate item is a dict
                                if not isinstance(item, dict):
                                    raise TypeError(f"Expected item to be a dict, got {type(item)}")
                                
                                single_msg = self.adapter.create_template(item)
                                single_inp = self.adapter.prepare_inputs([single_msg], self.processor, self.model)
                                
                                # prepare_inputs may return a dict (single item) or list (multiple items)
                                # For single item, it should be a dict or dict-like (e.g., BatchFeature)
                                if isinstance(single_inp, list):
                                    # If it's a list with one item, extract it
                                    if len(single_inp) == 1:
                                        single_inp = single_inp[0]
                                    else:
                                        # Multiple items in list, process each
                                        for inp in single_inp:
                                            # Convert BatchFeature or other dict-like objects to dict if needed
                                            from collections.abc import Mapping
                                            if isinstance(inp, Mapping) and not isinstance(inp, dict):
                                                inp = dict(inp)
                                            elif not isinstance(inp, dict):
                                                raise TypeError(f"Expected input to be a dict or dict-like (Mapping), got {type(inp)}")
                                            single_output = self.adapter.infer(self.model, self.processor, inp, self.cfg['runtime']["max_new_tokens"])
                                            if isinstance(single_output, list):
                                                outputs.extend(single_output)
                                            else:
                                                outputs.append(single_output)
                                        continue
                                
                                # Convert BatchFeature or other dict-like objects to dict if needed
                                from collections.abc import Mapping
                                if isinstance(single_inp, Mapping) and not isinstance(single_inp, dict):
                                    single_inp = dict(single_inp)
                                elif not isinstance(single_inp, dict):
                                    raise TypeError(f"Expected prepare_inputs to return a dict or dict-like (Mapping) for single item, got {type(single_inp)}")
                                
                                # Calculate input tokens for this item
                                token_count = _count_input_tokens(single_inp)
                                # Optional: rough image-token bump for vLLM Gemma prompts (kept approximate)
                                if isinstance(single_inp, dict) and single_inp.get("multi_modal_data") and single_inp["multi_modal_data"].get("image"):
                                    images = single_inp["multi_modal_data"]["image"]
                                    num_images = len(images) if isinstance(images, list) else 1
                                    token_count += num_images * 256
                                max_input_tokens = max(max_input_tokens, token_count)
                                
                                single_output = self.adapter.infer(self.model, self.processor, single_inp, self.cfg['runtime']["max_new_tokens"])
                                # Ensure single_output is a list
                                if isinstance(single_output, list):
                                    outputs.extend(single_output)
                                else:
                                    outputs.append(single_output)
                            except Exception as e:
                                print(f"Error processing item individually: {e}")
                                import traceback
                                traceback.print_exc()
                                # Add empty string as placeholder to maintain alignment
                                outputs.append("")
                    
                    # Log maximum input tokens for individually processed batch
                    if should_process_individually:
                        print(f"Batch {batch_counter + 1}: Max input tokens = {max_input_tokens}")
                    else:
                        # All items have similar structure, batch them together
                        all_inputs = []
                        for item in new_items:
                            single_msg = self.adapter.create_template(item)
                            single_inp = self.adapter.prepare_inputs([single_msg], self.processor, self.model)
                            
                            # prepare_inputs returns dict for single item or list for multiple
                            if isinstance(single_inp, list):
                                # If list, extract the single item (should be only one)
                                if len(single_inp) == 1:
                                    single_inp = single_inp[0]
                                else:
                                    raise ValueError(f"Expected single input from prepare_inputs, got list with {len(single_inp)} items")
                            
                            # Convert BatchFeature or other dict-like objects to dict if needed
                            from collections.abc import Mapping
                            if isinstance(single_inp, Mapping) and not isinstance(single_inp, dict):
                                single_inp = dict(single_inp)
                            elif not isinstance(single_inp, dict):
                                raise TypeError(f"Expected prepare_inputs to return a dict or dict-like (Mapping), got {type(single_inp)}")
                            
                            all_inputs.append(single_inp)
                        
                        batched_inputs = self.adapter.stack_inputs(all_inputs, self.model)
                        
                        # Calculate max input tokens for batched inputs
                        for inp in all_inputs:
                            token_count = _count_input_tokens(inp)
                            # Optional: rough image-token bump for vLLM Gemma prompts (kept approximate)
                            if isinstance(inp, dict) and inp.get("multi_modal_data") and inp["multi_modal_data"].get("image"):
                                images = inp["multi_modal_data"]["image"]
                                num_images = len(images) if isinstance(images, list) else 1
                                token_count += num_images * 256
                            max_input_tokens = max(max_input_tokens, token_count)
                        
                        outputs = self.adapter.infer(self.model, self.processor, batched_inputs, self.cfg['runtime']["max_new_tokens"])
                        
                        # Log maximum input tokens for batched processing
                        print(f"Batch {batch_counter + 1}: Max input tokens = {max_input_tokens}")
                else:
                    messages = [self.adapter.create_template(item) for item in new_items]
                    inputs = self.adapter.prepare_inputs(messages, self.processor, self.model)
                    
                    # Calculate max input tokens
                    if isinstance(inputs, list):
                        for inp in inputs:
                            max_input_tokens = max(max_input_tokens, _count_input_tokens(inp))
                    else:
                        max_input_tokens = max(max_input_tokens, _count_input_tokens(inputs))
                    
                    outputs = self.adapter.infer(self.model, self.processor, inputs, self.cfg['runtime']["max_new_tokens"])
                    
                    # Log maximum input tokens for non-gemma models
                    print(f"Batch {batch_counter + 1}: Max input tokens = {max_input_tokens}")
                
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
                if batch_counter % 10 == 0:
                    self._append_to_csv(out_file, results_buffer)
                    results_buffer = [] 

                # Clear CUDA cache periodically
                if batch_counter % 5 == 0:
                    torch.cuda.empty_cache()

            except Exception as e:
                print(f"Error in batch: {e}")
                # Still log batch info even on error (batch_counter already incremented)
                print(f"Batch {batch_counter}: Max input tokens = {max_input_tokens} (error occurred)")
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