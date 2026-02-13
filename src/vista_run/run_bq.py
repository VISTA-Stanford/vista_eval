import os
import argparse
import json
import yaml
import torch
import pandas as pd
import re
from pathlib import Path
from collections import defaultdict
from collections.abc import Mapping
from queue import Queue
from threading import Thread
from tqdm import tqdm
from torch.utils.data import DataLoader
from google.cloud import bigquery
from google.cloud import storage
# import meds_reader
# from meds_tools import patient_timeline
# from meds2text.ontology import OntologyDescriptionLookupTable

from vqa_dataset import PromptDataset, prompt_collate
from models import load_model_adapter
from data_tools.utils.query_utils import VISTA_BENCH_DATASET, fetch_task_data_from_bq
from data_tools.utils.meds_timeline_utils import (
    count_unique_event_dates,
    truncate_timeline,
)
from data_tools.utils.task_data_utils import (
    resolve_local_bq_cache_path,
    resolve_timeline_csv_path,
    resolve_timeline_csv_filename,
    find_bq_timeline_column,
    merge_bq_with_timeline_csv,
)
from data_tools.utils.csv_utils import append_to_csv as append_to_csv_util

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

        # Image-style prompts (used only for experiment 'no_timeline')
        image_prompts_path = os.path.join(
            self.base_path,
            self.cfg['paths'].get('image_prompts', 'tasks/image_valid_tasks.json')
        )
        self.image_prompts_map = {}
        if os.path.exists(image_prompts_path):
            with open(image_prompts_path, 'r') as f:
                self.image_prompts_map = json.load(f)

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

        # 6. Initialize retrieval (optional)
        retrieval_cfg = self.cfg.get("retrieval", {})
        if retrieval_cfg.get("enabled"):
            try:
                from retrieval import LocalPatientRetriever
                self.retriever = LocalPatientRetriever(
                    corpus_dir=retrieval_cfg["corpus_dir"],
                    cache_dir=retrieval_cfg["cache_dir"],
                )
            except ImportError as e:
                raise ImportError(
                    "retrieval.enabled is true but meds_mcp not installed. "
                    "Install with: pip install -e '.[retrieval]'"
                ) from e
        else:
            self.retriever = None
        self.retrieval_cfg = retrieval_cfg

        # 7. Constrained decoding configuration
        self.constrained_cfg = {
            "enabled": True,
            "mode": "task_mapping",   # task_mapping | binary_only
            "exclude_insufficient": True,
            "max_choices": 12,
        }
        user_constrained_cfg = self.cfg.get("constrained_decoding", {})
        if isinstance(user_constrained_cfg, dict):
            self.constrained_cfg.update(user_constrained_cfg)

    def _set_envs(self, model_dir):
        os.environ.update({
            "HF_HOME": model_dir,
            "TRANSFORMERS_CACHE": model_dir,
            "VLLM_CACHE_ROOT": model_dir,
            "TOKENIZERS_PARALLELISM": "false"
        })

    @staticmethod
    def _normalize_choice_text(text):
        return re.sub(r"\s+", " ", str(text).strip().lower())

    @staticmethod
    def _is_insufficient_choice(text):
        normalized = TaskOrchestrator._normalize_choice_text(text)
        return ("insufficient" in normalized) or ("missing data" in normalized)

    @staticmethod
    def _sorted_mapping_items(mapping):
        def _sort_key(item):
            key = item[0]
            try:
                return (0, float(key))
            except (TypeError, ValueError):
                return (1, str(key))
        return sorted(mapping.items(), key=_sort_key)

    def build_constrained_choices(self, task_info):
        """
        Build deterministic constrained choices per task.
        - Binary tasks preserve legacy behavior: ["Yes", "No"].
        - Non-binary tasks use task mapping labels (optionally excluding insufficient/missing labels).
        """
        if not self.constrained_cfg.get("enabled", True):
            return None

        mode = str(self.constrained_cfg.get("mode", "task_mapping")).strip().lower()
        is_binary = bool(task_info.get("is_binary", False))

        if mode == "binary_only":
            return ["Yes", "No"] if is_binary else None

        # Preserve existing binary behavior exactly.
        if is_binary:
            return ["Yes", "No"]

        mapping = task_info.get("mapping")
        if not isinstance(mapping, dict) or not mapping:
            return None

        exclude_insufficient = bool(self.constrained_cfg.get("exclude_insufficient", True))
        max_choices = self.constrained_cfg.get("max_choices", 12)

        choices = []
        seen = set()
        for _, value in self._sorted_mapping_items(mapping):
            if value is None:
                continue
            choice = str(value).strip()
            if not choice:
                continue
            if exclude_insufficient and self._is_insufficient_choice(choice):
                continue
            normalized = self._normalize_choice_text(choice)
            if normalized in seen:
                continue
            seen.add(normalized)
            choices.append(choice)

        if isinstance(max_choices, int) and max_choices > 0 and len(choices) > max_choices:
            print(
                f"    Constrained choices exceeded max_choices={max_choices}; "
                f"truncating from {len(choices)} to {max_choices}."
            )
            choices = choices[:max_choices]

        if len(choices) < 2:
            return None
        return choices

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
            # Data variants:
            # - no_report/timeline_only/report: use *_subsampled_no_img_report.csv AND require patient timeline merge
            # - no_timeline: use *_subsampled_no_img_report.csv BUT do NOT require patient timeline (image-only)
            # - all other experiments: use normal CSV and require patient timeline merge
            needs_report_timeline = any(e in ('no_report', 'timeline_only', 'report', 'retrieved_timeline') for e in experiments)
            needs_no_timeline = 'no_timeline' in experiments
            needs_normal = any(e not in ('no_report', 'timeline_only', 'report', 'no_timeline', 'retrieved_timeline') for e in experiments)

            loaded_normal = self._load_task_data(task_info, use_no_report_csv=False, require_timeline=True) if needs_normal else None
            loaded_no_report = self._load_task_data(task_info, use_no_report_csv=True, require_timeline=True) if needs_report_timeline else None
            loaded_no_timeline = self._load_task_data(task_info, use_no_report_csv=True, require_timeline=False) if needs_no_timeline else None
            for experiment in experiments:
                print(f"\n>>> Starting Task: {task_name} | Experiment: {experiment}")
                if experiment == 'no_timeline':
                    loaded = loaded_no_timeline
                elif experiment in ('no_report', 'timeline_only', 'report', 'retrieved_timeline'):
                    loaded = loaded_no_report
                else:
                    loaded = loaded_normal
                if loaded is None:
                    continue
                df, timeline_col, source_csv = loaded
                self._process_single_task_with_data(task_info, experiment, df, timeline_col)

    def _load_task_data(self, task_info, use_no_report_csv=False, require_timeline=True):
        """
        Query BigQuery once per task and merge with local CSV timelines.
        Returns (df, timeline_col, source_csv) or None on failure.
        Same data is reused for all experiments.
        When use_no_report_csv=True, loads from *_subsampled_no_img_report.csv (for no_report and report experiments).
        When require_timeline=False, we still use the local CSV to restrict to the subsampled cohort
        (via inner join on person_id), but we do NOT require a patient timeline column in the CSV.
        """
        task_name = task_info['task_name']
        source_csv = task_info['task_source_csv']
        use_subsampled = self.cfg.get('subsample', False)

        # --- LOAD FROM LOCAL CACHE OR BIGQUERY ---
        local_bq_path = resolve_local_bq_cache_path(self.base_path, source_csv)
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
            full_table_id = f"{self.project_id}.{VISTA_BENCH_DATASET}.{source_csv}"
            print(f"\n>>> Loading data for task: {task_name} (one BigQuery query for all experiments)")
            print(f"    Querying BigQuery table: {full_table_id}...")
            df = fetch_task_data_from_bq(self.bq_client, full_table_id, task_name)
            if df is None or df.empty:
                print(f"!!! No data found for task '{task_name}' in BigQuery.")
                return None
            print(f"    Loaded {len(df)} rows from BigQuery.")

        if 'index' not in df.columns:
            df['index'] = df.index

        timeline_col = find_bq_timeline_column(df)
        if timeline_col not in df.columns:
            df[timeline_col] = None

        # --- LOAD PATIENT TIMELINES FROM LOCAL CSV ---
        csv_path = resolve_timeline_csv_path(
            self.base_path, source_csv, task_name, use_subsampled, use_no_report_csv
        )
        if not csv_path.exists():
            # Try v1_2 path: base_path/v1_2/source_csv/filename
            csv_path_v1_2 = self.base_path / "v1_2" / source_csv / resolve_timeline_csv_filename(
                task_name, use_subsampled, use_no_report_csv
            )
            if csv_path_v1_2.exists():
                csv_path = csv_path_v1_2
                print(f"    Using v1_2 path: {csv_path}")
            else:
                print(f"!!! Error: Local CSV file not found at {csv_path}")
                print(f"    Also checked v1_2 path: {csv_path_v1_2}")
                return None
        print(f"    Loading patient timelines from local CSV: {csv_path}")
        try:
            csv_df = pd.read_csv(csv_path)
            print(f"    Loaded {len(csv_df)} rows from local CSV.")
            if 'person_id' not in df.columns:
                print(f"!!! Error: 'person_id' column not found in BigQuery data.")
                return None
            if 'person_id' not in csv_df.columns:
                print(f"!!! Error: 'person_id' column not found in CSV data.")
                return None
            if require_timeline:
                df = merge_bq_with_timeline_csv(df, csv_df, timeline_col, use_no_report_csv)
                if df is None:
                    print(f"!!! Error: Patient timeline column not found in CSV or merge failed.")
                    return None
                matched_count = df[timeline_col].notna().sum()
                print(f"    Matched {matched_count} out of {len(df)} rows with patient timelines.")
            else:
                keep_ids = csv_df[['person_id']].drop_duplicates()
                df = df.merge(keep_ids, on='person_id', how='inner')
                print(f"    Restricted to {len(df)} rows via person_id join (no timeline merge).")
        except Exception as e:
            print(f"!!! Error loading/merging CSV: {e}")
            return None

        if require_timeline:
            df['unique_events'] = df[timeline_col].apply(count_unique_event_dates)
            truncation_config = self.cfg.get('timeline_truncation', None)
            df[timeline_col] = df[timeline_col].apply(lambda x: truncate_timeline(x, truncation_config))
        else:
            df['unique_events'] = float("nan")
        return (df, timeline_col, source_csv)

    def _prepare_batch_for_inference(self, batch, existing_indices, constrained_choices):
        """
        Prepare a batch for inference (CPU work: create_template, prepare_inputs).
        Returns (new_items, inference_batches, max_input_tokens) or None if no new items.
        inference_batches: list of {"indices": [...], "inputs": ...}
        """
        new_items = [item for item in batch if item['raw_row']['index'] not in existing_indices]
        if not new_items:
            return None

        tokenizer = getattr(self.processor, "tokenizer", None) if getattr(self, "processor", None) else None
        if tokenizer is None:
            tokenizer = getattr(self.model, "tokenizer", None)

        def _count_text_tokens(text: str) -> int:
            if text is None:
                return 0
            text = str(text)
            if tokenizer is not None and hasattr(tokenizer, "encode"):
                try:
                    return len(tokenizer.encode(text, add_special_tokens=False))
                except (TypeError, Exception):
                    try:
                        return len(tokenizer.encode(text))
                    except Exception:
                        pass
            return len(text.split())

        def _count_input_tokens(inp) -> int:
            if isinstance(inp, dict):
                ids = inp.get("input_ids", None)
                if ids is not None and hasattr(ids, "shape"):
                    try:
                        return int(ids.shape[-1]) if len(ids.shape) >= 2 else int(ids.shape[0])
                    except Exception:
                        pass
                if "prompt" in inp:
                    return _count_text_tokens(inp["prompt"])
                if "text" in inp:
                    return _count_text_tokens(inp["text"])
                return 0
            if isinstance(inp, tuple) and len(inp) >= 1:
                return _count_text_tokens(inp[0])
            if isinstance(inp, str):
                return _count_text_tokens(inp)
            return 0

        max_input_tokens = 0
        inference_batches = []

        if 'gemma' in self.model_type:
            def _get_image_group_key(item):
                image = item.get('image', None)
                if image is None:
                    return 0
                return len(image) if isinstance(image, list) else 1

            groups = defaultdict(list)
            for i, item in enumerate(new_items):
                groups[_get_image_group_key(item)].append((i, item))

            for key in sorted(groups.keys()):
                indexed_items = groups[key]
                indices = [x[0] for x in indexed_items]
                items = [x[1] for x in indexed_items]
                group_inputs = []
                for item in items:
                    try:
                        if not isinstance(item, dict):
                            raise TypeError(f"Expected item to be a dict, got {type(item)}")
                        single_msg = self.adapter.create_template(item)
                        single_inp = self.adapter.prepare_inputs([single_msg], self.processor, self.model)
                        if isinstance(single_inp, list):
                            single_inp = single_inp[0] if single_inp else {}
                        if isinstance(single_inp, Mapping) and not isinstance(single_inp, dict):
                            single_inp = dict(single_inp)
                        elif not isinstance(single_inp, dict):
                            raise TypeError(f"Expected dict or Mapping, got {type(single_inp)}")
                        group_inputs.append(single_inp)
                    except Exception as e:
                        print(f"Error preparing item: {e}")
                        group_inputs.append({})
                batched_inputs = self.adapter.stack_inputs(group_inputs, self.model)
                inference_batches.append({"indices": indices, "inputs": batched_inputs})
                for inp in group_inputs:
                    token_count = _count_input_tokens(inp)
                    if isinstance(inp, dict) and inp.get("multi_modal_data") and inp["multi_modal_data"].get("image"):
                        images = inp["multi_modal_data"]["image"]
                        num_images = len(images) if isinstance(images, list) else 1
                        token_count += num_images * 256
                    max_input_tokens = max(max_input_tokens, token_count)
        else:
            messages = [self.adapter.create_template(item) for item in new_items]
            inputs = self.adapter.prepare_inputs(messages, self.processor, self.model)
            inference_batches.append({"indices": list(range(len(new_items))), "inputs": inputs})
            if isinstance(inputs, list):
                for inp in inputs:
                    max_input_tokens = max(max_input_tokens, _count_input_tokens(inp))
            else:
                max_input_tokens = max(max_input_tokens, _count_input_tokens(inputs))

        return (new_items, inference_batches, max_input_tokens)

    def _run_inference_on_batches(self, inference_batches, constrained_choices):
        """Run inference on prepared batches, return outputs in order of indices."""
        if not inference_batches:
            return []
        max_idx = max(i for b in inference_batches for i in b["indices"])
        output_list = [""] * (max_idx + 1)
        for batch in inference_batches:
            indices = batch["indices"]
            inputs = batch["inputs"]
            outputs = self.adapter.infer(
                self.model, self.processor, inputs,
                self.cfg['runtime']["max_new_tokens"],
                constrained_choices=constrained_choices
            )
            for i, out in zip(indices, outputs):
                output_list[i] = out
        return output_list

    def _process_single_task_with_data(self, task_info, experiment, df, timeline_col):
        """Run inference for one (task, experiment) using already-loaded task data."""
        task_name = task_info['task_name']
        source_csv = task_info['task_source_csv']
        constrained_choices = self.build_constrained_choices(task_info)
        if constrained_choices:
            print(f"    Constrained decoding enabled with {len(constrained_choices)} choices.")
        else:
            print("    Constrained decoding disabled for this task.")

        # We keep the local folder structure for results
        save_dir = self.results_base / source_csv / task_name / self.file_model_name
        save_dir.mkdir(parents=True, exist_ok=True)
        out_file = save_dir / f"{task_name}_results_{experiment}.csv"

        # Overwrite (don't resume) for retrieved_timeline experiment
        if experiment == "retrieved_timeline":
            out_file.unlink(missing_ok=True)

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
        df_exp = df.copy()
        if experiment == 'no_timeline':
            # no_timeline: image-only prompt from image_valid_tasks.json (no patient timeline)
            base_prompt_template = self.image_prompts_map.get(task_name, "")
            if not base_prompt_template:
                # Fallback if task missing in image_valid_tasks.json
                base_prompt_template = self.prompts_map.get(task_name, "")
            df_exp['dynamic_prompt'] = base_prompt_template
        else:
            base_prompt_template = self.prompts_map.get(task_name, "[PATIENT_TIMELINE]")
            if experiment == 'report':
                # report: timeline + "Radiology Report:" + report column (no images; uses _subsampled_no_img_report CSV)
                def timeline_with_report(row):
                    timeline = str(row[timeline_col]) if pd.notna(row[timeline_col]) else ""
                    report = str(row['report']) if 'report' in row and pd.notna(row.get('report')) else ""
                    combined = f"{timeline}\nRadiology Report: {report}".rstrip()
                    return base_prompt_template.replace('[PATIENT_TIMELINE]', combined)
                df_exp['dynamic_prompt'] = df_exp.apply(timeline_with_report, axis=1)
            elif experiment == 'retrieved_timeline':
                if self.retriever is None:
                    raise ValueError(
                        "retrieved_timeline experiment requires retrieval.enabled=true in config"
                    )
                from retrieval import run_iterative_retrieval_batch
                rc = self.retrieval_cfg
                max_rows = rc.get("max_rows")
                truncation_config = self.cfg.get("timeline_truncation", None)
                iterations_log_dir = rc.get("iterations_log_dir") or (
                    self.results_base / "retrieval_logs" / source_csv / task_name / self.file_model_name
                )
                save_log = rc.get("save_iterations_log", True)
                if max_rows:
                    df_exp = df_exp.head(max_rows).copy()
                if save_log:
                    Path(iterations_log_dir).mkdir(parents=True, exist_ok=True)
                prompts_and_logs = []
                batch_size = rc.get("retrieval_batch_size", 8)
                rows_list = list(df_exp.iterrows())
                for i in tqdm(range(0, len(rows_list), batch_size), desc="Retrieval"):
                    batch_rows = rows_list[i : i + batch_size]
                    batch_data = []
                    use_time_filter = rc.get("use_time_filter", False)
                    for _, r in batch_rows:
                        entry = {
                            "person_id": str(r["person_id"]),
                            "question": str(r.get("question", r.get("label_description", ""))),
                        }
                        if use_time_filter:
                            embed_time = r.get("embed_time")
                            if embed_time is not None and pd.notna(embed_time):
                                et = pd.to_datetime(embed_time, errors="coerce")
                                if pd.notna(et):
                                    entry["end_date"] = et.strftime("%Y-%m-%d")
                                    start_dt = et - pd.DateOffset(months=rc.get("months_before", 6))
                                    entry["start_date"] = start_dt.strftime("%Y-%m-%d")
                        batch_data.append(entry)
                    results = run_iterative_retrieval_batch(
                        self.retriever, self.adapter, self.model, self.processor,
                        batch_data=batch_data,
                        task_name=task_name,
                        max_iterations=rc.get("max_iterations", 3),
                        keywords_per_iteration=rc.get("keywords_per_iteration", 5),
                        records_per_keyword=rc.get("records_per_keyword", 5),
                    )
                    for (_, row), result in zip(batch_rows, results):
                        combined = truncate_timeline(result.timeline_str, truncation_config)
                        prompts_and_logs.append((base_prompt_template.replace("[PATIENT_TIMELINE]", combined), result))
                df_exp["dynamic_prompt"] = [p for p, _ in prompts_and_logs]
                if save_log:
                    csv_log_path = Path(iterations_log_dir) / "retrieval_keywords.csv"
                    log_rows = []
                    for (_, row), (_, result) in zip(df_exp.iterrows(), prompts_and_logs):
                        for log_entry in result.iterations_log:
                            log_rows.append({
                                "person_id": str(row["person_id"]),
                                "task": task_name,
                                "model": self.file_model_name,
                                "iteration": log_entry["iteration"],
                                # "keywords": ", ".join(log_entry["keywords"]),
                                "all_keywords_so_far": ", ".join(log_entry.get("all_keywords_so_far", [])),
                                # "keyword_reasoning": log_entry.get("keyword_reasoning", ""),
                                # "raw_model_output": log_entry.get("raw_model_output", ""),
                                "num_results": sum(log_entry.get("num_results_per_keyword", [])),
                                "total_unique": log_entry.get("total_unique_so_far", 0),
                            })
                    if log_rows:
                        pd.DataFrame(log_rows).to_csv(csv_log_path, mode="w", index=False)
            else:
                df_exp['dynamic_prompt'] = df_exp[timeline_col].apply(lambda x: base_prompt_template.replace('[PATIENT_TIMELINE]', str(x)))

        ct_dir = self.cfg.get('paths', {}).get('ct_dir')
        dataset = PromptDataset(df=df_exp, prompt_col='dynamic_prompt', experiment=experiment, storage_client=self.storage_client, model_type=self.model_type, ct_dir=ct_dir) 
        num_workers = 4
        loader = DataLoader(
            dataset,
            batch_size=self.cfg['runtime']['batch_size'],
            shuffle=False,
            prefetch_factor=8,
            collate_fn=prompt_collate,
            num_workers=num_workers,
            persistent_workers=num_workers > 0,
            pin_memory=True,
        )

        # 4. Inference Loop (producer-consumer: overlap prepare_inputs with GPU infer)
        results_buffer = []
        batch_counter = 0
        prefetch_queue = Queue(maxsize=2)
        sentinel = object()

        def producer():
            try:
                for batch in loader:
                    prepared = self._prepare_batch_for_inference(batch, existing_indices, constrained_choices)
                    if prepared is not None:
                        prefetch_queue.put(prepared)
            except Exception as e:
                print(f"Producer error: {e}")
            finally:
                prefetch_queue.put(sentinel)

        producer_thread = Thread(target=producer, daemon=True)
        producer_thread.start()

        pbar = tqdm(desc=f"Inference {task_name} | {experiment}")

        while True:
            got = prefetch_queue.get()
            if got is sentinel:
                break

            new_items, inference_batches, max_input_tokens = got

            try:
                outputs = self._run_inference_on_batches(inference_batches, constrained_choices)

                # Process outputs
                for item, out in zip(new_items, outputs):
                    # We drop the heavy timeline column before saving to CSV
                    if timeline_col in item['raw_row']:
                        res_row = item['raw_row'].drop(labels=[timeline_col]).to_dict()
                    else:
                        res_row = item['raw_row'].to_dict()
                    # if 'note_text' in res_row:
                    #     res_row = item['raw_row'].drop(labels=["note_text"]).to_dict()
                    # if 'patient_string' in res_row:
                    #     res_row = item['raw_row'].drop(labels=["patient_string"]).to_dict()
                    # if 'report' in res_row:
                    #     res_row = item['raw_row'].drop(labels=["report"]).to_dict()
                    # Handle both dict (with logprobs) and legacy string output
                    if isinstance(out, dict):
                        res_row['model_response'] = out.get("text", "")
                        res_row['cumulative_logprob'] = out.get("cumulative_logprob")
                        res_row['log_probs'] = out.get("log_probs")
                    else:
                        res_row['model_response'] = out
                        res_row['cumulative_logprob'] = None
                        res_row['log_probs'] = None
                    
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
                pbar.update(1)
                print(f"Batch {batch_counter}: Max input tokens = {max_input_tokens}")

                # Flush to disk every 10 batches
                if batch_counter % 10 == 0:
                    append_to_csv_util(out_file, results_buffer)
                    results_buffer = []

                # Clear CUDA cache periodically (reduced frequency to avoid sync stalls)
                if batch_counter % 20 == 0:
                    torch.cuda.empty_cache()

            except Exception as e:
                print(f"Error in batch: {e}")
                print(f"Max input tokens = {max_input_tokens} (error occurred)")
                # import traceback
                # traceback.print_exc()

        pbar.close()
        producer_thread.join(timeout=5)

        # 6. Final Save for remaining items
        if results_buffer:
            append_to_csv_util(out_file, results_buffer)

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
