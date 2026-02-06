"""
Subsample from BigQuery data (CSV export at vista_bench/bigquery_data_2_3) instead of
task CSVs. Same logic as subsample_csv.py (CT in bucket, excluded labels, task groups)
but source rows are from the BQ CSV files with split == 'test' only.
"""
import warnings
from pathlib import Path

# BigQuery falls back to REST when the Storage module is not installed; suppress the warning.
# warnings.filterwarnings(
#     "ignore",
#     message=".*BigQuery Storage module not found.*",
#     category=UserWarning,
# )

import numpy as np
import pandas as pd

from data_tools.utils.config_utils import load_tasks_from_config, get_bq_client
from data_tools.utils.ct_utils import filter_person_ids_by_bucket_existence
from data_tools.utils.query_utils import (
    fetch_person_id_nifti_paths,
    check_ct_available_batch,
    get_notes_from_bq,
    DEFAULT_NOTE_DATASET,
    DEFAULT_NOTE_TABLE,
)
from data_tools.utils.task_utils import group_related_tasks, load_task_mappings

# GCS bucket settings (same as subsample_csv / download_subsampled_ct)
DEFAULT_GCS_BUCKET_NAME = "su-vista-uscentral1"
DEFAULT_GCS_PREFIX = "chaudhari_lab/ct_data/ct_scans/vista/nov25"

# Column name for train/test split in BQ CSV
SPLIT_COL = "split"
SPLIT_TEST = "test"
TASK_COL = "task"


def _has_reportable_note(bq_client, person_id, accession_number, note_dataset, note_table):
    """
    Return True if there exists a note in the note table for (person_id, _accession_number)
    whose note_text does NOT contain the string "non-reportable". Same idea as get_report_note tooling.
    """
    try:
        notes_df = get_notes_from_bq(
            bq_client,
            int(person_id),
            str(accession_number).strip(),
            dataset=note_dataset,
            table=note_table,
        )
    except Exception:
        return False
    if notes_df.empty or "note_text" not in notes_df.columns:
        return False
    # At least one note must have non-null note_text that does not contain "non-reportable"
    note_text = notes_df["note_text"]
    has_reportable = note_text.notna() & (
        ~note_text.astype(str).str.contains("non-reportable", case=False, na=False)
    )
    return has_reportable.any()


def _is_excluded_label(label_val, excluded_labels):
    if excluded_labels is None:
        return False
    if pd.isna(label_val):
        return False
    if label_val in excluded_labels:
        return True
    if str(label_val) in [str(x) for x in excluded_labels]:
        return True
    try:
        numeric_val = float(label_val)
        if numeric_val in [
            float(x) for x in excluded_labels if isinstance(x, (int, float))
        ]:
            return True
    except (ValueError, TypeError):
        pass
    return False


def process_one_task_from_df(
    task_name,
    table_name,
    df_task,
    target_n,
    excluded_labels,
    overwrite,
    bq_dataset_id,
    task_table_map,
    selected_person_ids,
    base_dir,
    gcs_bucket_name,
    gcs_prefix,
    note_dataset=DEFAULT_NOTE_DATASET,
    note_table=DEFAULT_NOTE_TABLE,
):
    """
    Run subsampling for one task using an already-filtered DataFrame (split=='test', task==task_name).
    Same CT/bucket and sampling logic as subsample_csv.process_one_csv.
    Returns (status, task_name, message).
    """
    base_path = Path(base_dir)
    out_dir = base_path / table_name
    out_dir.mkdir(parents=True, exist_ok=True)
    new_path = out_dir / f"{task_name}_subsampled.csv"

    if new_path.exists() and not overwrite:
        return (
            "skip",
            task_name,
            f"output file {new_path.name} already exists (use overwrite=True to replace)",
        )

    bq_table_name = task_table_map.get(task_name)
    if bq_table_name is None:
        return ("skip", task_name, f"no BigQuery table mapping for task '{task_name}'")

    label_col = next((c for c in df_task.columns if c.lower() == "label"), None)
    if label_col is None:
        return ("skip", task_name, f"no 'label' col. cols={list(df_task.columns)}")

    df = df_task.copy()
    if excluded_labels:
        mask = df[label_col].apply(
            lambda x: not _is_excluded_label(x, excluded_labels)
        )
        df = df[mask]
    if len(df) == 0:
        return ("skip", task_name, "all rows excluded (insufficient follow-up labels)")

    person_id_col = next((c for c in df.columns if c.lower() == "person_id"), None)
    if person_id_col is None:
        return ("skip", task_name, f"no 'person_id' col. cols={list(df.columns)}")

    if selected_person_ids is not None and len(selected_person_ids) > 0:
        selected_strs = {str(pid) for pid in selected_person_ids}
        df = df[df[person_id_col].astype(str).isin(selected_strs)]
        if len(df) == 0:
            return (
                "skip",
                task_name,
                f"no rows matching pre-selected person_ids for task '{task_name}'",
            )

    # Require _accession_number present and not null/empty
    accession_col = next((c for c in df.columns if c == "_accession_number"), None)
    if accession_col is None:
        return ("skip", task_name, "no '_accession_number' column")
    df = df[df[accession_col].notna() & (df[accession_col].astype(str).str.strip() != "")]
    if len(df) == 0:
        return ("skip", task_name, "no rows with non-null, non-empty _accession_number")

    # Keep only rows where (person_id, _accession_number) has a note whose note_text does not contain "non-reportable"
    unique_pairs = df[[person_id_col, accession_col]].drop_duplicates()
    bq_client = get_bq_client()
    valid_pairs = set()
    for _, row in unique_pairs.iterrows():
        pid, acc = row[person_id_col], row[accession_col]
        # if _has_reportable_note(bq_client, pid, acc, note_dataset, note_table):
        valid_pairs.add((str(pid), str(acc).strip()))
    df["_pid_acc"] = list(zip(df[person_id_col].astype(str), df[accession_col].astype(str).str.strip()))
    df = df[df["_pid_acc"].isin(valid_pairs)].drop(columns=["_pid_acc"])
    if len(df) == 0:
        return (
            "skip",
            task_name,
            "no rows with reportable note (note_text without 'non-reportable')",
        )

    unique_person_ids = df[person_id_col].dropna().unique()
    all_person_ids = []
    for pid in unique_person_ids:
        try:
            if isinstance(pid, (np.ndarray, pd.Series)):
                pid = pid.item() if hasattr(pid, "item") else pid[0]
            elif isinstance(pid, (list, tuple)):
                pid = pid[0] if len(pid) > 0 else None
            if pid is not None:
                try:
                    if np.isnan(float(pid)):
                        continue
                except (ValueError, TypeError):
                    pass
                all_person_ids.append(pid)
        except Exception:
            continue

    if not all_person_ids:
        return ("skip", task_name, "no person_ids in filtered df")

    print(f"    Checking CT availability for {len(all_person_ids)} person_ids (task '{task_name}')...")
    available_person_ids = check_ct_available_batch(
        all_person_ids, dataset_id=bq_dataset_id, table_name=bq_table_name
    )
    print(f"    Found {len(available_person_ids)} person_ids with non-null nifti_path in BQ")

    if not available_person_ids:
        return ("skip", task_name, "no person_ids with nifti_path in BigQuery")

    path_pairs = []
    nifti_path_col = next((c for c in df.columns if c.lower() == "nifti_path"), None)
    local_path_col = next((c for c in df.columns if c.lower() == "local_path"), None)
    path_col = nifti_path_col or local_path_col
    available_strs = {str(pid) for pid in available_person_ids}
    if path_col and path_col in df.columns:
        df_paths = df[df[person_id_col].astype(str).isin(available_strs)][
            [person_id_col, path_col]
        ].dropna(subset=[path_col])
        for _, row in df_paths.iterrows():
            path_pairs.append((row[person_id_col], row[path_col]))
    if not path_pairs:
        path_pairs = fetch_person_id_nifti_paths(
            list(available_person_ids), bq_dataset_id, bq_table_name
        )

    available_person_ids = filter_person_ids_by_bucket_existence(
        available_person_ids, path_pairs, bucket_name=gcs_bucket_name, prefix=gcs_prefix
    )
    print(
        f"    After GCS bucket check: {len(available_person_ids)} person_ids with CT present in bucket"
    )

    if not available_person_ids:
        return ("skip", task_name, "no person_ids with nifti_path present in GCS bucket")

    unique_ids_str = list({str(pid) for pid in available_person_ids})
    if selected_person_ids is not None and len(selected_person_ids) > 0:
        selected_strs = set(unique_ids_str)
    else:
        n_select = min(target_n, len(unique_ids_str))
        rng = np.random.default_rng(42)
        selected_strs = set(rng.choice(unique_ids_str, size=n_select, replace=False).tolist())

    mask = df[person_id_col].astype(str).isin(selected_strs)
    subsampled_df = df[mask]
    n_person_ids = subsampled_df[person_id_col].nunique()

    subsampled_df.to_csv(new_path, index=False)
    return (
        "ok",
        task_name,
        f"wrote {new_path.name} ({len(subsampled_df)} rows, {n_person_ids} person_ids)",
    )


def subsample_from_bq_parallel(
    base_path,
    target_n=50,
    config_path=None,
    valid_tasks_json_path=None,
    overwrite=False,
    bq_dataset_id="vista_bench_v1_1",
    local_bq_data_dir=None,
    gcs_bucket_name=DEFAULT_GCS_BUCKET_NAME,
    gcs_prefix=DEFAULT_GCS_PREFIX,
    note_dataset=DEFAULT_NOTE_DATASET,
    note_table=DEFAULT_NOTE_TABLE,
):
    """
    Subsample from BigQuery data CSVs (vista_bench/bigquery_data_2_3). Only rows with
    split == 'test' are considered. Requires _accession_number non-null and a note in the
    note table whose note_text does not contain "non-reportable". For each task, same
    logic as subsample_csv: exclude labels, require CT (nifti_path) in GCS bucket,
    sample up to target_n, respect task groups. Writes to base_path / {table_name} / {task_name}_subsampled.csv.
    """
    base_dir = Path(base_path)
    if local_bq_data_dir is None:
        local_bq_data_dir = base_dir / "bigquery_data_2_3"
    else:
        local_bq_data_dir = Path(local_bq_data_dir)
    print(f"Using local BigQuery data from: {local_bq_data_dir}")
    print(f"Filtering to split == '{SPLIT_TEST}' only")

    task_exclusions = {}
    task_table_map = {}
    if valid_tasks_json_path:
        task_exclusions, task_table_map = load_task_mappings(valid_tasks_json_path)
        if task_exclusions:
            print(f"Loaded exclusions for {len(task_exclusions)} tasks from {valid_tasks_json_path}")
        if task_table_map:
            print(f"Loaded BigQuery table mappings for {len(task_table_map)} tasks")

    valid_tasks = None
    if config_path:
        valid_tasks = load_tasks_from_config(config_path)
        if valid_tasks:
            print(f"Loaded {len(valid_tasks)} tasks from config: {sorted(valid_tasks)}")
        else:
            print(
                "Warning: No tasks found in config or error loading config. Processing all tasks in BQ files."
            )

    # Tasks we will process: from config if set, else all that have a table mapping
    tasks_to_process = sorted(valid_tasks) if valid_tasks else sorted(task_table_map.keys())
    if valid_tasks:
        tasks_to_process = [t for t in tasks_to_process if t in valid_tasks]
    if not tasks_to_process:
        print("No tasks to process.")
        return

    task_groups = group_related_tasks(set(tasks_to_process))

    # Group by table (BQ file) so we read each file once. Order tasks so that within
    # each group, the first task is processed first (needed for selected_person_ids reuse).
    table_to_tasks = {}
    for task_name in tasks_to_process:
        table_name = task_table_map.get(task_name)
        if table_name is None:
            continue
        if table_name not in table_to_tasks:
            table_to_tasks[table_name] = []
        table_to_tasks[table_name].append(task_name)
    # Order tasks so that within each group, the first task is processed before the rest.
    for table_name in table_to_tasks:
        task_list = set(table_to_tasks[table_name])
        ordered = []
        for base_name, group_tasks in task_groups.items():
            for t in group_tasks:
                if t in task_list and t not in ordered:
                    ordered.append(t)
        for t in table_to_tasks[table_name]:
            if t not in ordered:
                ordered.append(t)
        table_to_tasks[table_name] = ordered

    if task_groups:
        print(f"Related task groups (same patients per group): {list(task_groups.keys())}")
        for base_name, group_tasks in task_groups.items():
            print(f"  {base_name}: {group_tasks}")

    selected_person_ids_map = {}
    ok = skip = err = 0

    for table_name, task_list in table_to_tasks.items():
        bq_file = local_bq_data_dir / table_name
        if not bq_file.exists():
            print(f"[SKIP] BQ file not found: {bq_file}")
            continue
        print(f"\nReading BQ table CSV: {table_name}")
        try:
            df_all = pd.read_csv(
                bq_file, sep=None, engine="python", on_bad_lines="warn"
            )
        except Exception as e:
            print(f"[ERR]  Failed to read {bq_file}: {e}")
            continue
        if SPLIT_COL not in df_all.columns:
            print(f"[SKIP] No '{SPLIT_COL}' column in {table_name}. cols={list(df_all.columns)}")
            continue
        if TASK_COL not in df_all.columns:
            print(f"[SKIP] No '{TASK_COL}' column in {table_name}. cols={list(df_all.columns)}")
            continue
        df_test = df_all[df_all[SPLIT_COL].astype(str).str.strip().str.lower() == SPLIT_TEST.lower()]
        print(f"    Rows with split=='test': {len(df_test)} (of {len(df_all)})")

        for task_name in task_list:
            df_task = df_test[df_test[TASK_COL].astype(str) == task_name]
            if len(df_task) == 0:
                skip += 1
                print(f"[SKIP] {task_name}: no rows with task=='{task_name}' and split=='test'")
                continue

            excluded_labels = task_exclusions.get(task_name, set())
            selected_person_ids = None
            for base_name, group_tasks in task_groups.items():
                if task_name in group_tasks:
                    if task_name == group_tasks[0]:
                        break
                    selected_person_ids = selected_person_ids_map.get(base_name)
                    break

            is_first_in_group = False
            base_name_for_group = None
            for base_name, group_tasks in task_groups.items():
                if task_name == group_tasks[0]:
                    is_first_in_group = True
                    base_name_for_group = base_name
                    break

            print(f"\nProcessing task: {task_name}")
            result = process_one_task_from_df(
                task_name=task_name,
                table_name=table_name,
                df_task=df_task,
                target_n=target_n,
                excluded_labels=excluded_labels,
                overwrite=overwrite,
                bq_dataset_id=bq_dataset_id,
                task_table_map=task_table_map,
                selected_person_ids=selected_person_ids,
                base_dir=str(base_dir),
                gcs_bucket_name=gcs_bucket_name,
                gcs_prefix=gcs_prefix,
                note_dataset=note_dataset,
                note_table=note_table,
            )
            status, _, msg = result
            if status == "ok":
                ok += 1
                print(f"[OK]   {task_name}: {msg}")
                if is_first_in_group and base_name_for_group is not None:
                    out_path = base_dir / table_name / f"{task_name}_subsampled.csv"
                    if out_path.exists():
                        try:
                            sub_df = pd.read_csv(out_path, nrows=0)
                            pid_col = next(
                                (c for c in sub_df.columns if c.lower() == "person_id"), None
                            )
                            if pid_col:
                                sub_df = pd.read_csv(out_path, usecols=[pid_col])
                                selected_person_ids_map[base_name_for_group] = set(
                                    sub_df[pid_col].dropna().astype(str).unique()
                                )
                                print(
                                    f"  Stored {len(selected_person_ids_map[base_name_for_group])} person_ids for reuse in group"
                                )
                        except Exception as e:
                            print(f"  Error reading subsampled CSV for group: {e}")
            elif status == "skip":
                skip += 1
                print(f"[SKIP] {task_name}: {msg}")
            else:
                err += 1
                print(f"[ERR]  {task_name}: {msg}")

    print(f"\nDone. OK={ok}, SKIP={skip}, ERR={err}")


if __name__ == "__main__":
    PATH = "/home/rdcunha/vista_project/vista_bench"
    CONFIG_PATH = "/home/rdcunha/vista_project/vista_eval_vlm/configs/all_tasks.yaml"
    VALID_TASKS_JSON_PATH = "/home/rdcunha/vista_project/vista_bench/tasks/valid_tasks.json"
    subsample_from_bq_parallel(
        PATH,
        target_n=100,
        config_path=CONFIG_PATH,
        valid_tasks_json_path=VALID_TASKS_JSON_PATH,
        overwrite=True,
    )
