import os
import csv
import sys
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd

from data_tools.utils.config_utils import load_tasks_from_config
from data_tools.utils.ct_utils import filter_person_ids_by_bucket_existence
from data_tools.utils.query_utils import (
    fetch_person_id_nifti_paths,
    check_ct_available_batch,
)
from data_tools.utils.task_utils import group_related_tasks, load_task_mappings

csv.field_size_limit(sys.maxsize)

# GCS bucket settings for NIfTI CT scans (same as download_subsampled_ct.py)
DEFAULT_GCS_BUCKET_NAME = "su-vista-uscentral1"
DEFAULT_GCS_PREFIX = "chaudhari_lab/ct_data/ct_scans/vista/nov25"


def process_one_csv(args):
    """Worker: read one CSV, sample up to target_n person_ids that have CT (nifti_path), write _subsampled.csv.
    Only counts nifti_path as present if the CT scan actually exists in the GCS bucket (same check as download_subsampled_ct).
    No class balancing - representative of the dataset.
    If selected_person_ids is provided (for related tasks, e.g. died_of_cancer_2_yr reusing 1_yr patients),
    filter to those person_ids and keep only rows with CT - same patients across the group."""
    (
        csv_path,
        target_n,
        excluded_labels,
        overwrite,
        bq_dataset_id,
        task_table_map,
        selected_person_ids,
        local_bq_data_dir,
        gcs_bucket_name,
        gcs_prefix,
    ) = args
    csv_path = Path(csv_path)

    if csv_path.stem.endswith("_subsampled"):
        return ("skip", str(csv_path), "already subsampled")

    new_path = csv_path.with_name(f"{csv_path.stem}_subsampled.csv")
    if new_path.exists() and not overwrite:
        return (
            "skip",
            str(csv_path),
            f"output file {new_path.name} already exists (use overwrite=True to replace)",
        )

    task_name = csv_path.stem
    bq_table_name = task_table_map.get(task_name)

    if bq_table_name is None:
        return ("skip", str(csv_path), f"no BigQuery table mapping found for task '{task_name}'")

    try:
        df = pd.read_csv(csv_path, sep=None, engine="python", on_bad_lines="warn")

        label_col = next((c for c in df.columns if c.lower() == "label"), None)
        if label_col is None:
            return ("skip", str(csv_path), f"no 'label' col. cols={list(df.columns)}")

        if excluded_labels:

            def is_excluded_label(label_val):
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

            mask = df[label_col].apply(lambda x: not is_excluded_label(x))
            df = df[mask]

            if len(df) == 0:
                return ("skip", str(csv_path), "all rows excluded (insufficient follow-up labels)")

        person_id_col = next((c for c in df.columns if c.lower() == "person_id"), None)
        if person_id_col is None:
            return ("skip", str(csv_path), f"no 'person_id' col. cols={list(df.columns)}")

        if selected_person_ids is not None and len(selected_person_ids) > 0:
            selected_strs = {str(pid) for pid in selected_person_ids}
            df = df[df[person_id_col].astype(str).isin(selected_strs)]
            if len(df) == 0:
                return (
                    "skip",
                    str(csv_path),
                    f"no rows matching pre-selected person_ids for task '{task_name}'",
                )
            print(
                f"    Filtered to {len(df)} rows matching pre-selected person_ids (same patients as related task)"
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

        print(f"    Checking CT availability for {len(all_person_ids)} person_ids (task '{task_name}')...")
        available_person_ids = check_ct_available_batch(
            all_person_ids, dataset_id=bq_dataset_id, table_name=bq_table_name
        )
        print(f"    Found {len(available_person_ids)} person_ids with non-null nifti_path in BQ")

        if not available_person_ids:
            return ("skip", str(csv_path), "no person_ids with nifti_path in BigQuery")

        path_pairs = []
        nifti_path_col = next((c for c in df.columns if c.lower() == "nifti_path"), None)
        local_path_col = next((c for c in df.columns if c.lower() == "local_path"), None)
        path_col = nifti_path_col or local_path_col
        available_strs = {str(pid) for pid in available_person_ids}
        if path_col:
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
            return ("skip", str(csv_path), "no person_ids with nifti_path present in GCS bucket")

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
            str(csv_path),
            f"wrote {new_path.name} ({len(subsampled_df)} rows, {n_person_ids} person_ids)",
        )

    except Exception as e:
        return ("err", str(csv_path), repr(e))


def subsample_csvs_parallel(
    base_path,
    target_n=50,
    workers=None,
    config_path=None,
    valid_tasks_json_path=None,
    overwrite=False,
    bq_dataset_id="vista_bench_v1_1",
    local_bq_data_dir=None,
    gcs_bucket_name=DEFAULT_GCS_BUCKET_NAME,
    gcs_prefix=DEFAULT_GCS_PREFIX,
):
    """Subsample each task CSV to up to target_n person_ids that have a CT scan (nifti_path) present in the GCS bucket.
    Only person_ids whose nifti_path exists in the bucket (same check as download_subsampled_ct) are counted as having CT.
    When local_bq_data_dir is set (default: base_path/bigquery_data_2_3), reads from local files there (same name as BQ table).
    Uses tasks from config_path (e.g. all_tasks.yaml). No class balancing - representative of the dataset."""
    base_dir = Path(base_path)
    if local_bq_data_dir is None:
        local_bq_data_dir = str(base_dir / "bigquery_data_2_3")
    print(f"Using local BigQuery data from: {local_bq_data_dir}")

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
                "Warning: No tasks found in config or error loading config. Processing all CSVs."
            )

    all_csv_files = [p for p in base_dir.rglob("*.csv") if not p.stem.endswith("_subsampled")]

    if valid_tasks:
        csv_files = []
        for csv_path in all_csv_files:
            task_name = csv_path.stem
            if task_name in valid_tasks:
                csv_files.append(csv_path)
        print(
            f"Filtered to {len(csv_files)} CSVs matching tasks from config (out of {len(all_csv_files)} total)."
        )
    else:
        csv_files = all_csv_files

    if not csv_files:
        print("No CSVs found.")
        return

    if workers is None:
        workers = max(1, (os.cpu_count() or 2) - 1)

    task_names = {p.stem for p in csv_files}
    task_groups = group_related_tasks(task_names)
    if task_groups:
        print(f"Related task groups (same patients per group): {list(task_groups.keys())}")
        for base_name, group_tasks in task_groups.items():
            print(f"  {base_name}: {group_tasks}")

    selected_person_ids_map = {}
    for base_name, group_tasks in task_groups.items():
        first_task = group_tasks[0]
        first_csv = next((p for p in csv_files if p.stem == first_task), None)
        if first_csv is None:
            continue
        print(f"\nProcessing first task in group '{base_name}': {first_task}")
        excluded_labels = task_exclusions.get(first_task, set())
        args = (
            str(first_csv),
            target_n,
            excluded_labels,
            overwrite,
            bq_dataset_id,
            task_table_map,
            None,
            local_bq_data_dir,
            gcs_bucket_name,
            gcs_prefix,
        )
        result = process_one_csv(args)
        if result[0] == "ok":
            subsampled_path = first_csv.with_name(f"{first_csv.stem}_subsampled.csv")
            if subsampled_path.exists():
                try:
                    sub_df = pd.read_csv(subsampled_path)
                    pid_col = next((c for c in sub_df.columns if c.lower() == "person_id"), None)
                    if pid_col:
                        selected_person_ids_map[base_name] = set(
                            sub_df[pid_col].dropna().astype(str).unique()
                        )
                        print(
                            f"  Stored {len(selected_person_ids_map[base_name])} person_ids for reuse in {group_tasks[1:]}"
                        )
                except Exception as e:
                    print(f"  Error reading subsampled CSV: {e}")

    tasks = []
    processed_first = set()
    for csv_path in csv_files:
        task_name = csv_path.stem
        excluded_labels = task_exclusions.get(task_name, set())
        selected_person_ids = None
        for base_name, group_tasks in task_groups.items():
            if task_name in group_tasks:
                if task_name == group_tasks[0]:
                    processed_first.add(task_name)
                    break
                selected_person_ids = selected_person_ids_map.get(base_name)
                break
        tasks.append(
            (
                str(csv_path),
                target_n,
                excluded_labels,
                overwrite,
                bq_dataset_id,
                task_table_map,
                selected_person_ids,
                local_bq_data_dir,
                gcs_bucket_name,
                gcs_prefix,
            )
        )

    remaining = [t for t in tasks if Path(t[0]).stem not in processed_first]
    print(
        f"\nProcessing {len(remaining)} remaining tasks (up to {target_n} person_ids each, same patients within groups)..."
    )
    ok = skip = err = 0
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(process_one_csv, t) for t in remaining]
        for fut in as_completed(futures):
            status, path, msg = fut.result()
            name = Path(path).name
            if status == "ok":
                ok += 1
                print(f"[OK]   {name}: {msg}")
            elif status == "skip":
                skip += 1
                print(f"[SKIP] {name}: {msg}")
            else:
                err += 1
                print(f"[ERR]  {name}: {msg}")

    print(f"\nDone. OK={ok}, SKIP={skip}, ERR={err}")


if __name__ == "__main__":
    PATH = "/home/rdcunha/vista_project/vista_bench"
    CONFIG_PATH = "/home/rdcunha/vista_project/vista_eval_vlm/configs/all_tasks.yaml"
    VALID_TASKS_JSON_PATH = "/home/rdcunha/vista_project/vista_bench/tasks/valid_tasks.json"
    subsample_csvs_parallel(
        PATH,
        target_n=50,
        workers=1,
        config_path=CONFIG_PATH,
        valid_tasks_json_path=VALID_TASKS_JSON_PATH,
        overwrite=True,
    )
