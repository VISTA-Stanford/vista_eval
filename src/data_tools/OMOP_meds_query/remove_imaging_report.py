"""
Build CSVs with patient timeline that skips imaging codes (no radiology report in timeline).
Reads subsampled CSVs (from config tasks), uses get_described_events_window with embed_time as end_time
and 6 months prior as start_time, get_llm_event_string skipping imaging codes; writes 'patient_string'
into new CSVs ending in _subsampled_no_img_report with all other columns unchanged.
"""

from pathlib import Path

import pandas as pd
import yaml
import json

import meds_reader
from meds_tools import patient_timeline
from meds2text.ontology import OntologyDescriptionLookupTable
from data_tools.utils.meds_timeline_utils import get_llm_event_string
from data_tools.utils.config_utils import load_tasks_and_base_dir, load_task_source_csv

def process_subsampled_csv(
    csv_path: Path,
    database: meds_reader.SubjectDatabase,
    lookup_table: OntologyDescriptionLookupTable,
    embed_time_col: str = "embed_time",
    person_id_col: str = "person_id",
    months_before: int = 6,
) -> pd.DataFrame:
    """
    Read subsampled CSV and replace 'patient_string' with timeline built from
    get_described_events_window (end_time=embed_time, start_time=6 months prior, subject_id=person_id)
    and get_llm_event_string_no_imaging. All other columns unchanged.
    """
    df = pd.read_csv(csv_path)

    if embed_time_col not in df.columns:
        raise ValueError(f"CSV missing column '{embed_time_col}': {csv_path}")
    if person_id_col not in df.columns:
        raise ValueError(f"CSV missing column '{person_id_col}': {csv_path}")

    # Ensure we have a patient_string column (create if missing so output has same structure)
    if "patient_string" not in df.columns:
        df["patient_string"] = ""

    embed_times = pd.to_datetime(df[embed_time_col], errors="coerce")
    start_times = embed_times - pd.DateOffset(months=months_before)

    new_strings = []
    for i, row in df.iterrows():
        subject_id = row[person_id_col]
        end_time = embed_times.iloc[i]
        start_time = start_times.iloc[i]

        if pd.isna(end_time) or pd.isna(subject_id):
            new_strings.append(row.get("patient_string", "") if pd.notna(row.get("patient_string")) else "")
            continue

        try:
            window_df = patient_timeline.get_described_events_window(
                database=database,
                lookup_table=lookup_table,
                subject_id=subject_id,
                end_time=end_time,
                start_time=start_time,
            )
            # Event DataFrame has MultiIndex (subject_id, time); expose 'time' as column for get_llm_event_string
            if not window_df.empty and "time" not in window_df.columns and hasattr(window_df.index, "names") and window_df.index.names:
                window_df = window_df.reset_index()
            patient_string = get_llm_event_string(window_df, include_text=True, exclude_report=True)
        except Exception as e:
            print(f"  [WARN] Row {i} (person_id={subject_id}): {e}")
            patient_string = row.get("patient_string", "") if pd.notna(row.get("patient_string")) else ""

        new_strings.append(patient_string)

    df["patient_string"] = new_strings
    return df


def run(
    config_path: str,
    valid_tasks_json_path: str,
    meds_db_path: str,
    ontology_path: str,
    overwrite: bool = False,
):
    """
    For each subsampled task CSV from config, build no-imaging-report timeline and save
    to same location with suffix _subsampled_no_img_report (e.g. .../task_subsampled_no_img_report.csv).
    """
    tasks, base_dir = load_tasks_and_base_dir(config_path)
    task_to_source = load_task_source_csv(valid_tasks_json_path)
    base_path = Path(base_dir)

    if not base_path.exists():
        print(f"Error: base_dir not found: {base_path}")
        return

    lookup = OntologyDescriptionLookupTable()
    lookup.load(ontology_path)
    database = meds_reader.SubjectDatabase(meds_db_path)

    for task_name in tasks:
        source_csv = task_to_source.get(task_name)
        if not source_csv:
            print(f"[SKIP] No task_source_csv for task '{task_name}'")
            continue

        csv_path = base_path / source_csv / f"{task_name}_subsampled.csv"
        if not csv_path.exists():
            print(f"[SKIP] File not found: {csv_path}")
            continue

        out_path = csv_path.with_name(f"{csv_path.stem}_no_img_report.csv")
        if out_path.exists() and not overwrite:
            print(f"[SKIP] Output exists (use overwrite=True): {out_path.name}")
            continue

        print(f"Processing: {csv_path.name} -> {out_path.name}")
        try:
            df_out = process_subsampled_csv(
                csv_path,
                database=database,
                lookup_table=lookup,
                embed_time_col="embed_time",
                person_id_col="person_id",
                months_before=6,
            )
            df_out.to_csv(out_path, index=False)
            print(f"  Wrote {len(df_out)} rows to {out_path}")
        except Exception as e:
            print(f"[ERROR] {csv_path.name}: {e}")
            raise

    print("Done.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build _subsampled_no_img_report CSVs with timeline skipping imaging codes.")
    parser.add_argument("--config", default="/home/rdcunha/vista_project/vista_eval_vlm/configs/all_tasks.yaml", help="Path to all_tasks.yaml")
    parser.add_argument("--valid-tasks", default="/home/rdcunha/vista_project/vista_bench/tasks/valid_tasks.json", help="Path to valid_tasks.json")
    parser.add_argument("--meds-db", default="/home/rdcunha/vista_project/vista_bench/thoracic_cohort_meds/vista_thoracic_cohort_v0_db", help="Path to meds reader DB")
    parser.add_argument("--ontology", default="/home/rdcunha/vista_project/vista_bench/thoracic_cohort_meds/athena_omop_ontologies", help="Path to ontology")
    parser.add_argument("--overwrite", default=True, help="Overwrite existing _subsampled_no_img_report files")
    args = parser.parse_args()

    run(
        config_path=args.config,
        valid_tasks_json_path=args.valid_tasks,
        meds_db_path=args.meds_db,
        ontology_path=args.ontology,
        overwrite=args.overwrite,
    )
