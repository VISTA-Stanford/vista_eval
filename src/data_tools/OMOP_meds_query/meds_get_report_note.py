"""
Build a report column from _subsampled_no_img_report CSVs by:
- Loading tasks from config (all_tasks.yaml)
- For each task, reading the _subsampled_no_img_report CSV
- When '_accession_number' is present: query BigQuery 'note' for note_id only (person_id + _accession_number),
  get patient_df via get_described_events_window (same window as remove_imaging_report / test_meds_tools),
  filter rows by note_id, and collect text_value formatted like get_llm_event_string (NOTE: clean_text).
- Save only the report column to report.csv (one file per task, in the same directory as the input CSV).
"""

from pathlib import Path

import pandas as pd
from google.cloud import bigquery

import meds_reader
from meds_tools import patient_timeline
from meds2text.ontology import OntologyDescriptionLookupTable

from data_tools.utils.query_utils import get_note_ids_from_bq
from data_tools.utils.report_utils import build_report_from_patient_df_by_note_id
from data_tools.utils.config_utils import load_tasks_and_base_dir, load_task_source_csv


def process_csv_to_reports(
    csv_path: Path,
    database: meds_reader.SubjectDatabase,
    lookup_table: OntologyDescriptionLookupTable,
    bq_client: bigquery.Client,
    embed_time_col: str = "embed_time",
    person_id_col: str = "person_id",
    accession_col: str = "_accession_number",
    months_before: int = 6,
    max_text_len: int | None = None,
    note_dataset: str | None = None,
    note_table: str | None = None,
) -> pd.DataFrame:
    """
    Read _subsampled_no_img_report CSV. For each row with person_id and accession:
    query BQ for note_id, get patient_df (get_described_events_window), filter by note_id,
    build report from text_value. Return a DataFrame with a single column 'report' (one row per CSV row).
    """
    df = pd.read_csv(csv_path)

    if (
        accession_col not in df.columns
        or person_id_col not in df.columns
        or embed_time_col not in df.columns
    ):
        return pd.DataFrame({"report": [""] * len(df)})

    embed_times = pd.to_datetime(df[embed_time_col], errors="coerce")
    start_times = embed_times - pd.DateOffset(months=months_before)

    bq_kwargs = {}
    if note_dataset is not None:
        bq_kwargs["dataset"] = note_dataset
    if note_table is not None:
        bq_kwargs["table"] = note_table

    reports = []
    for i, row in df.iterrows():
        person_id = row[person_id_col]
        accession = row.get(accession_col)
        end_time = embed_times.iloc[i]
        start_time = start_times.iloc[i]

        report_text = ""
        if (
            pd.notna(person_id)
            and pd.notna(accession)
            and str(accession).strip()
            and pd.notna(end_time)
        ):
            try:
                note_ids = get_note_ids_from_bq(
                    bq_client, person_id, str(accession), **bq_kwargs
                )
                if note_ids:
                    patient_df = patient_timeline.get_described_events_window(
                        database=database,
                        lookup_table=lookup_table,
                        subject_id=person_id,
                        end_time=end_time,
                        start_time=start_time,
                    )
                    if (
                        not patient_df.empty
                        and "time" not in patient_df.columns
                        and hasattr(patient_df.index, "names")
                        and patient_df.index.names
                    ):
                        patient_df = patient_df.reset_index()
                    report_text = build_report_from_patient_df_by_note_id(
                        patient_df, note_ids, max_text_len=max_text_len
                    )
            except Exception as e:
                print(
                    f"  [WARN] Row {i} (person_id={person_id}, accession={accession}): {e}"
                )

        reports.append(report_text)

    return pd.DataFrame({"report": reports})


def run(
    config_path: str,
    valid_tasks_json_path: str,
    meds_db_path: str,
    ontology_path: str,
    bq_project_id: str = "som-nero-plevriti-deidbdf",
    max_text_len: int | None = None,
    report_filename: str = "report.csv",
):
    """
    For each task in config, load the _subsampled_no_img_report CSV, build report column from
    BQ note_id + patient_df text_value (MEDS), and save only the report column to report.csv
    in the same directory as the input CSV.
    """
    tasks, base_dir = load_tasks_and_base_dir(config_path)
    task_to_source = load_task_source_csv(valid_tasks_json_path)
    base_path = Path(base_dir)

    if not base_path.exists():
        print(f"Error: base_dir not found: {base_path}")
        return

    bq_client = bigquery.Client(project=bq_project_id)
    lookup = OntologyDescriptionLookupTable()
    lookup.load(ontology_path)
    database = meds_reader.SubjectDatabase(meds_db_path)

    for task_name in tasks:
        source_csv = task_to_source.get(task_name)
        if not source_csv:
            print(f"[SKIP] No task_source_csv for task '{task_name}'")
            continue

        csv_path = base_path / source_csv / f"{task_name}_subsampled_no_img_report.csv"
        if not csv_path.exists():
            print(f"[SKIP] File not found: {csv_path}")
            continue

        out_path = base_path / source_csv / report_filename

        print(f"Processing: {csv_path.name} -> {out_path.name}")
        try:
            report_df = process_csv_to_reports(
                csv_path,
                database=database,
                lookup_table=lookup,
                bq_client=bq_client,
                embed_time_col="embed_time",
                person_id_col="person_id",
                accession_col="_accession_number",
                months_before=6,
                max_text_len=max_text_len,
            )
            report_df.to_csv(out_path, index=False)
            n_nan = report_df["report"].isna().sum()
            print(f"  Wrote {len(report_df)} rows (report column only) to {out_path}")
            print(f"  'report' entries that are NaN: {n_nan}")
        except Exception as e:
            print(f"[ERROR] {csv_path.name}: {e}")
            raise

    print("Done.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Build report column from BQ note_id + MEDS patient_df text_value; save to report.csv.",
    )
    parser.add_argument(
        "--config",
        default="/home/rdcunha/vista_project/vista_eval_vlm/configs/all_tasks.yaml",
        help="Path to all_tasks.yaml",
    )
    parser.add_argument(
        "--valid-tasks",
        default="/home/rdcunha/vista_project/vista_bench/tasks/valid_tasks.json",
        help="Path to valid_tasks.json",
    )
    parser.add_argument(
        "--meds-db",
        default="/home/rdcunha/vista_project/vista_bench/thoracic_cohort_meds/vista_thoracic_cohort_v0_db",
        help="Path to meds reader DB",
    )
    parser.add_argument(
        "--ontology",
        default="/home/rdcunha/vista_project/vista_bench/thoracic_cohort_meds/athena_omop_ontologies",
        help="Path to ontology",
    )
    parser.add_argument(
        "--bq-project",
        default="som-nero-plevriti-deidbdf",
        help="BigQuery project ID",
    )
    parser.add_argument(
        "--max-text-len",
        type=int,
        default=None,
        help="If set, truncate each text_value to this many characters.",
    )
    parser.add_argument(
        "--report-filename",
        default="report.csv",
        help="Output filename for report column (default: report.csv)",
    )
    args = parser.parse_args()

    run(
        config_path=args.config,
        valid_tasks_json_path=args.valid_tasks,
        meds_db_path=args.meds_db,
        ontology_path=args.ontology,
        bq_project_id=args.bq_project,
        max_text_len=args.max_text_len,
        report_filename=args.report_filename,
    )
