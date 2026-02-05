"""
Add a 'report' column to _subsampled_no_img_report CSVs by:
- Loading tasks from config (all_tasks.yaml)
- For each task, reading the _subsampled_no_img_report CSV
- When '_accession_number' is present: query BigQuery 'note' for note_id and note_text
  (person_id + _accession_number), then format note_text the same way text_value is
  formatted in get_llm_event_string (NOTE: clean_text) and add to the 'report' column.
"""

from pathlib import Path

import pandas as pd
from google.cloud import bigquery

try:
    from .remove_imaging_report import load_tasks_and_base_dir, load_task_source_csv
except ImportError:
    from remove_imaging_report import load_tasks_and_base_dir, load_task_source_csv


# BigQuery dataset and table for note
BQ_DATASET = "oncology_omop54_confidential_irb76049_nov2025"
BQ_NOTE_TABLE = "note"


def get_notes_from_bq(
    bq_client: bigquery.Client,
    person_id: int,
    accession_number: str,
    dataset: str = BQ_DATASET,
    table: str = BQ_NOTE_TABLE,
) -> pd.DataFrame:
    """
    Query BigQuery 'note' for note_id and note_text matching person_id and _accession_number.
    Returns a DataFrame with columns note_id, note_text (may be multiple rows).
    """
    if pd.isna(person_id) or pd.isna(accession_number) or accession_number == "":
        return pd.DataFrame(columns=["note_id", "note_text"])
    query = f"""
    SELECT note_id, note_text
    FROM `{dataset}.{table}`
    WHERE person_id = @person_id AND _accession_number = @accession_number
    """
    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("person_id", "INT64", int(person_id)),
            bigquery.ScalarQueryParameter("accession_number", "STRING", str(accession_number).strip()),
        ]
    )
    return bq_client.query(query, job_config=job_config).to_dataframe()


def format_note_text_like_text_value(
    note_text: str,
    max_text_len: int | None = None,
) -> str:
    """
    Format note_text the same way text_value is formatted in get_llm_event_string
    (test_meds_tools): replace newlines with space, strip, optional truncation, NOTE: prefix.
    Returns a single line (or empty string if no content).
    """
    if pd.isna(note_text):
        return ""
    clean_text = str(note_text).replace("\n", " ").strip()
    if not clean_text:
        return ""
    if max_text_len and len(clean_text) > max_text_len:
        clean_text = clean_text[:max_text_len] + "..."
    return f"NOTE: {clean_text}"


def build_report_from_notes(
    notes_df: pd.DataFrame,
    max_text_len: int | None = None,
) -> str:
    """
    Build report string from note_text column in notes_df, formatting each like text_value.
    Returns newline-joined lines (same format as get_llm_event_string for text_value).
    """
    if notes_df.empty or "note_text" not in notes_df.columns:
        return ""
    lines = []
    for _, row in notes_df.iterrows():
        line = format_note_text_like_text_value(row["note_text"], max_text_len=max_text_len)
        if line:
            lines.append(line)
    return "\n".join(lines)


def process_csv_add_report(
    csv_path: Path,
    bq_client: bigquery.Client,
    person_id_col: str = "person_id",
    accession_col: str = "_accession_number",
    overwrite_report: bool = False,
    max_text_len: int | None = None,
) -> pd.DataFrame:
    """
    Read _subsampled_no_img_report CSV. If 'report' already exists and not overwrite_report, return as-is.
    If accession_col is present, for each row with person_id and accession: query BQ 'note' for
    note_id and note_text, then format note_text like text_value and set as 'report'.
    Add column 'report' and return the dataframe.
    """
    df = pd.read_csv(csv_path)

    if "report" in df.columns and not overwrite_report:
        return df

    if accession_col not in df.columns:
        df["report"] = ""
        return df

    if person_id_col not in df.columns:
        df["report"] = ""
        return df

    reports = []
    for i, row in df.iterrows():
        person_id = row[person_id_col]
        accession = row.get(accession_col)

        report_text = ""
        if pd.notna(person_id) and pd.notna(accession) and str(accession).strip():
            try:
                notes_df = get_notes_from_bq(bq_client, person_id, str(accession))
                report_text = build_report_from_notes(notes_df, max_text_len=max_text_len)
            except Exception as e:
                print(f"  [WARN] Row {i} (person_id={person_id}, accession={accession}): {e}")

        reports.append(report_text)

    df["report"] = reports
    return df


def run(
    config_path: str,
    valid_tasks_json_path: str,
    bq_project_id: str = "som-nero-plevriti-deidbdf",
    overwrite: bool = False,
    max_text_len: int | None = None,
):
    """
    For each task in config, load the _subsampled_no_img_report CSV, add 'report' column
    (using _accession_number + person_id -> BQ note note_id, note_text -> format note_text like text_value),
    and save back to the same file.
    """
    tasks, base_dir = load_tasks_and_base_dir(config_path)
    task_to_source = load_task_source_csv(valid_tasks_json_path)
    base_path = Path(base_dir)

    if not base_path.exists():
        print(f"Error: base_dir not found: {base_path}")
        return

    bq_client = bigquery.Client(project=bq_project_id)

    for task_name in tasks:
        source_csv = task_to_source.get(task_name)
        if not source_csv:
            print(f"[SKIP] No task_source_csv for task '{task_name}'")
            continue

        csv_path = base_path / source_csv / f"{task_name}_subsampled_no_img_report.csv"
        if not csv_path.exists():
            print(f"[SKIP] File not found: {csv_path}")
            continue

        print(f"Processing: {csv_path.name}")
        try:
            df_out = process_csv_add_report(
                csv_path,
                bq_client=bq_client,
                person_id_col="person_id",
                accession_col="_accession_number",
                overwrite_report=overwrite,
                max_text_len=max_text_len,
            )
            df_out.to_csv(csv_path, index=False)
            n_nan = df_out["report"].isna().sum()
            print(f"  Wrote {len(df_out)} rows with 'report' column to {csv_path}")
            print(f"  'report' entries that are NaN: {n_nan}")
        except Exception as e:
            print(f"[ERROR] {csv_path.name}: {e}")
            raise

    print("Done.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Add 'report' column to _subsampled_no_img_report CSVs from BQ note (note_id, note_text); format note_text like text_value.",
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
        "--bq-project",
        default="som-nero-plevriti-deidbdf",
        help="BigQuery project ID",
    )
    parser.add_argument(
        "--overwrite",
        default="True",
        help="Overwrite existing 'report' column if present",
    )
    parser.add_argument(
        "--max-text-len",
        type=int,
        default=None,
        help="If set, truncate each note_text to this many characters (same as get_llm_event_string).",
    )
    args = parser.parse_args()

    run(
        config_path=args.config,
        valid_tasks_json_path=args.valid_tasks,
        bq_project_id=args.bq_project,
        overwrite=args.overwrite,
        max_text_len=args.max_text_len,
    )
