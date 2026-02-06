# """
# Add a 'report' column to _subsampled_no_img_report CSVs by:
# - Loading tasks from config (all_tasks.yaml)
# - For each task, reading the _subsampled_no_img_report CSV
# - When '_accession_number' is present: query BigQuery 'note' for note_id and note_text
#   (person_id + _accession_number), then format note_text the same way text_value is
#   formatted in get_llm_event_string (NOTE: clean_text) and add to the 'report' column.
# """

from pathlib import Path

import pandas as pd
from google.cloud import bigquery

from data_tools.utils.query_utils import get_notes_from_bq_batch
from data_tools.utils.report_utils import build_report_from_notes
from data_tools.utils.config_utils import load_tasks_and_base_dir, load_task_source_csv


def process_csv_add_report(
    csv_path: Path,
    bq_client: bigquery.Client,
    person_id_col: str = "person_id",
    accession_col: str = "_accession_number",
    overwrite_report: bool = False,
    max_text_len: int | None = None,
    note_dataset: str | None = None,
    note_table: str | None = None,
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

    note_dataset = note_dataset or None
    note_table = note_table or None

    # Unique (person_id, accession) pairs with non-null accession
    mask_valid = df[accession_col].notna() & (df[accession_col].astype(str).str.strip() != "")
    pairs = list(
        df.loc[mask_valid, [person_id_col, accession_col]]
        .drop_duplicates()
        .itertuples(index=False, name=None)
    )
    if not pairs:
        df["report"] = ""
        return df

    # Single batched query for all pairs
    batch_kwargs = {}
    if note_dataset is not None:
        batch_kwargs["dataset"] = note_dataset
    if note_table is not None:
        batch_kwargs["table"] = note_table
    try:
        notes_batch = get_notes_from_bq_batch(bq_client, pairs, **batch_kwargs)
    except Exception as e:
        print(f"  [WARN] Batch notes query failed: {e}")
        df["report"] = ""
        return df

    # Map (person_id, _accession_number) -> notes DataFrame (note_id, note_text, optional note_datetime for build_report_from_notes)
    notes_by_key = {}
    if not notes_batch.empty and "person_id" in notes_batch.columns and "_accession_number" in notes_batch.columns:
        cols = ["note_id", "note_text"]
        if "note_datetime" in notes_batch.columns:
            cols.append("note_datetime")
        for (pid, acc), grp in notes_batch.groupby(["person_id", "_accession_number"]):
            key = (int(pid), str(acc).strip())
            notes_by_key[key] = grp[[c for c in cols if c in grp.columns]].copy()

    def report_for_row(person_id, accession):
        if pd.isna(person_id) or pd.isna(accession) or str(accession).strip() == "":
            return ""
        try:
            key = (int(person_id), str(accession).strip())
        except (ValueError, TypeError):
            return ""
        notes_df = notes_by_key.get(key, pd.DataFrame(columns=["note_id", "note_text"]))
        if notes_df.empty:
            return ""
        if "note_text" in notes_df.columns:
            has_non_reportable = notes_df["note_text"].astype(str).str.contains(
                "non-reportable", case=False, na=False
            ).any()
            if has_non_reportable:
                return pd.NA
        return build_report_from_notes(notes_df, max_text_len=max_text_len)

    df["report"] = [
        report_for_row(row[person_id_col], row.get(accession_col))
        for _, row in df.iterrows()
    ]
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

