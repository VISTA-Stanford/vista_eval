import meds_reader
from meds_tools import patient_timeline
from meds2text.ontology import OntologyDescriptionLookupTable
import pandas as pd

def get_llm_event_string(
    df: pd.DataFrame, 
    include_text: bool = True, 
    max_text_len: int | None = None
) -> str:
    """
    Converts a DataFrame into an LLM-optimized string.
    
    Args:
        df: The patient events DataFrame.
        include_text: If False, ignores the 'text_value' field entirely.
        max_text_len: If set, truncates the 'text_value' to this many characters.
    """
    if df.empty:
        return "No clinical events found for this period."

    temp_df = df
    lines = []

    for _, row in temp_df.iterrows():
        event_parts = []
        
        # 1. Extract Time
        if 'time' in row and pd.notnull(row['time']):
            time_str = row['time'].strftime('%Y-%m-%d %H:%M')
            event_parts.append(f"[{time_str}]")
            
        # 2. Extract Code and Description
        if 'code' in row and pd.notnull(row['code']):
            if 'code' == 'STANFORD_NOTE/imaging' or 'code' == 'STANFORD_NOTE/imaging-non-reportable':
                continue
            desc = f" ({row['description']})" if pd.notnull(row.get('description')) and row['description'] != "" else ""
            event_parts.append(f"{row['code']}{desc}")

        # 3. Extract Numeric Value + Unit
        if 'numeric_value' in row and pd.notnull(row['numeric_value']):
            unit_str = f" {row['unit']}" if pd.notnull(row.get('unit')) else ""
            event_parts.append(f"VALUE: {row['numeric_value']}{unit_str}")

        # 4. Extract Text Notes (Conditional)
        if include_text and 'text_value' in row and pd.notnull(row['text_value']):
            clean_text = str(row['text_value']).replace('\n', ' ').strip()
            
            if clean_text:
                # Optional Truncation logic
                if max_text_len and len(clean_text) > max_text_len:
                    clean_text = clean_text[:max_text_len] + "..."
                
                event_parts.append(f"NOTE: {clean_text}")

        # Combine into a single pipe-delimited line
        if event_parts:
            lines.append(" | ".join(event_parts))

    return "\n".join(lines)


def _populated(series: pd.Series) -> pd.Series:
    """True where value is non-null and (for strings) non-empty after strip."""
    if series.dtype == object or series.dtype.name == "string":
        return series.notna() & (series.astype(str).str.strip() != "")
    return series.notna()


def rows_by_image_note_status(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Among rows with a populated 'image_occurrence_id', split into:
    - rows_with_note: rows that also have a populated 'note_id'
    - rows_without_note: rows that do not have a populated 'note_id'

    'Populated' means non-null and (for strings) non-empty after strip.

    Returns:
        (rows_with_note, rows_without_note)
    """
    if df.empty:
        return pd.DataFrame(), pd.DataFrame()

    required = ["image_occurrence_id", "note_id"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"DataFrame missing columns: {missing}")

    has_io = _populated(df["image_occurrence_id"])
    has_note = _populated(df["note_id"])

    with_image = df.loc[has_io]
    has_note_in_subset = has_note.loc[has_io]
    rows_with_note = with_image.loc[has_note_in_subset].copy()
    rows_without_note = with_image.loc[~has_note_in_subset].copy()

    return rows_with_note, rows_without_note


def rows_missing_series_study_uid(
    df: pd.DataFrame,
    uid_columns: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Among rows that have both 'note_id' and 'image_occurrence_id' populated,
    split into rows that have both 'image_series_uid' and 'image_study_uid'
    vs rows missing either UID.

    Returns:
        (rows_with_both_uids, rows_missing_uid)
    """
    if uid_columns is None:
        uid_columns = ["image_series_uid", "image_study_uid"]
    if df.empty:
        return pd.DataFrame(), pd.DataFrame()

    required = ["image_occurrence_id", "note_id"] + list(uid_columns)
    missing_cols = [c for c in required if c not in df.columns]
    if missing_cols:
        raise ValueError(f"DataFrame missing columns: {missing_cols}")

    has_io = _populated(df["image_occurrence_id"])
    has_note = _populated(df["note_id"])
    with_note_and_io = df.loc[has_io & has_note]

    has_series = _populated(with_note_and_io["image_series_uid"])
    has_study = _populated(with_note_and_io["image_study_uid"])
    has_both_uids = has_series & has_study

    rows_with_both_uids = with_note_and_io.loc[has_both_uids].copy()
    rows_missing_uid = with_note_and_io.loc[~has_both_uids].copy()

    return rows_with_both_uids, rows_missing_uid


SUBJECT_ID = 135908719
START_DATE = '2025-01-01'
END_DATE = '2025-12-19'

# database params
PATH_TO_MEDS_READER_DB = "/home/rdcunha/vista_project/vista_bench/thoracic_cohort_meds/vista_thoracic_cohort_v0_db"
ONTOLOGY_PATH = "/home/rdcunha/vista_project/vista_bench/thoracic_cohort_meds/athena_omop_ontologies"

# Setup
lookup = OntologyDescriptionLookupTable()
lookup.load(ONTOLOGY_PATH)
database = meds_reader.SubjectDatabase(PATH_TO_MEDS_READER_DB)

###################################################
# Get patient timeline by time window (dataframe) #
###################################################
df_patient = patient_timeline.get_described_events_window(
    database=database,
    lookup_table=lookup,
    subject_id=SUBJECT_ID,
    start_time=START_DATE,
    end_time=END_DATE
)
print(df_patient.columns)
print(len(df_patient))
print(df_patient['text_value'].nunique())
# print(df_patient.head())

# print(f"Total events found: {len(df_patient)}")
print("--------------------------")

# Rows with image_occurrence_id: both populated vs missing note_id
# required_for_note_check = ["image_occurrence_id", "note_id"]
# required_for_uid_check = required_for_note_check + ["image_series_uid", "image_study_uid"]

# if all(c in df_patient.columns for c in required_for_note_check):
#     rows_with_note, rows_without_note = rows_by_image_note_status(df_patient)
#     print("Rows with image_occurrence_id AND populated note_id:")
#     print(rows_with_note)
#     print("--------------------------")
#     print("Rows with image_occurrence_id but NO populated note_id:")
#     print(rows_without_note)

#     # Among rows with note_id + image_occurrence_id, check image_series_uid and image_study_uid
#     if all(c in df_patient.columns for c in required_for_uid_check):
#         rows_with_both_uids, rows_missing_uid = rows_missing_series_study_uid(df_patient)
#         print("--------------------------")
#         print("Rows with note_id+image_occurrence_id AND both image_series_uid & image_study_uid:")
#         print(rows_with_both_uids)
#         print("--------------------------")
#         print("Rows with note_id+image_occurrence_id but MISSING image_series_uid and/or image_study_uid:")
#         print(rows_missing_uid)
#     else:
#         missing_uid_cols = [c for c in ["image_series_uid", "image_study_uid"] if c not in df_patient.columns]
#         print("--------------------------")
#         print(f"Skipping UID check: missing columns {missing_uid_cols}")
# else:
#     print("Columns 'image_occurrence_id' and/or 'note_id' not in dataframe; skipping image/note check.")

# if not df_patient.empty:
#     print(df_patient[['code', 'description']].head())

#######################################
#     Convert to string format        #
#######################################
# patient_string = get_llm_event_string(df_patient, include_text=True)
# print(patient_string[:200])