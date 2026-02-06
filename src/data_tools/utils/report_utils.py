"""
Report and note text formatting utilities.
Format note_text the same way text_value is formatted in get_llm_event_string (e.g. NOTE: prefix, clean_text).
"""
import pandas as pd


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


def _format_note_datetime(note_datetime) -> str:
    """Format note_datetime as '[YYYY-MM-DD HH:MM]' or return empty string if missing/invalid."""
    if pd.isna(note_datetime):
        return ""
    try:
        dt = pd.to_datetime(note_datetime)
        return f"[{dt.strftime('%Y-%m-%d %H:%M')}]"
    except (ValueError, TypeError, AttributeError):
        return ""


def build_report_from_notes(
    notes_df: pd.DataFrame,
    max_text_len: int | None = None,
) -> str:
    """
    Build report string from note_text (and optional note_datetime) in notes_df.
    When note_datetime is present, each line is "[YYYY-MM-DD HH:MM] | NOTE: clean_text".
    Otherwise "NOTE: clean_text" only. Returns newline-joined lines.
    """
    if notes_df.empty or "note_text" not in notes_df.columns:
        return ""
    has_datetime = "note_datetime" in notes_df.columns
    lines = []
    for _, row in notes_df.iterrows():
        note_line = format_note_text_like_text_value(row["note_text"], max_text_len=max_text_len)
        if not note_line:
            continue
        if has_datetime:
            dt_prefix = _format_note_datetime(row.get("note_datetime"))
            if dt_prefix:
                note_line = f"{dt_prefix} | {note_line}"
        lines.append(note_line)
    return "\n".join(lines)


def build_report_from_patient_df_by_note_id(
    patient_df: pd.DataFrame,
    note_ids: list,
    max_text_len: int | None = None,
) -> str:
    """
    From patient_df (e.g. output of get_described_events_window), keep only rows whose note_id
    is in note_ids. For those rows, format text_value like get_llm_event_string (NOTE: clean_text).
    Returns newline-joined string.
    """
    if patient_df.empty or not note_ids:
        return ""
    if "note_id" not in patient_df.columns:
        return ""
    mask = patient_df["note_id"].astype("Int64").isin(note_ids)
    subset = patient_df.loc[mask]
    lines = []
    for _, row in subset.iterrows():
        if "text_value" not in row or pd.isnull(row["text_value"]):
            continue
        line = format_note_text_like_text_value(row["text_value"], max_text_len=max_text_len)
        if line:
            lines.append(line)
    return "\n".join(lines)
