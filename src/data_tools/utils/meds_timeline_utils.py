import re

import pandas as pd


def count_unique_event_dates(timeline_text) -> int:
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
    pattern = r'\[(\d{4}-\d{2}-\d{2})\s+\d{2}:\d{2}\]'

    dates = re.findall(pattern, text_str)
    return len(set(dates)) if dates else 0


def get_first_4_rows(text) -> str:
    """
    Return the first 4 lines of the patient timeline.
    Used so every experiment (including no_timeline) includes the first 4 rows.
    """
    # if pd.isna(text) or text is None:
    #     return ""
    # text_str = str(text).strip()
    # if not text_str:
    #     return ""
    # lines = text_str.split("\n")
    # return "\n".join(lines[:4])
    pass


def truncate_timeline(text, truncation_config=None) -> str:
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

    lines = text_str.split('\n')
    if len(lines) <= 4:
        return text_str

    first_4_rows = ''
    remaining_text = text_str

    if truncation_config is None:
        return first_4_rows + '\n' + remaining_text

    mode = truncation_config.get('mode', 'max_chars')

    if mode == 'max_chars':
        max_chars = truncation_config.get('max_chars', 5000)
        first_4_chars = len(first_4_rows)
        remaining_max_chars = max_chars - first_4_chars - len('\n')

        if remaining_max_chars <= 0:
            return first_4_rows

        if len(remaining_text) > remaining_max_chars:
            truncated_remaining = remaining_text[:remaining_max_chars] + "... [TRUNCATED]"
        else:
            truncated_remaining = remaining_text

        return first_4_rows + '\n' + truncated_remaining

    elif mode == 'last_k_events':
        initial_k = truncation_config.get('k', 10)
        safety_max_chars = truncation_config.get('max_chars', 180000)

        pattern = r'\[(\d{4}-\d{2}-\d{2})\s+\d{2}:\d{2}\]\s*\|'
        matches = list(re.finditer(pattern, text_str))

        if len(matches) == 0:
            if len(text_str) > safety_max_chars:
                return text_str[:safety_max_chars] + "... [TRUNCATED]"
            return text_str

        date_match_pairs = [(m.group(1), m) for m in matches]

        seen_dates = set()
        unique_dates_list = []
        for date, _ in date_match_pairs:
            if date not in seen_dates:
                seen_dates.add(date)
                unique_dates_list.append(date)

        if len(unique_dates_list) <= initial_k:
            if len(text_str) > safety_max_chars:
                return text_str[:safety_max_chars] + "... [TRUNCATED]"
            return text_str

        current_k = initial_k
        while current_k > 0:
            first_date_in_last_k = unique_dates_list[-current_k]
            start_pos = None

            for date, match in date_match_pairs:
                if date == first_date_in_last_k:
                    start_pos = match.start()
                    break

            if start_pos is None:
                current_k -= 1
                continue

            rest_start = max(start_pos, len(first_4_rows) + 1)
            truncated = text_str[rest_start:]

            if len(truncated) <= safety_max_chars:
                return first_4_rows + "\n" + truncated

            current_k -= 1

        # Fallback: even k=1 was too long
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
                safety_matches = list(re.finditer(pattern, truncated[:safety_max_chars]))
                if safety_matches:
                    last_safe_match = safety_matches[-1]
                    event_end = truncated.find('\n', last_safe_match.end())
                    if event_end == -1:
                        event_end = last_safe_match.end()
                    return first_4_rows + "\n" + truncated[:event_end] + "\n... [TRUNCATED - even single date too long]"
                else:
                    return first_4_rows + "\n" + truncated[:safety_max_chars] + "... [TRUNCATED]"

        rest_max = safety_max_chars - len(first_4_rows) - 1
        if len(remaining_text) > rest_max:
            return first_4_rows + "\n" + remaining_text[:rest_max] + "... [TRUNCATED]"
        return first_4_rows + "\n" + remaining_text

    else:
        return first_4_rows + '\n' + remaining_text


def get_llm_event_string(
    df: pd.DataFrame, 
    include_text: bool = True, 
    max_text_len: int | None = None,
    exclude_report = True
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
            if exclude_report:
                code_val = str(row['code'])
                if 'STANFORD' in code_val:
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