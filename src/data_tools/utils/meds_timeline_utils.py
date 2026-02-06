import pandas as pd

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
                if 'STANFORD_NOTE/imaging' in code_val:
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