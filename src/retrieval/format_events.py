"""
Format meds_mcp search results into patient_string-style timeline format.
"""

import datetime
import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def _parse_timestamp(ts: Any) -> Optional[datetime.datetime]:
    """Parse timestamp from various formats. Returns None on failure."""
    if ts is None:
        return None
    if isinstance(ts, datetime.datetime):
        return ts
    if isinstance(ts, str):
        try:
            return datetime.datetime.fromisoformat(ts.replace("Z", "+00:00"))
        except ValueError:
            pass
        try:
            return datetime.datetime.strptime(ts, "%Y-%m-%d %H:%M")
        except ValueError:
            pass
    return None


def format_retrieved_events(
    results: List[Dict[str, Any]],
    exclude_report: bool = False,
) -> str:
    """
    Convert meds_mcp search results to patient_string format.

    Format: [YYYY-MM-DD HH:MM] | event_type/code/name | VALUE: ... | NOTE: ...

    Args:
        results: List of dicts from search_patient_events with keys:
            id, content, metadata, timestamp, event_type, code, name, person_id, score
        exclude_report: If True, exclude events with STANFORD or imaging in code/name.

    Returns:
        Formatted timeline string. Returns "No clinical events found for this period."
        when results is empty.
    """
    if not results:
        return "No clinical events found for this period."

    # Deduplicate by id
    seen: set = set()
    unique_results: List[Dict[str, Any]] = []
    for r in results:
        rid = r.get("id")
        if rid is not None and rid not in seen:
            seen.add(rid)
            unique_results.append(r)

    lines: List[str] = []
    for r in unique_results:
        metadata = r.get("metadata") or {}
        code = r.get("code") or metadata.get("code")
        name = r.get("name") or metadata.get("name")
        event_type = r.get("event_type") or metadata.get("event_type", "")
        value = metadata.get("value", "")
        ts = r.get("timestamp") or metadata.get("timestamp")

        if exclude_report:
            code_str = str(code) if code is not None else ""
            name_str = str(name) if name is not None else ""
            combined = f"{code_str} {name_str}".upper()
            if "STANFORD" in combined or "IMAGING" in combined:
                continue

        event_parts: List[str] = []

        # Timestamp
        dt = _parse_timestamp(ts)
        if dt is not None:
            event_parts.append(f"[{dt.strftime('%Y-%m-%d %H:%M')}]")
        else:
            event_parts.append("[unknown]")

        # Code/name/event_type
        desc_parts = []
        if code:
            desc_parts.append(str(code))
        if name and str(name).strip():
            desc_parts.append(f"({name})")
        if not desc_parts and event_type:
            desc_parts.append(str(event_type))
        if desc_parts:
            event_parts.append(" ".join(desc_parts))

        # Value content
        if value is not None and str(value).strip():
            val_str = str(value).replace("\n", " ").strip()
            if val_str:
                event_parts.append(f"VALUE: {val_str}")

        if len(event_parts) >= 2:
            lines.append((" | ".join(event_parts), dt or datetime.datetime.min))

    # Sort by timestamp
    lines.sort(key=lambda x: x[1])
    return "\n".join(l[0] for l in lines) if lines else "No clinical events found for this period."
