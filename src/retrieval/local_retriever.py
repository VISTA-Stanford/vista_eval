"""
Local in-process patient retriever using meds_mcp (no server).
"""

import datetime
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)

try:
    from meds_mcp.server.rag.simple_storage import (
        initialize_document_store,
        get_document_store,
    )
    from meds_mcp.server.tools.search import PatientTimelineRetriever, SearchFilters
    from meds_mcp.server.tools.search import SortOrder
except ImportError as e:
    raise ImportError(
        "meds_mcp is required for retrieval. Install with: pip install -e '.[retrieval]'"
    ) from e


def _parse_date(value: Union[str, datetime.datetime, None]) -> Optional[datetime.datetime]:
    """Parse date string to datetime. Supports YYYY-MM-DD, YYYY-MM-DD HH:MM, YYYY-MM-DD HH:MM:SS."""
    if value is None:
        return None
    if isinstance(value, datetime.datetime):
        return value
    s = str(value).strip()
    if not s:
        return None
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d"):
        try:
            return datetime.datetime.strptime(s, fmt)
        except ValueError:
            continue
    return None


class LocalPatientRetriever:
    """
    In-process patient timeline retriever using meds_mcp.

    corpus_dir must contain {person_id}.xml files for each patient to search.
    """

    def __init__(self, corpus_dir: str, cache_dir: str):
        """
        Initialize meds_mcp document store.

        Args:
            corpus_dir: Directory containing patient XML files ({person_id}.xml).
            cache_dir: Directory for BM25 index cache.

        Raises:
            FileNotFoundError: If corpus_dir does not exist.
        """
        corpus_path = Path(corpus_dir)
        if not corpus_path.exists():
            raise FileNotFoundError(
                f"Corpus directory not found: {corpus_dir}. "
                "Ensure XML files are downloaded and path is correct."
            )
        if not corpus_path.is_dir():
            raise NotADirectoryError(f"Corpus path is not a directory: {corpus_dir}")

        initialize_document_store(
            data_dir=str(corpus_path),
            cache_dir=cache_dir,
            load_all_patients=True,
        )
        store = get_document_store()
        if store is None:
            raise RuntimeError("Document store failed to initialize")
        self._retriever = PatientTimelineRetriever(store)

    def search(
        self,
        person_id: str,
        query: str,
        max_results: int = 20,
        start_date: Optional[Union[str, datetime.datetime]] = None,
        end_date: Optional[Union[str, datetime.datetime]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Search patient timeline events by query.

        Args:
            person_id: Patient ID (string, e.g. "136055918").
            query: Search query string (keywords).
            max_results: Maximum number of results to return.
            start_date: Optional start date (YYYY-MM-DD or YYYY-MM-DD HH:MM).
                Limit results to events on or after this date.
            end_date: Optional end date (YYYY-MM-DD or YYYY-MM-DD HH:MM).
                Limit results to events on or before this date.

        Returns:
            List of event dicts with id, content, metadata, timestamp, event_type,
            code, name, person_id, score. Returns [] if patient XML is missing
            or no results.
        """
        person_id_str = str(person_id).strip()
        if not person_id_str:
            return []

        start_dt = _parse_date(start_date)
        end_dt = _parse_date(end_date)
        if start_date is not None and start_dt is None:
            logger.warning("Invalid start_date=%r, ignoring", start_date)
        if end_date is not None and end_dt is None:
            logger.warning("Invalid end_date=%r, ignoring", end_date)

        # SearchFilters.time_range requires both start and end; use defaults when only one given
        filter_start = start_dt if start_dt is not None else (datetime.datetime.min if end_dt is not None else None)
        filter_end = end_dt if end_dt is not None else (datetime.datetime.max if start_dt is not None else None)

        filters = SearchFilters(
            max_results=max_results,
            sort_by=SortOrder.RELEVANCE,
            start=filter_start,
            end=filter_end,
        )
        try:
            return self._retriever.search(
                query=query,
                person_id=person_id_str,
                filters=filters,
            )
        except Exception as e:
            logger.warning(
                "Retrieval failed for person_id=%s: %s",
                person_id_str,
                e,
                exc_info=False,
            )
            return []
