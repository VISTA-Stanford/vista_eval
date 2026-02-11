"""Retrieval package for iterative VLM-driven patient timeline retrieval."""

from retrieval.format_events import format_retrieved_events
from retrieval.iterative_retrieval import (
    run_iterative_retrieval,
    run_iterative_retrieval_batch,
    IterativeRetrievalResult,
)


def __getattr__(name):
    """Lazy import LocalPatientRetriever to avoid loading meds_mcp until needed."""
    if name == "LocalPatientRetriever":
        from retrieval.local_retriever import LocalPatientRetriever
        return LocalPatientRetriever
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "LocalPatientRetriever",
    "run_iterative_retrieval",
    "run_iterative_retrieval_batch",
    "format_retrieved_events",
    "IterativeRetrievalResult",
]
