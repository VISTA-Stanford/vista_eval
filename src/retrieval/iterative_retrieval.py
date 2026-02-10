"""
Iterative VLM-driven patient timeline retrieval.
"""

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from retrieval.format_events import format_retrieved_events

if TYPE_CHECKING:
    from retrieval.local_retriever import LocalPatientRetriever
from retrieval.keyword_prompts import (
    KEYWORD_EXTRACTION_TEMPLATE,
    KEYWORD_EXTRACTION_WITH_PREV,
    ITERATION_DECISION_TEMPLATE,
)

logger = logging.getLogger(__name__)


def _run_vlm_text_only(
    adapter: Any,
    model: Any,
    processor: Any,
    prompt: str,
    max_tokens: int = 128,
) -> str:
    """
    Run VLM inference with text-only prompt (no image).

    Args:
        adapter: Model adapter with create_template, prepare_inputs, infer.
        model: Loaded model.
        processor: Loaded processor.
        prompt: Text prompt.
        max_tokens: Max tokens to generate.

    Returns:
        Generated text, stripped.
    """
    item = {"question": prompt}
    messages = [adapter.create_template(item)]
    inputs = adapter.prepare_inputs(messages, processor, model)
    outputs = adapter.infer(model, processor, inputs, max_tokens)
    if isinstance(outputs, list) and outputs:
        return str(outputs[0]).strip()
    return ""


def _parse_keywords(raw: str) -> List[str]:
    """Parse comma-separated keywords from VLM output."""
    if not raw or not str(raw).strip():
        return []
    parts = [p.strip() for p in str(raw).split(",") if p.strip()]
    return parts[:10]


def _parse_decision(raw: str) -> str:
    """Parse STOP or CONTINUE from VLM output. Default to STOP on ambiguity."""
    s = str(raw).strip().upper()
    if "CONTINUE" in s:
        return "CONTINUE"
    return "STOP"


@dataclass
class IterativeRetrievalResult:
    """Result of iterative retrieval."""

    timeline_str: str
    iterations_log: List[Dict[str, Any]] = field(default_factory=list)
    all_keywords: List[str] = field(default_factory=list)


def run_iterative_retrieval(
    retriever: "LocalPatientRetriever",
    vlm_adapter: Any,
    vlm_model: Any,
    vlm_processor: Any,
    person_id: str,
    task_name: str,
    question: str,
    max_iterations: int = 3,
    max_results_per_query: int = 15,
    keyword_extraction_template: Optional[str] = None,
    iteration_decision_template: Optional[str] = None,
) -> IterativeRetrievalResult:
    """
    Run iterative retrieval: VLM keywords -> search -> merge -> VLM decision.

    Args:
        retriever: LocalPatientRetriever instance.
        vlm_adapter: Model adapter.
        vlm_model: Loaded model.
        vlm_processor: Loaded processor.
        person_id: Patient ID (string).
        task_name: Task name for context.
        question: Question for context.
        max_iterations: Maximum retrieval iterations.
        max_results_per_query: Max results per search.
        keyword_extraction_template: Override for keyword prompt.
        iteration_decision_template: Override for decision prompt.

    Returns:
        IterativeRetrievalResult with timeline_str, iterations_log, all_keywords.
    """
    kw_tpl = keyword_extraction_template or KEYWORD_EXTRACTION_TEMPLATE
    dec_tpl = iteration_decision_template or ITERATION_DECISION_TEMPLATE

    person_id_str = str(person_id).strip()
    all_results: List[Dict[str, Any]] = []
    seen_ids: set = set()
    iterations_log: List[Dict[str, Any]] = []
    all_keywords_flat: List[str] = []

    prev_summary = ""

    for iteration in range(1, max_iterations + 1):
        # 1. Generate keywords
        if iteration == 1:
            prompt = kw_tpl.format(
                task_name=task_name,
                question=question,
                prev_context="",
            )
        else:
            prompt = KEYWORD_EXTRACTION_WITH_PREV.format(
                task_name=task_name,
                question=question,
                prev_summary=prev_summary[:500] if prev_summary else "(none)",
            )

        try:
            kw_raw = _run_vlm_text_only(
                vlm_adapter, vlm_model, vlm_processor, prompt, max_tokens=128
            )
        except Exception as e:
            logger.warning("Keyword extraction failed: %s, using fallback", e)
            kw_raw = task_name or question[:100]

        keywords = _parse_keywords(kw_raw)
        if not keywords:
            keywords = [task_name] if task_name else [question[:80].strip()]
            logger.info("Keyword fallback: %s", keywords)

        query = " ".join(keywords)
        all_keywords_flat.extend(keywords)

        # 2. Search
        results = retriever.search(
            person_id=person_id_str,
            query=query,
            max_results=max_results_per_query,
        )

        num_found = len(results)

        # 3. Merge and dedupe
        for r in results:
            rid = r.get("id")
            if rid and rid not in seen_ids:
                seen_ids.add(rid)
                all_results.append(r)

        total_unique = len(all_results)

        # 4. Format for preview/summary
        formatted = format_retrieved_events(all_results, exclude_report=True)
        sample_preview = formatted[:500] if formatted else ""

        if iteration > 1:
            prev_summary = formatted

        # 5. Decide STOP or CONTINUE
        summary_context = ""
        if sample_preview:
            summary_context = f"Summary of what was found:\n{sample_preview}"

        dec_prompt = dec_tpl.format(
            task_name=task_name,
            question=question[:500],
            current_iteration=iteration,
            max_iterations=max_iterations,
            keywords=", ".join(keywords),
            num_found=num_found,
            total_unique=total_unique,
            summary_context=summary_context,
        )

        try:
            dec_raw = _run_vlm_text_only(
                vlm_adapter, vlm_model, vlm_processor, dec_prompt, max_tokens=16
            )
        except Exception as e:
            logger.warning("Iteration decision failed: %s, defaulting to STOP", e)
            dec_raw = "STOP"

        decision = _parse_decision(dec_raw)

        iterations_log.append({
            "iteration": iteration,
            "keywords": keywords,
            "num_results": num_found,
            "total_unique_so_far": total_unique,
            "decision": decision,
            "sample_events_preview": sample_preview,
        })

        if decision == "STOP":
            break

    timeline_str = format_retrieved_events(all_results, exclude_report=True)
    return IterativeRetrievalResult(
        timeline_str=timeline_str,
        iterations_log=iterations_log,
        all_keywords=list(dict.fromkeys(all_keywords_flat)),
    )
