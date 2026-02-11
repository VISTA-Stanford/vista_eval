"""
Iterative VLM-driven patient timeline retrieval.

Fixed iterations, 5 keywords per step, top 5 records per keyword, no VLM decision.
"""

import json
import logging
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

from retrieval.format_events import format_retrieved_events
from retrieval.keyword_prompts import KEYWORD_EXTRACTION_TEMPLATE

if TYPE_CHECKING:
    from retrieval.local_retriever import LocalPatientRetriever

logger = logging.getLogger(__name__)


def _run_vlm_text_only(
    adapter: Any,
    model: Any,
    processor: Any,
    prompt: str,
    max_tokens: int = 256,
) -> str:
    """
    Run VLM inference with text-only prompt (no image).
    """
    item = {"question": prompt}
    messages = [adapter.create_template(item)]
    inputs = adapter.prepare_inputs(messages, processor, model)
    outputs = adapter.infer(model, processor, inputs, max_tokens)
    if isinstance(outputs, list) and outputs:
        return str(outputs[0]).strip()
    return ""


def _run_vlm_batch(
    adapter: Any,
    model: Any,
    processor: Any,
    prompts: List[str],
    max_tokens: int = 256,
) -> List[str]:
    """
    Run VLM inference with a batch of text-only prompts.
    Returns list of raw output strings (one per prompt).
    """
    if not prompts:
        return []
    messages = [adapter.create_template({"question": p}) for p in prompts]
    inputs = adapter.prepare_inputs(messages, processor, model)
    outputs = adapter.infer(model, processor, inputs, max_tokens)
    if not isinstance(outputs, list):
        return [""] * len(prompts)
    return [str(o).strip() if o is not None else "" for o in outputs]


def _parse_reasoning_and_keywords(raw: str) -> Tuple[str, List[str]]:
    """
    Parse format:
     <clinical_reasoning>... [Reasoning]: ... </clinical_reasoning>

    <answer>
    ["k1", "k2", "k3", "k4", "k5"]
    </answer>

    Returns (reasoning_str, keywords_list). Keywords truncated to 5, padded if fewer.
    """
    reasoning = ""
    keywords: List[str] = []

    if not raw or not str(raw).strip():
        return reasoning, keywords

    text = str(raw).strip()

    # Extract reasoning from <think>...</think>
    think_match = re.search(
        r"<clinical_reasoning>\s*(.+?)\s*</clinical_reasoning>",
        text,
        re.IGNORECASE | re.DOTALL,
    )
    if think_match:
        reasoning = think_match.group(1).strip()

    # Extract keywords from <answer>["k1", "k2", ...]</answer>
    answer_match = re.search(
        r"<answer>\s*(.+?)\s*</answer>",
        text,
        re.IGNORECASE | re.DOTALL,
    )
    if answer_match:
        inner = answer_match.group(1).strip()
        # Parse JSON array: ["k1", "k2", "k3", "k4", "k5"]
        try:
            keywords = json.loads(inner)
            if isinstance(keywords, list):
                keywords = [str(k).strip() for k in keywords if k][:5]
        except json.JSONDecodeError:
            # Fallback: split on comma, strip quotes
            keywords = [
                k.strip().strip('"\'') for k in re.findall(r'["\']([^"\']*)["\']', inner)
            ][:5]
            if not keywords:
                keywords = [p.strip().strip('"\'') for p in inner.split(",") if p.strip()][:5]

    # Fallback: comma-separated or plain list
    if not keywords:
        parts = [p.strip().strip('"\'') for p in text.split(",") if p.strip()]
        keywords = parts[:5]

    return reasoning, keywords


def _ensure_five_keywords(
    keywords: List[str],
    task_name: str,
    question: str,
) -> List[str]:
    """Ensure exactly 5 keywords; pad with fallbacks if fewer, truncate if more."""
    if len(keywords) >= 5:
        return keywords[:5]
    fallbacks = [task_name] if task_name else []
    if question:
        # Extract first few meaningful words from question (avoid very short words)
        words = [w for w in question.split() if len(w) > 3][:5]
        fallbacks.extend(words)
    while len(keywords) < 5 and fallbacks:
        candidate = fallbacks.pop(0)
        if candidate and candidate not in keywords:
            keywords.append(candidate)
    return keywords[:5]


@dataclass
class IterativeRetrievalResult:
    """Result of iterative retrieval."""

    timeline_str: str
    iterations_log: List[Dict[str, Any]] = field(default_factory=list)
    all_keywords: List[str] = field(default_factory=list)
    keyword_reasoning: List[str] = field(default_factory=list)


def run_iterative_retrieval(
    retriever: "LocalPatientRetriever",
    vlm_adapter: Any,
    vlm_model: Any,
    vlm_processor: Any,
    person_id: str,
    task_name: str,
    question: str,
    max_iterations: int = 3,
    keywords_per_iteration: int = 5,
    records_per_keyword: int = 5,
    keyword_extraction_template: Optional[str] = None,
) -> IterativeRetrievalResult:
    """
    Run iterative retrieval: fixed iterations, 5 keywords, 5 records per keyword.

    No VLM decision; always runs full max_iterations.
    Keywords regenerated each iteration using retrieved timeline as context.
    """
    kw_tpl = keyword_extraction_template or KEYWORD_EXTRACTION_TEMPLATE

    person_id_str = str(person_id).strip()
    all_results: List[Dict[str, Any]] = []
    seen_ids: set = set()
    iterations_log: List[Dict[str, Any]] = []
    all_keywords_flat: List[str] = []
    keyword_reasoning_list: List[str] = []

    prev_timeline = ""
    searched_keywords_list: List[str] = []

    for iteration in range(1, max_iterations + 1):
        # 1. Build prompt with current progress
        task_query = question or task_name
        patient_timeline = (
            prev_timeline[:2000] if prev_timeline else "No evidence retrieved yet."
        )
        searched_keywords = (
            ", ".join(searched_keywords_list) if searched_keywords_list else "No previous searches."
        )
        prompt = kw_tpl.format(
            task_query=task_query,
            patient_timeline=patient_timeline,
            searched_keywords=searched_keywords,
        )

        reasoning = ""
        try:
            kw_raw = _run_vlm_text_only(
                vlm_adapter, vlm_model, vlm_processor, prompt, max_tokens=256
            )
        except Exception as e:
            logger.warning("Keyword extraction failed: %s, using fallback", e)
            kw_raw = ""
            reasoning = f"Fallback: {e}"

        if kw_raw:
            reasoning_parsed, keywords = _parse_reasoning_and_keywords(kw_raw)
            if reasoning_parsed:
                reasoning = reasoning_parsed
            keywords = _ensure_five_keywords(keywords, task_name, question)
        else:
            keywords = _ensure_five_keywords([], task_name, question)

        keyword_reasoning_list.append(reasoning)
        all_keywords_flat.extend(keywords)
        searched_keywords_list.extend(keywords)

        # 2. Search per keyword (top 5 records each)
        results_this_iter: List[Dict[str, Any]] = []
        num_per_keyword: List[int] = []

        for kw in keywords:
            res = retriever.search(
                person_id=person_id_str,
                query=kw,
                max_results=records_per_keyword,
            )
            num_per_keyword.append(len(res))
            for r in res:
                rid = r.get("id")
                if rid and rid not in seen_ids:
                    seen_ids.add(rid)
                    all_results.append(r)
                    results_this_iter.append(r)

        total_unique = len(all_results)

        # 3. Format timeline for next iteration context
        prev_timeline = format_retrieved_events(all_results, exclude_report=True)

        iterations_log.append({
            "iteration": iteration,
            "keywords": keywords,
            "all_keywords_so_far": list(dict.fromkeys(searched_keywords_list)),
            "num_results_per_keyword": num_per_keyword,
            "total_unique_so_far": total_unique,
            "keyword_reasoning": reasoning,
            "raw_model_output": kw_raw,
        })

    timeline_str = format_retrieved_events(all_results, exclude_report=True)
    return IterativeRetrievalResult(
        timeline_str=timeline_str,
        iterations_log=iterations_log,
        all_keywords=list(dict.fromkeys(all_keywords_flat)),
        keyword_reasoning=keyword_reasoning_list,
    )


def run_iterative_retrieval_batch(
    retriever: "LocalPatientRetriever",
    vlm_adapter: Any,
    vlm_model: Any,
    vlm_processor: Any,
    batch_data: List[Dict[str, str]],
    task_name: str,
    max_iterations: int = 3,
    keywords_per_iteration: int = 5,
    records_per_keyword: int = 5,
    keyword_extraction_template: Optional[str] = None,
) -> List[IterativeRetrievalResult]:
    """
    Run iterative retrieval for a batch of patients. Batches VLM keyword extraction
    across patients per iteration for efficiency.

    Args:
        batch_data: List of dicts with keys person_id and question.
        task_name: Task name for fallback keywords.

    Returns:
        List of IterativeRetrievalResult (one per patient, same order as batch_data).
    """
    kw_tpl = keyword_extraction_template or KEYWORD_EXTRACTION_TEMPLATE
    n = len(batch_data)

    # Per-patient state
    all_results_per_patient: List[List[Dict[str, Any]]] = [[] for _ in range(n)]
    seen_ids_per_patient: List[set] = [set() for _ in range(n)]
    iterations_log_per_patient: List[List[Dict[str, Any]]] = [[] for _ in range(n)]
    all_keywords_flat_per_patient: List[List[str]] = [[] for _ in range(n)]
    keyword_reasoning_per_patient: List[List[str]] = [[] for _ in range(n)]
    prev_timeline_per_patient: List[str] = [""] * n
    searched_keywords_per_patient: List[List[str]] = [[] for _ in range(n)]

    for iteration in range(1, max_iterations + 1):
        # 1. Build prompts for all patients
        prompts: List[str] = []
        for i in range(n):
            person_id = str(batch_data[i].get("person_id", "")).strip()
            question = str(batch_data[i].get("question", "")).strip()
            task_query = question or task_name
            patient_timeline = (
                prev_timeline_per_patient[i][:2000]
                if prev_timeline_per_patient[i]
                else "No evidence retrieved yet."
            )
            searched_keywords = (
                ", ".join(searched_keywords_per_patient[i])
                if searched_keywords_per_patient[i]
                else "No previous searches."
            )
            prompt = kw_tpl.format(
                task_query=task_query,
                patient_timeline=patient_timeline,
                searched_keywords=searched_keywords,
            )
            prompts.append(prompt)

        # 2. Single VLM batch call
        try:
            kw_raw_list = _run_vlm_batch(
                vlm_adapter, vlm_model, vlm_processor,
                prompts, max_tokens=256
            )
        except Exception as e:
            logger.warning("Batch keyword extraction failed: %s, using fallback", e)
            kw_raw_list = [""] * n

        # 3. Parse keywords per patient and run BM25
        for i in range(n):
            person_id_str = str(batch_data[i].get("person_id", "")).strip()
            question = str(batch_data[i].get("question", "")).strip()
            kw_raw = kw_raw_list[i] if i < len(kw_raw_list) else ""

            reasoning = ""
            if kw_raw:
                reasoning_parsed, keywords = _parse_reasoning_and_keywords(kw_raw)
                if reasoning_parsed:
                    reasoning = reasoning_parsed
                keywords = _ensure_five_keywords(keywords, task_name, question)
            else:
                keywords = _ensure_five_keywords([], task_name, question)

            keyword_reasoning_per_patient[i].append(reasoning)
            all_keywords_flat_per_patient[i].extend(keywords)
            searched_keywords_per_patient[i].extend(keywords)

            # BM25 search per keyword
            num_per_keyword: List[int] = []
            start_date = batch_data[i].get("start_date")
            end_date = batch_data[i].get("end_date")
            for kw in keywords:
                res = retriever.search(
                    person_id=person_id_str,
                    query=kw,
                    max_results=records_per_keyword,
                    start_date=start_date,
                    end_date=end_date,
                )
                num_per_keyword.append(len(res))
                for r in res:
                    rid = r.get("id")
                    if rid and rid not in seen_ids_per_patient[i]:
                        seen_ids_per_patient[i].add(rid)
                        all_results_per_patient[i].append(r)

            total_unique = len(all_results_per_patient[i])
            iterations_log_per_patient[i].append({
                "iteration": iteration,
                "keywords": keywords,
                "all_keywords_so_far": list(dict.fromkeys(searched_keywords_per_patient[i])),
                "num_results_per_keyword": num_per_keyword,
                "total_unique_so_far": total_unique,
                "keyword_reasoning": reasoning,
                "raw_model_output": kw_raw,
            })

            # Update timeline for next iteration
            prev_timeline_per_patient[i] = format_retrieved_events(
                all_results_per_patient[i], exclude_report=True
            )

    # Build results
    return [
        IterativeRetrievalResult(
            timeline_str=format_retrieved_events(all_results_per_patient[i], exclude_report=True),
            iterations_log=iterations_log_per_patient[i],
            all_keywords=list(dict.fromkeys(all_keywords_flat_per_patient[i])),
            keyword_reasoning=keyword_reasoning_per_patient[i],
        )
        for i in range(n)
    ]
