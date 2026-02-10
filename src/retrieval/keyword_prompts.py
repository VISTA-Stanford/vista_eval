"""
Prompt templates for VLM keyword extraction and iteration decision.
"""

KEYWORD_EXTRACTION_TEMPLATE = """Given the following clinical task and question, generate search keywords to find relevant patient timeline events.

Task: {task_name}
Question: {question}
{prev_context}

Output exactly 3-7 comma-separated medical search terms (conditions, procedures, labs, medications, findings).
Output ONLY the keywords, no other text. Example: lung cancer, radiation therapy, progression, biopsy"""

KEYWORD_EXTRACTION_WITH_PREV = """Given the following clinical task and question, generate search keywords to find relevant patient timeline events.

Task: {task_name}
Question: {question}
Previously retrieved summary (first 500 chars): {prev_summary}

Output exactly 3-7 comma-separated medical search terms to find ADDITIONAL relevant events not yet covered.
Output ONLY the keywords, no other text."""

ITERATION_DECISION_TEMPLATE = """You are deciding whether to search again for more patient timeline information.

Task: {task_name}
Question: {question}
Iteration: {current_iteration} of {max_iterations}
Keywords used this iteration: {keywords}
Number of timeline events retrieved this iteration: {num_found}
Total unique events retrieved so far: {total_unique}
{summary_context}

If we have ENOUGH information to answer the question, reply with exactly: STOP
If we need MORE information, reply with exactly: CONTINUE

Output ONLY one word: STOP or CONTINUE"""
