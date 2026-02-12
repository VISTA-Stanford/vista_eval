"""
Prompt templates for VLM keyword extraction (5 keywords + reasoning).
"""
# Template placeholders: {task_query}, {patient_timeline}, {searched_keywords}
# - patient_timeline: formatted events from previous iterations (or "No evidence retrieved yet." on first)
# - searched_keywords: comma-separated keywords already searched (or "No previous searches." on first)
KEYWORD_EXTRACTION_TEMPLATE = """
You are an expert Clinical AI Research Agent. Your goal is to iteratively retrieve a patient timeline to answer a specific clinical question: {task_query}

### CURRENT PROGRESS
<current_evidence>
{patient_timeline}
</current_evidence>

<search_history>
{searched_keywords}
</search_history>

### INSTRUCTIONS
1. **Initial Step (Cold Start)**: If <current_evidence> is empty or indicates no evidence has been retrieved, focus on identifying the foundational information needed (e.g., hospital codes, initial diagnosis, the most recent radiology reports, or pathology results) to begin answering the query.
2. **Gap Analysis (Subsequent Turns)**: If evidence is present, analyze it against the {task_query}. Identify specific "missing links" (e.g., a specific scan date, a biopsy measurement, or a treatment start date).
3. **Search Strategy**: Select EXACTLY 5 search terms highly specific to the missing information or foundational data. Use clinical terminology (e.g., "RECIST" instead of "tumor size").
4. **Keyword Optimization**: Use medically relevant, clinical terminology (e.g., "RECIST" instead of "tumor size") to improve BM25 retrieval accuracy.
5. **Negative Constraint**: DO NOT use any keywords found in <search_history>. If a previous search yielded no new info, pivot to a new clinical term.

### RESPONSE FORMAT (STRICT)
You must follow the format shown in the examples. Provide your reasoning inside <clinical_reasoning> tags and the final keyword list inside <answer> tags. Do not include text outside of the tags and do not include any extra text.

### OUTPUT EXAMPLES
**Example 1: Initial Iteration (No Evidence)**
<clinical_reasoning>
No evidence has been retrieved yet. To answer whether the tumor has progressed, I first need to find the baseline imaging and the most recent CT reports to establish a comparison.
</clinical_reasoning>
<answer>
["Malignant neoplasm", "CT abdomen", "oncology notes", "baseline imaging", "radiology report"]
</answer>

**Example 2: Subsequent Iteration (Gap Analysis)**
<clinical_reasoning>
I will not repeat any keywords from the <search_history> tags. Current evidence shows a CT scan from 2023-01-10 but lacks a follow-up to determine progression. I need to find the next scan or any pathology reports after January 2023.
</clinical_reasoning>
<answer>
["follow-up CT", "embolism", "metformin", "new lesions", "respiratory failure"]
</answer>

**Example 3: Subsequent Iteration (Gap Analysis)**
<clinical_reasoning>
I will not repeat any keywords from the <search_history> tags. Currently information pertaining to hospital codes outside of radiology and pathology notes are missing.
</clinical_reasoning>
<answer>
["ICD10CM", "cancer stage", "pneumonitis", "ICD-C34.82", "lung failure"]
</answer>
"""
