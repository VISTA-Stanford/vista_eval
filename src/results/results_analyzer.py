import os
import json
import re
import pandas as pd
from pathlib import Path


def _clean_extracted_answer(answer):
    """Cleans an extracted answer: strip, normalize whitespace, remove wrapping quotes and common prefixes/suffixes."""
    if not answer or not str(answer).strip():
        return ""
    answer = str(answer).strip()
    if (answer.startswith('"') and answer.endswith('"')) or (answer.startswith("'") and answer.endswith("'")):
        answer = answer[1:-1].strip()
    prefixes = [
        r'^answer:\s*', r'^response:\s*', r'^the answer is\s*', r'^answer is\s*',
        r'^the response is\s*', r'^response is\s*',
    ]
    for p in prefixes:
        answer = re.sub(p, '', answer, flags=re.IGNORECASE).strip()
    answer = re.sub(r'\s+', ' ', answer)
    answer = re.sub(r'\n+', ' ', answer)
    answer = re.sub(r'\s*\[TRUNCATED.*?\]\s*$', '', answer, flags=re.IGNORECASE).strip()
    answer = re.sub(r'\s*\[.*?truncated.*?\]\s*$', '', answer, flags=re.IGNORECASE).strip()
    return answer.strip()


def _extract_boxed(text):
    """Extract all \\boxed{...} contents, handling nested braces."""
    out = []
    s = str(text)
    i = 0
    needle = '\\boxed{'
    while True:
        start = s.find(needle, i)
        if start == -1:
            break
        start += len(needle)
        depth = 1
        j = start
        while j < len(s) and depth > 0:
            if s[j] == '{':
                depth += 1
            elif s[j] == '}':
                depth -= 1
            j += 1
        if depth == 0:
            out.append(s[start : j - 1].strip())
        i = j
    return out


def _extract_answer_candidates(text):
    """
    Extract all possible answer strings from the model's raw output.
    Used for correctness checking: we consider the model correct if *any* candidate matches.
    - Checks \\boxed{...}, <answer>...</answer>, <label>...</label>, angle brackets <Yes> or <"Yes">.
    - Splits on '|' and checks each segment (tags/boxed/angle brackets + raw trimmed segment).
    - If no '|', adds last word and last phrase (last 2–3 words) as fallbacks.
    """
    if pd.isna(text) or not str(text).strip():
        return []
    text = str(text).strip()
    candidates = []
    seen = set()

    def add(raw):
        c = _clean_extracted_answer(raw)
        if not c:
            return
        k = c.lower()
        if k not in seen:
            seen.add(k)
            candidates.append(c)

    def collect_from(s):
        for boxed in _extract_boxed(s):
            add(boxed)
        for m in re.finditer(r'<answer>(.*?)</answer>', s, re.DOTALL | re.IGNORECASE):
            if m.group(1).strip():
                add(m.group(1))
            else:
                # Empty tags: <answer></answer> No - add text right after closing tag
                rest = s[m.end():].strip()
                if rest:
                    add(rest)
        for m in re.finditer(r'<label>(.*?)</label>', s, re.DOTALL | re.IGNORECASE):
            if m.group(1).strip():
                add(m.group(1))
            else:
                # Empty tags: <label></label> No - add text right after closing tag
                rest = s[m.end():].strip()
                if rest:
                    add(rest)
        # Extract content from angle brackets like <Yes> or <"Yes"> (but not <answer>/<label> tags)
        # Pattern: < followed by optional quote, content, optional quote, >
        # Handles: <Yes>, <"Yes">, <'Yes'>, <Yes, No>
        # Only extract short angle brackets (<= 50 chars) to avoid matching long explanations
        for m in re.finditer(r'<(["\']?)([^<>"\'/]+?)\1>', s):
            content = m.group(2).strip()
            # Skip if it looks like an XML tag (contains / or is a known tag name)
            if content and not content.startswith('/') and content.lower() not in ['answer', 'label']:
                # Only add if it's a short answer (likely to be Yes/No/short label)
                # Skip long explanations that are in angle brackets
                if len(content) <= 50:
                    add(content)

    collect_from(text)

    # Explicit handling for <explanation> | answer format - add the answer after "> |"
    pipe_after_angle = re.search(r'>\s*\|\s*(.+)$', text, re.DOTALL)
    if pipe_after_angle:
        add(pipe_after_angle.group(1))

    if '|' in text:
        # Split on the last occurrence of '|'
        parts = text.rsplit('|', 1)
        if len(parts) == 2:
            left_part, right_part = parts[0].strip(), parts[1].strip()
            # Process both parts
            if left_part:
                collect_from(left_part)
                add(left_part)
            if right_part:
                collect_from(right_part)
                add(right_part)
        else:
            # Fallback: if rsplit doesn't work as expected, process all parts
            for part in text.split('|'):
                part = part.strip()
                if not part:
                    continue
                collect_from(part)
                add(part)
        
        # Final check: if '|' is present, also check the left side of the first '|'
        first_pipe_idx = text.find('|')
        if first_pipe_idx > 0:  # '|' is not at the beginning
            left_side = text[:first_pipe_idx].strip()
            if left_side:
                collect_from(left_side)
                add(left_side)
    else:
        words = text.split()
        if words:
            add(words[-1])
        if len(words) >= 2:
            add(' '.join(words[-2:]))
        if len(words) >= 3:
            add(' '.join(words[-3:]))
        if text:
            add(text)

    return candidates


def _extract_answer(text):
    """
    Extract the primary answer from the model's raw output (for display / cleaned_response).
    Priority: \\boxed{} > <answer> (if non-empty) > <label> > double quotes "answer" > <explanation> | answer > pipe last segment > angle brackets <Yes> or <"Yes"> > last word/phrase > full text.
    If <answer></answer> is empty, continues searching other patterns.
    """
    if pd.isna(text) or not str(text).strip():
        return ""
    text = str(text).strip()
    boxed = _extract_boxed(text)
    if boxed:
        return _clean_extracted_answer(boxed[-1])
    m = re.search(r'<answer>(.*?)</answer>', text, re.DOTALL | re.IGNORECASE)
    if m:
        content = m.group(1).strip()
        if content:
            return _clean_extracted_answer(content)
        # Empty tags: <answer></answer> - check if answer is right after the closing tag
        after = re.search(r'</answer>\s*(.+)$', text, re.DOTALL | re.IGNORECASE)
        if after:
            return _clean_extracted_answer(after.group(1))
        # If <answer></answer> is empty and no content after, continue searching other patterns
        # (fall through to <label>, pipe, angle brackets, etc.) - don't return here
    
    # Check <label> tags (always check, even if <answer></answer> was empty)
    m = re.search(r'<label>(.*?)</label>', text, re.DOTALL | re.IGNORECASE)
    if m:
        content = m.group(1).strip()
        if content:
            return _clean_extracted_answer(content)
        # Empty tags: <label></label> No - answer is right after the closing tag
        after = re.search(r'</label>\s*(.+)$', text, re.DOTALL | re.IGNORECASE)
        if after:
            return _clean_extracted_answer(after.group(1))

    # Check for answers in double quotes: "answer"
    quoted_match = re.search(r'"([^"]+)"', text)
    if quoted_match:
        quoted_content = quoted_match.group(1).strip()
        if quoted_content and len(quoted_content) <= 100:  # Reasonable length limit
            return _clean_extracted_answer(quoted_content)

    # Explicit handling for <explanation> | answer format (reasoning in angle brackets, then pipe, then answer)
    m = re.search(r'>\s*\|\s*(.+)$', text, re.DOTALL)
    if m:
        return _clean_extracted_answer(m.group(1))
    
    # Check for pipes (before angle brackets) to handle cases like "<explanation> | No"
    if '|' in text:
        # Split on the last occurrence of '|'
        parts = text.rsplit('|', 1)
        if len(parts) == 2:
            left_part, right_part = parts[0].strip(), parts[1].strip()
            # Check right part (after last '|') first
            if right_part:
                # Check for angle brackets in the right part (e.g., | <Yes> or | <"Yes">)
                angle_match = re.search(r'<(["\']?)([^<>"\'/]+?)\1>', right_part)
                if angle_match:
                    content = angle_match.group(2).strip()
                    if content and content.lower() not in ['answer', 'label']:
                        return _clean_extracted_answer(content)
                return _clean_extracted_answer(right_part)
            # If right part is empty, check left part
            if left_part:
                return _clean_extracted_answer(left_part)
        else:
            # Fallback: if rsplit doesn't work as expected
            parts = [p.strip() for p in text.split('|') if p.strip()]
            if parts:
                last_part = parts[-1]
                # Check for angle brackets in the last part
                angle_match = re.search(r'<(["\']?)([^<>"\'/]+?)\1>', last_part)
                if angle_match:
                    content = angle_match.group(2).strip()
                    if content and content.lower() not in ['answer', 'label']:
                        return _clean_extracted_answer(content)
                return _clean_extracted_answer(last_part)
    
    # Check for angle brackets (e.g., <Yes> or <"Yes">) - but only short ones to avoid matching explanations
    # Only match angle brackets that are short (<= 50 chars) to avoid matching long explanations
    angle_matches = list(re.finditer(r'<(["\']?)([^<>"\'/]+?)\1>', text))
    for angle_match in angle_matches:
        content = angle_match.group(2).strip()
        if content and content.lower() not in ['answer', 'label']:
            # Only extract if it's a short answer (likely to be Yes/No/short label)
            # Skip long explanations that are in angle brackets
            if len(content) <= 50:
                return _clean_extracted_answer(content)

        # Final check: if '|' is present and no answer found yet, check the left side of first '|'
    if '|' in text:
        first_pipe_idx = text.find('|')
        if first_pipe_idx > 0:  # '|' is not at the beginning
            left_side = text[:first_pipe_idx].strip()
            if left_side:
                return _clean_extracted_answer(left_side)
    
    words = text.split()
    if words:
        return _clean_extracted_answer(words[-1])
    return _clean_extracted_answer(text)


def _normalize_label_for_mapping(label):
    """
    Convert label (int, float, or str) to the string key used in valid_tasks.json mapping.
    Mapping keys are always strings like "0", "1", "-1". Labels from CSVs can be ints or
    floats (e.g. 1.0); we normalize so 1.0 -> "1", 0.0 -> "0" for correct lookup.
    """
    if pd.isna(label) or label is None:
        return None
    try:
        f = float(label)
        if f == int(f):
            return str(int(f))
        return str(f)
    except (TypeError, ValueError):
        return str(label)


def map_label_to_answer(label, mapping):
    """
    Map a label (int/float/str) to the correct answer string using the task mapping.
    Uses _normalize_label_for_mapping so 1.0 -> "1", 0 -> "0" etc. match mapping keys.
    """
    if not mapping:
        return label
    key = _normalize_label_for_mapping(label)
    if key is None:
        return None
    return mapping.get(key, label)


def is_answer_correct(model_response, mapped_label):
    """
    True if any extracted answer candidate matches the gold mapped_label (case-insensitive).
    Handles |-splits, \\boxed{}, <answer>/<label>, and last word/phrase fallbacks.
    """
    if pd.isna(mapped_label) or mapped_label is None:
        return False
    target = str(mapped_label).strip().lower()
    if not target:
        return False
    for c in _extract_answer_candidates(model_response):
        if str(c).strip().lower() == target:
            return True
    return False


class DynamicResultsAnalyzer:
    def __init__(self, base_path='/home/dcunhrya/vista_bench', 
                 results_path='/home/dcunhrya/results'):
        self.base_path = Path(base_path)
        self.results_path = Path(results_path)
        
        # Load tasks once into a dictionary for fast lookup by task_name
        # Note: adjust path if your 'tasks' folder name varies (e.g., 'configs')
        tasks_json = self.base_path / 'tasks' / 'valid_tasks.json'
            
        with open(tasks_json, 'r') as f:
            tasks_list = json.load(f)
            self.task_registry = {t['task_name']: t for t in tasks_list}

    def _clean_question(self, text):
        """
        Filters the question to keep only the part before 'OPTIONS'.
        """
        if pd.isna(text):
            return ""
        # Split on 'OPTIONS' (case-insensitive check is safer)
        # Using regex split to handle 'OPTIONS', 'Options', etc.
        parts = re.split(r'\bOPTIONS\b', str(text), flags=re.IGNORECASE)
        return parts[0].strip()

    def analyze_all_models(self):
        print(f"Scanning results in: {self.results_path}")
        result_files = list(self.results_path.rglob("*_results.csv"))
        
        if not result_files:
            print("No result files found. Check your paths.")
            return

        for file_path in result_files:
            if file_path.name == "results_analyzed.csv":
                continue
                
            # Pattern matching for task names
            task_name = file_path.name.replace("_subsampled_results.csv", "").replace("_results.csv", "")
            
            if task_name not in self.task_registry:
                continue

            task_info = self.task_registry[task_name]
            mapping = task_info.get('mapping', {})
            
            print(f"Processing Model: {file_path.parent.name} | Task: {task_name}")
            self._process_file(file_path, mapping)

    def _process_file(self, file_path, mapping):
        try:
            df = pd.read_csv(file_path)
            
            # 1. Filter Question: Remove everything from 'OPTIONS' onwards
            df['question'] = df['question'].apply(self._clean_question)

            # 2. Extract Response: Handle \\boxed{}, <answer>/<label>, pipe |, last word/phrase
            df['cleaned_response'] = df['model_response'].apply(_extract_answer)

            # 3. Map Label: Numeric key -> String label (handles int/float labels vs string mapping keys)
            df['mapped_label'] = df['label'].apply(lambda lbl: map_label_to_answer(lbl, mapping))

            # 4. Accuracy Check: any extracted candidate matches mapped_label
            df['is_correct'] = df.apply(
                lambda x: is_answer_correct(x['model_response'], x['mapped_label']),
                axis=1
            )

            # 5. Save results_analyzed.csv
            cols_to_keep = ['index', 'question', 'label', 'mapped_label', 'model_response', 'cleaned_response', 'is_correct']
            existing_cols = [c for c in cols_to_keep if c in df.columns]
            
            out_path = file_path.parent / "results_analyzed.csv"
            df[existing_cols].to_csv(out_path, index=False)
            
        except Exception as e:
            print(f"Error processing {file_path}: {e}")

if __name__ == "__main__":
    analyzer = DynamicResultsAnalyzer()
    analyzer.analyze_all_models()