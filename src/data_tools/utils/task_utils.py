"""
Task grouping and task mapping utilities (valid_tasks.json, related tasks).
"""
import json
import re


def group_related_tasks(task_names):
    """
    Group tasks that differ only by year number (e.g., died_of_cancer_1_yr, died_of_cancer_2_yr).

    Returns:
        dict mapping base_name to sorted list of task names (sorted by year)
    """
    task_groups = {}
    pattern = r"^(.+)_(\d+)_yr$"

    for task_name in task_names:
        match = re.match(pattern, task_name)
        if match:
            base_name = match.group(1)
            year = int(match.group(2))
            if base_name not in task_groups:
                task_groups[base_name] = []
            task_groups[base_name].append((year, task_name))

    result = {}
    for base_name, tasks in task_groups.items():
        if len(tasks) > 1:
            sorted_tasks = sorted(tasks, key=lambda x: x[0])
            result[base_name] = [task_name for _, task_name in sorted_tasks]
    return result


def load_task_mappings(valid_tasks_json_path):
    """
    Load task mappings from valid_tasks.json.

    Returns:
        task_exclusions: dict mapping task_name to sets of excluded label values
        task_table_map: dict mapping task_name to BigQuery table name (task_source_csv)
    """
    try:
        with open(valid_tasks_json_path, "r") as f:
            tasks = json.load(f)

        task_exclusions = {}
        task_table_map = {}
        for task in tasks:
            task_name = task.get("task_name")
            mapping = task.get("mapping", {})
            task_source_csv = task.get("task_source_csv")

            if task_source_csv:
                task_table_map[task_name] = task_source_csv

            excluded_labels = set()
            excluded_labels.add(-1)
            excluded_labels.add("-1")
            for label_str, mapped_value in mapping.items():
                if mapped_value == "Insufficient follow-up or missing data":
                    excluded_labels.add(label_str)
                    try:
                        excluded_labels.add(int(label_str))
                    except ValueError:
                        pass
                    try:
                        excluded_labels.add(float(label_str))
                    except ValueError:
                        pass

            if excluded_labels:
                task_exclusions[task_name] = excluded_labels

        return task_exclusions, task_table_map
    except Exception as e:
        print(f"Error loading task mappings from {valid_tasks_json_path}: {e}")
        return {}, {}
