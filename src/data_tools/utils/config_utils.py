import yaml
import json
from google.cloud import bigquery
from google.auth import default

_bq_client = None


def load_tasks_from_config(config_path):
    """Load task list from YAML config file."""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            tasks = config.get('tasks', [])
            return set(tasks)
    except Exception as e:
        print(f"Error loading config from {config_path}: {e}")
        return set()

def load_tasks_and_base_dir(config_path: str):
    """Load task list and base_dir from YAML config."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    tasks = config.get("tasks", [])
    base_dir = config.get("paths", {}).get("base_dir", "")
    return list(tasks), base_dir


def load_task_source_csv(valid_tasks_json_path: str):
    """Load task_name -> task_source_csv from valid_tasks.json."""
    with open(valid_tasks_json_path, "r") as f:
        tasks = json.load(f)
    return {t["task_name"]: t["task_source_csv"] for t in tasks if t.get("task_source_csv")}

def get_bq_client():
    global _bq_client
    if _bq_client is None:
        project_id = "som-nero-plevriti-deidbdf"
        
        # 1. Broaden the scope to cloud-platform
        # 2. Let the library find the JSON file automatically (it checks ADC paths)
        credentials, auth_project_id = default(
            scopes=["https://www.googleapis.com/auth/cloud-platform"]
        )
        
        # Use the project_id you want for data, but ensure credentials have a quota project
        _bq_client = bigquery.Client(
            project=project_id, 
            credentials=credentials
        )
    return _bq_client