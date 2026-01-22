#!/usr/bin/env bash

# Configuration
CONFIG_FILE="configs/all_tasks.yaml"
MAX_MODELS=4

export HF_TOKEN="${HF_TOKEN:?set HF_TOKEN}

source /home/dcunhrya/vista_eval/.venv/bin/activate

# Extract models from YAML config file using Python
MODEL_LIST=$(cd /home/dcunhrya/vista_eval && python3 -c "
import yaml
import sys

try:
    with open('$CONFIG_FILE', 'r') as f:
        config = yaml.safe_load(f)
    
    models = config.get('models', [])
    if not models:
        print('ERROR: No models found in $CONFIG_FILE', file=sys.stderr)
        print('Please add a \"models\" section with type and name for each model.', file=sys.stderr)
        sys.exit(1)
    
    if len(models) > $MAX_MODELS:
        print(f'ERROR: Too many models ({len(models)}). Maximum is $MAX_MODELS.', file=sys.stderr)
        sys.exit(1)
    
    for model in models:
        if 'type' not in model or 'name' not in model:
            print('ERROR: Each model must have \"type\" and \"name\" fields.', file=sys.stderr)
            sys.exit(1)
        print(f\"{model['type']}|{model['name']}\")
except FileNotFoundError:
    print(f'ERROR: Config file $CONFIG_FILE not found.', file=sys.stderr)
    sys.exit(1)
except yaml.YAMLError as e:
    print(f'ERROR: Failed to parse YAML: {e}', file=sys.stderr)
    sys.exit(1)
except Exception as e:
    print(f'ERROR: {e}', file=sys.stderr)
    sys.exit(1)
")

if [ $? -ne 0 ]; then
    echo "$MODEL_LIST" >&2
    exit 1
fi

# Parse into arrays
MODEL_COUNT=0
while IFS='|' read -r model_type model_name; do
    if [ -n "$model_type" ] && [ -n "$model_name" ]; then
        MODEL_TYPES[$MODEL_COUNT]="$model_type"
        MODEL_NAMES[$MODEL_COUNT]="$model_name"
        MODEL_COUNT=$((MODEL_COUNT + 1))
    fi
done < <(echo "$MODEL_LIST")

if [ $MODEL_COUNT -eq 0 ]; then
    echo "ERROR: No valid models found in $CONFIG_FILE"
    echo "Please ensure the 'models' section exists with 'type' and 'name' for each model."
    exit 1
fi

# Tasks are read from the config file by default (via run.py)
# Command line arguments can override if needed, but typically not used
TASK_LIST=""

echo "Starting Inference Orchestrator for $MODEL_COUNT model(s)..."
for idx in $(seq 0 $((MODEL_COUNT - 1))); do
    echo "  Model $((idx + 1)): Type=${MODEL_TYPES[$idx]}, Name=${MODEL_NAMES[$idx]}"
done

if [ -n "$TASK_LIST" ]; then
    echo "Running specific tasks: $TASK_LIST"
fi

# Create log directory
LOG_DIR="logs/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

# Function to run a single model
run_model() {
    local gpu_id=$1
    local model_type=$2
    local model_name=$3
    local log_file="$LOG_DIR/model_${gpu_id}_$(echo $model_name | tr '/' '_').log"
    
    echo "[GPU $gpu_id] Starting ${model_type}/${model_name} - Log: $log_file"
    
    # Set CUDA_VISIBLE_DEVICES for this process
    export CUDA_VISIBLE_DEVICES=$gpu_id
    
    if [ -z "$TASK_LIST" ]; then
        python3 src/vista_run/run.py --config "$CONFIG_FILE" --type "$model_type" --name "$model_name" > "$log_file" 2>&1
    else
        python3 src/vista_run/run.py --config "$CONFIG_FILE" --type "$model_type" --name "$model_name" --tasks $TASK_LIST > "$log_file" 2>&1
    fi
    
    local exit_code=$?
    if [ $exit_code -eq 0 ]; then
        echo "[GPU $gpu_id] Completed ${model_type}/${model_name} successfully"
    else
        echo "[GPU $gpu_id] Failed ${model_type}/${model_name} with exit code $exit_code"
    fi
    return $exit_code
}

# Run models in parallel (background jobs)
declare -a PIDS
for idx in $(seq 0 $((MODEL_COUNT - 1))); do
    run_model $idx "${MODEL_TYPES[$idx]}" "${MODEL_NAMES[$idx]}" &
    PIDS[$idx]=$!
done

# Wait for all background jobs to complete
echo "Waiting for all models to complete..."
FAILED_COUNT=0
for idx in $(seq 0 $((MODEL_COUNT - 1))); do
    wait ${PIDS[$idx]}
    if [ $? -ne 0 ]; then
        FAILED_COUNT=$((FAILED_COUNT + 1))
    fi
done

echo ""
echo "All models completed. Logs available in: $LOG_DIR"
if [ $FAILED_COUNT -gt 0 ]; then
    echo "Warning: $FAILED_COUNT model(s) failed. Check logs for details."
    exit 1
else
    echo "All models finished successfully!"
    exit 0
fi