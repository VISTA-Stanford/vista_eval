#!/usr/bin/env bash
#
# Same behavior as bq_gcp.sh but with selectable GPU nodes (8 total).
# GPU nodes are read from configs/all_tasks.yaml under weill.gpu_nodes (e.g. [0, 1, 2, 3]).
# Override: ./weill.sh 0 1 2 3   or   WEILL_GPUS="0 2 4 6" ./weill.sh
# If there are more models than nodes, models are queued sequentially per node.
#

set -e

# Avoid vLLM/Pydantic "latin-1 codec can't encode character" when model config has Unicode (e.g. fancy quotes)
export LC_ALL="${LC_ALL:-en_US.UTF-8}"
export LANG="${LANG:-en_US.UTF-8}"
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1

# Project root (directory containing configs/ and src/)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

# Configuration (same as bq_gcp.sh)
CONFIG_FILE="configs/all_tasks_nayebi.yaml"

# Venv: use PROJECT_ROOT venv or override with VISTA_VENV
if [ -n "$VISTA_VENV" ]; then
    source "$VISTA_VENV"
else
    [ -f "$PROJECT_ROOT/.venv/bin/activate" ] && source "$PROJECT_ROOT/.venv/bin/activate" || true
fi

# GPU nodes: from command-line args, or WEILL_GPUS env, or config weill.gpu_nodes
if [ $# -ge 1 ]; then
    WEILL_GPUS="$*"
elif [ -z "$WEILL_GPUS" ]; then
    WEILL_GPUS=$(python3 -c "
import yaml
import sys
try:
    with open('$CONFIG_FILE', 'r') as f:
        config = yaml.safe_load(f)
    weill = config.get('weill') or {}
    nodes = weill.get('gpu_nodes', [])
    if not nodes:
        print('ERROR: No weill.gpu_nodes in $CONFIG_FILE. Add under weill: gpu_nodes: [0, 1, ...]', file=sys.stderr)
        sys.exit(1)
    print(' '.join(str(int(x)) for x in nodes))
except Exception as e:
    print(f'ERROR: {e}', file=sys.stderr)
    sys.exit(1)
")
    if [ $? -ne 0 ]; then
        echo "$WEILL_GPUS" >&2
        exit 1
    fi
fi

# Parse GPU list into array
GPU_IDS=()
for g in $WEILL_GPUS; do
    [[ "$g" =~ ^[0-9]+$ ]] && GPU_IDS+=( "$g" ) || { echo "Invalid GPU id: $g"; exit 1; }
done
NODES=${#GPU_IDS[@]}
if [ "$NODES" -eq 0 ]; then
    echo "ERROR: No valid GPU IDs. Set weill.gpu_nodes in $CONFIG_FILE or pass GPU IDs as arguments."
    exit 1
fi

# Extract models from YAML (no max limit; we queue)
MODEL_LIST=$(python3 -c "
import yaml
import sys

try:
    with open('$CONFIG_FILE', 'r') as f:
        config = yaml.safe_load(f)
    
    models = config.get('models', [])
    if not models:
        print('ERROR: No models found in $CONFIG_FILE', file=sys.stderr)
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
    exit 1
fi

TASK_LIST=""

# Create log directory
LOG_DIR="logs/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

echo "Weill: $MODEL_COUNT model(s) on $NODES GPU node(s): ${GPU_IDS[*]}"
for idx in $(seq 0 $((MODEL_COUNT - 1))); do
    echo "  Model $((idx + 1)): Type=${MODEL_TYPES[$idx]}, Name=${MODEL_NAMES[$idx]}"
done
if [ -n "$TASK_LIST" ]; then
    echo "Running specific tasks: $TASK_LIST"
fi

# Assign each model to a node (round-robin): node (idx % NODES) gets model idx
# So node j runs model indices: j, j+NODES, j+2*NODES, ...
run_model() {
    local gpu_id=$1
    local model_type=$2
    local model_name=$3
    local log_file="$LOG_DIR/model_${gpu_id}_$(echo "$model_name" | tr '/' '_').log"
    echo "[GPU $gpu_id] Starting ${model_type}/${model_name} - Log: $log_file"
    export CUDA_VISIBLE_DEVICES=$gpu_id
    if [ -z "$TASK_LIST" ]; then
        python3 src/vista_run/run_bq.py --config "$CONFIG_FILE" --type "$model_type" --name "$model_name" > "$log_file" 2>&1
    else
        python3 src/vista_run/run_bq.py --config "$CONFIG_FILE" --type "$model_type" --name "$model_name" --tasks $TASK_LIST > "$log_file" 2>&1
    fi
    local exit_code=$?
    if [ $exit_code -eq 0 ]; then
        echo "[GPU $gpu_id] Completed ${model_type}/${model_name} successfully"
    else
        echo "[GPU $gpu_id] Failed ${model_type}/${model_name} with exit code $exit_code"
    fi
    return $exit_code
}

# Run one node's queue: run all models assigned to this node sequentially
run_node_queue() {
    local node_index=$1
    local gpu_id=${GPU_IDS[$node_index]}
    local any_failed=0
    local idx=$node_index
    while [ $idx -lt $MODEL_COUNT ]; do
        run_model "$gpu_id" "${MODEL_TYPES[$idx]}" "${MODEL_NAMES[$idx]}" || any_failed=1
        idx=$((idx + NODES))
    done
    return $any_failed
}

# Run each node's queue in parallel (one background job per node)
FAILED_COUNT=0
declare -a PIDS
for node_index in $(seq 0 $((NODES - 1))); do
    run_node_queue "$node_index" &
    PIDS[$node_index]=$!
done
echo "Waiting for all nodes to complete..."
for node_index in $(seq 0 $((NODES - 1))); do
    if ! wait ${PIDS[$node_index]}; then
        FAILED_COUNT=$((FAILED_COUNT + 1))
    fi
done

echo ""
echo "All models completed. Logs available in: $LOG_DIR"
if [ $FAILED_COUNT -gt 0 ]; then
    echo "Warning: $FAILED_COUNT node queue(s) had failures. Check logs for details."
    exit 1
else
    echo "All models finished successfully!"
    exit 0
fi
