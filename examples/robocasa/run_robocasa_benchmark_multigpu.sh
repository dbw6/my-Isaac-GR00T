#!/bin/bash

# RoboCasa Multi-GPU Benchmark Evaluation Script
# Runs multiple server-client pairs in parallel across GPUs

set -e

# Configuration
N_EPISODES=${N_EPISODES:-10}
N_ENVS=${N_ENVS:-5}
MAX_STEPS=${MAX_STEPS:-720}
N_ACTION_STEPS=${N_ACTION_STEPS:-8}
BASE_PORT=${BASE_PORT:-5555}

# GPUs to use (space-separated list, e.g., "0 1 2 3")
GPUS=${GPUS:-"2 3 4 6"}
GPU_ARRAY=($GPUS)
NUM_GPUS=${#GPU_ARRAY[@]}

PYTHON_BIN="gr00t/eval/sim/robocasa/robocasa_uv/.venv/bin/python"
ROLLOUT_SCRIPT="gr00t/eval/rollout_policy.py"
SERVER_SCRIPT="gr00t/eval/run_gr00t_server.py"

# Output directory
RESULTS_DIR="robocasa_benchmark_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RESULTS_DIR"

# All 24 benchmark tasks
TASKS=(
    "robocasa_panda_omron/CoffeeSetupMug_PandaOmron_Env"
    "robocasa_panda_omron/CoffeeServeMug_PandaOmron_Env"
    "robocasa_panda_omron/CoffeePressButton_PandaOmron_Env"
    "robocasa_panda_omron/OpenSingleDoor_PandaOmron_Env"
    "robocasa_panda_omron/OpenDoubleDoor_PandaOmron_Env"
    "robocasa_panda_omron/CloseSingleDoor_PandaOmron_Env"
    "robocasa_panda_omron/CloseDoubleDoor_PandaOmron_Env"
    "robocasa_panda_omron/OpenDrawer_PandaOmron_Env"
    "robocasa_panda_omron/CloseDrawer_PandaOmron_Env"
    "robocasa_panda_omron/TurnOnMicrowave_PandaOmron_Env"
    "robocasa_panda_omron/TurnOffMicrowave_PandaOmron_Env"
    "robocasa_panda_omron/PnPCounterToCab_PandaOmron_Env"
    "robocasa_panda_omron/PnPCabToCounter_PandaOmron_Env"
    "robocasa_panda_omron/PnPCounterToSink_PandaOmron_Env"
    "robocasa_panda_omron/PnPSinkToCounter_PandaOmron_Env"
    "robocasa_panda_omron/PnPCounterToMicrowave_PandaOmron_Env"
    "robocasa_panda_omron/PnPMicrowaveToCounter_PandaOmron_Env"
    "robocasa_panda_omron/PnPCounterToStove_PandaOmron_Env"
    "robocasa_panda_omron/PnPStoveToCounter_PandaOmron_Env"
    "robocasa_panda_omron/TurnOnSinkFaucet_PandaOmron_Env"
    "robocasa_panda_omron/TurnOffSinkFaucet_PandaOmron_Env"
    "robocasa_panda_omron/TurnSinkSpout_PandaOmron_Env"
    "robocasa_panda_omron/TurnOnStove_PandaOmron_Env"
    "robocasa_panda_omron/TurnOffStove_PandaOmron_Env"
)

NUM_TASKS=${#TASKS[@]}

echo "=============================================="
echo "RoboCasa Multi-GPU Benchmark Evaluation"
echo "=============================================="
echo "GPUs: ${GPUS} ($NUM_GPUS GPUs)"
echo "Episodes per task: $N_EPISODES"
echo "Parallel environments per GPU: $N_ENVS"
echo "Total tasks: $NUM_TASKS"
echo "Results directory: $RESULTS_DIR"
echo ""
echo "Estimated time: ~$(echo "$N_EPISODES * $NUM_TASKS * 25 / $N_ENVS / $NUM_GPUS / 60" | bc) minutes"
echo "(With $NUM_GPUS GPUs running in parallel)"
echo "=============================================="
echo ""

# Function to cleanup background processes
cleanup() {
    echo "Cleaning up..."
    for pid in "${SERVER_PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            kill "$pid" 2>/dev/null || true
        fi
    done
    # Clean up temp files
    rm -f "$RESULTS_DIR/.task_queue" "$RESULTS_DIR/.task_queue.tmp" "$RESULTS_DIR/.queue_lock" 2>/dev/null || true
    wait
    echo "Cleanup complete."
}
trap cleanup EXIT

# Start servers on each GPU
declare -a SERVER_PIDS
echo "Starting $NUM_GPUS policy servers..."

for i in "${!GPU_ARRAY[@]}"; do
    GPU="${GPU_ARRAY[$i]}"
    PORT=$((BASE_PORT + i))
    
    echo "  Starting server on GPU $GPU, port $PORT..."
    
    CUDA_VISIBLE_DEVICES=$GPU uv run python $SERVER_SCRIPT \
        --model-path nvidia/GR00T-N1.6-3B \
        --embodiment-tag ROBOCASA_PANDA_OMRON \
        --use-sim-policy-wrapper \
        --port $PORT \
        > "$RESULTS_DIR/server_gpu${GPU}.log" 2>&1 &
    
    SERVER_PIDS+=($!)
done

# Wait for all servers to be ready
echo "Waiting for servers to initialize..."
sleep 30  # Initial wait for model loading

for i in "${!GPU_ARRAY[@]}"; do
    PORT=$((BASE_PORT + i))
    echo -n "  Checking port $PORT..."
    
    for attempt in {1..60}; do
        if nc -z 127.0.0.1 $PORT 2>/dev/null; then
            echo " ready!"
            break
        fi
        if [ $attempt -eq 60 ]; then
            echo " FAILED"
            echo "ERROR: Server on port $PORT failed to start. Check $RESULTS_DIR/server_gpu${GPU_ARRAY[$i]}.log"
            exit 1
        fi
        sleep 5
    done
done

echo ""
echo "All servers ready. Starting evaluation..."
echo ""

# Create task queue file
TASK_QUEUE="$RESULTS_DIR/.task_queue"
LOCK_FILE="$RESULTS_DIR/.queue_lock"
for TASK in "${TASKS[@]}"; do
    echo "$TASK" >> "$TASK_QUEUE"
done

# Function to get next task from queue (with locking)
get_next_task() {
    (
        flock -x 200
        if [ -s "$TASK_QUEUE" ]; then
            head -1 "$TASK_QUEUE"
            tail -n +2 "$TASK_QUEUE" > "$TASK_QUEUE.tmp"
            mv "$TASK_QUEUE.tmp" "$TASK_QUEUE"
        fi
    ) 200>"$LOCK_FILE"
}

# Worker function - runs tasks until queue is empty
worker() {
    local GPU_IDX=$1
    local GPU="${GPU_ARRAY[$GPU_IDX]}"
    local PORT=$((BASE_PORT + GPU_IDX))
    
    while true; do
        # Get next task from queue
        local TASK=$(get_next_task)
        
        # Exit if no more tasks
        if [ -z "$TASK" ]; then
            break
        fi
        
        local TASK_NAME=$(echo "$TASK" | cut -d'/' -f2)
        local OUTPUT_FILE="$RESULTS_DIR/${TASK_NAME}.txt"
        
        echo "[GPU $GPU] Running: $TASK_NAME"
        
        CUDA_VISIBLE_DEVICES=$GPU $PYTHON_BIN $ROLLOUT_SCRIPT \
            --n_episodes $N_EPISODES \
            --policy_client_host 127.0.0.1 \
            --policy_client_port $PORT \
            --max_episode_steps=$MAX_STEPS \
            --env_name "$TASK" \
            --n_action_steps $N_ACTION_STEPS \
            --n_envs $N_ENVS \
            > "$OUTPUT_FILE" 2>&1
        
        # Extract and display result
        local SUCCESS_RATE=$(grep "success rate:" "$OUTPUT_FILE" | tail -1 | awk '{print $NF}')
        echo "[GPU $GPU] Completed: $TASK_NAME - Success rate: $SUCCESS_RATE"
    done
}

# Start a worker for each GPU
declare -a WORKER_PIDS
for i in "${!GPU_ARRAY[@]}"; do
    worker "$i" &
    WORKER_PIDS+=($!)
done

# Wait for all workers to complete
echo ""
echo "All workers started. Tasks will be dynamically assigned as GPUs become free..."
wait "${WORKER_PIDS[@]}"

echo ""
echo "=============================================="
echo "AGGREGATING RESULTS"
echo "=============================================="
echo ""

# Aggregate results
FINAL_RESULTS="$RESULTS_DIR/final_results.txt"
echo "RoboCasa Benchmark Results - $(date)" > "$FINAL_RESULTS"
echo "Episodes per task: $N_EPISODES" >> "$FINAL_RESULTS"
echo "GPUs used: $GPUS" >> "$FINAL_RESULTS"
echo "=============================================" >> "$FINAL_RESULTS"
echo "" >> "$FINAL_RESULTS"

# Print results table
echo "| Task | Success Rate |"
echo "| ---- | ------------ |"
echo "| Task | Success Rate |" >> "$FINAL_RESULTS"
echo "| ---- | ------------ |" >> "$FINAL_RESULTS"

SUM=0
COUNT=0

for TASK in "${TASKS[@]}"; do
    TASK_NAME=$(echo "$TASK" | cut -d'/' -f2)
    OUTPUT_FILE="$RESULTS_DIR/${TASK_NAME}.txt"
    
    if [ -f "$OUTPUT_FILE" ]; then
        SUCCESS_RATE=$(grep "success rate:" "$OUTPUT_FILE" | tail -1 | awk '{print $NF}')
        if [ -n "$SUCCESS_RATE" ] && [ "$SUCCESS_RATE" != "N/A" ]; then
            PCT=$(echo "scale=1; $SUCCESS_RATE * 100" | bc -l)
            echo "| \`$TASK\` | ${PCT}% |"
            echo "| \`$TASK\` | ${PCT}% |" >> "$FINAL_RESULTS"
            SUM=$(echo "$SUM + $SUCCESS_RATE" | bc -l)
            COUNT=$((COUNT + 1))
        else
            echo "| \`$TASK\` | N/A |"
            echo "| \`$TASK\` | N/A |" >> "$FINAL_RESULTS"
        fi
    else
        echo "| \`$TASK\` | MISSING |"
        echo "| \`$TASK\` | MISSING |" >> "$FINAL_RESULTS"
    fi
done

# Calculate and print average
if [ $COUNT -gt 0 ]; then
    AVERAGE=$(echo "scale=4; $SUM / $COUNT" | bc -l)
    AVERAGE_PCT=$(echo "scale=2; $AVERAGE * 100" | bc -l)
    echo "| **Average** | ${AVERAGE_PCT}% |"
    echo "| **Average** | ${AVERAGE_PCT}% |" >> "$FINAL_RESULTS"
fi

echo ""
echo "Results saved to: $FINAL_RESULTS"
echo "Individual task logs saved in: $RESULTS_DIR/"

