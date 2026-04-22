geng i#!/bin/bash

# RoboCasa Benchmark Evaluation Script
# This script runs all 24 tasks from the RoboCasa benchmark and reports results

# Configuration
N_EPISODES=${N_EPISODES:-1000}  # Number of episodes per task (1000 for full benchmark based on 81.1% precision)
N_ENVS=${N_ENVS:-5}            # Number of parallel environments
MAX_STEPS=${MAX_STEPS:-720}    # Max steps per episode
N_ACTION_STEPS=${N_ACTION_STEPS:-8}
HOST=${HOST:-127.0.0.1}
PORT=${PORT:-5555}

PYTHON_BIN="gr00t/eval/sim/robocasa/robocasa_uv/.venv/bin/python"
ROLLOUT_SCRIPT="gr00t/eval/rollout_policy.py"

# Output file for results
RESULTS_FILE="robocasa_benchmark_results_$(date +%Y%m%d_%H%M%S).txt"

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

echo "=============================================="
echo "RoboCasa Benchmark Evaluation"
echo "=============================================="
echo "Episodes per task: $N_EPISODES"
echo "Parallel environments: $N_ENVS"
echo "Total tasks: 24"
echo "Results will be saved to: $RESULTS_FILE"
echo ""
echo "Estimated time: ~$(echo "$N_EPISODES * 24 * 25 / $N_ENVS / 60" | bc) minutes"
echo "(Based on ~25 sec/episode with $N_ENVS parallel envs)"
echo "=============================================="
echo ""

# Check if server is running
if ! nc -z $HOST $PORT 2>/dev/null; then
    echo "ERROR: Policy server not detected at $HOST:$PORT"
    echo "Please start the server first in another terminal:"
    echo ""
    echo "  uv run python gr00t/eval/run_gr00t_server.py \\"
    echo "      --model-path nvidia/GR00T-N1.6-3B \\"
    echo "      --embodiment-tag ROBOCASA_PANDA_OMRON \\"
    echo "      --use-sim-policy-wrapper"
    echo ""
    exit 1
fi

# Initialize results file
echo "RoboCasa Benchmark Results - $(date)" > "$RESULTS_FILE"
echo "Episodes per task: $N_EPISODES" >> "$RESULTS_FILE"
echo "=============================================" >> "$RESULTS_FILE"
echo "" >> "$RESULTS_FILE"

# Arrays to store results
declare -a SUCCESS_RATES

# Run each task
for i in "${!TASKS[@]}"; do
    TASK="${TASKS[$i]}"
    TASK_NUM=$((i + 1))
    TASK_NAME=$(echo "$TASK" | cut -d'/' -f2)
    
    echo "[$TASK_NUM/${#TASKS[@]}] Running: $TASK_NAME"
    echo "----------------------------------------"
    
    # Run evaluation and capture output
    OUTPUT=$(CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} $PYTHON_BIN $ROLLOUT_SCRIPT \
        --n_episodes $N_EPISODES \
        --policy_client_host $HOST \
        --policy_client_port $PORT \
        --max_episode_steps=$MAX_STEPS \
        --env_name "$TASK" \
        --n_action_steps $N_ACTION_STEPS \
        --n_envs $N_ENVS 2>&1)
    
    # Extract success rate from output
    SUCCESS_RATE=$(echo "$OUTPUT" | grep "success rate:" | tail -1 | awk '{print $NF}')
    
    if [ -z "$SUCCESS_RATE" ]; then
        echo "  WARNING: Could not extract success rate for $TASK_NAME"
        echo "  Output: $OUTPUT"
        SUCCESS_RATE="N/A"
    else
        SUCCESS_RATES+=("$SUCCESS_RATE")
        PERCENTAGE=$(echo "$SUCCESS_RATE * 100" | bc -l 2>/dev/null || echo "$SUCCESS_RATE")
        echo "  Success rate: ${PERCENTAGE}%"
    fi
    
    # Log to file
    echo "$TASK: $SUCCESS_RATE" >> "$RESULTS_FILE"
    echo ""
done

echo "=============================================="
echo "FINAL RESULTS"
echo "=============================================="

# Calculate average
if [ ${#SUCCESS_RATES[@]} -gt 0 ]; then
    SUM=0
    COUNT=0
    for rate in "${SUCCESS_RATES[@]}"; do
        if [[ "$rate" != "N/A" ]]; then
            SUM=$(echo "$SUM + $rate" | bc -l)
            COUNT=$((COUNT + 1))
        fi
    done
    
    if [ $COUNT -gt 0 ]; then
        AVERAGE=$(echo "scale=4; $SUM / $COUNT" | bc -l)
        AVERAGE_PCT=$(echo "scale=2; $AVERAGE * 100" | bc -l)
        echo ""
        echo "Average success rate: ${AVERAGE_PCT}%"
        echo "" >> "$RESULTS_FILE"
        echo "=============================================" >> "$RESULTS_FILE"
        echo "Average: $AVERAGE (${AVERAGE_PCT}%)" >> "$RESULTS_FILE"
    fi
fi

echo ""
echo "Results saved to: $RESULTS_FILE"
echo ""

# Print results table
echo "| Task | Success Rate |"
echo "| ---- | ------------ |"
for i in "${!TASKS[@]}"; do
    TASK="${TASKS[$i]}"
    TASK_NAME=$(echo "$TASK" | cut -d'/' -f2)
    if [ $i -lt ${#SUCCESS_RATES[@]} ]; then
        RATE="${SUCCESS_RATES[$i]}"
        if [[ "$RATE" != "N/A" ]]; then
            PCT=$(echo "scale=1; $RATE * 100" | bc -l)
            echo "| \`$TASK\` | ${PCT}% |"
        else
            echo "| \`$TASK\` | N/A |"
        fi
    fi
done
echo "| **Average** | ${AVERAGE_PCT}% |"

