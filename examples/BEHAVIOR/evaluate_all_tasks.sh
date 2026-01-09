#!/bin/bash

# Script to evaluate all 50 BEHAVIOR tasks using the provided checkpoint
# Usage: ./evaluate_all_tasks.sh [--n_episodes N] [--policy_client_host HOST] [--policy_client_port PORT] [--n_envs N] [--output_dir DIR]

set -e

# Default values
N_EPISODES=10
POLICY_CLIENT_HOST="127.0.0.1"
POLICY_CLIENT_PORT=5555
N_ENVS=1
N_ACTION_STEPS=8
MAX_EPISODE_STEPS=999999999
OUTPUT_DIR=""

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --n_episodes)
            N_EPISODES="$2"
            shift 2
            ;;
        --policy_client_host)
            POLICY_CLIENT_HOST="$2"
            shift 2
            ;;
        --policy_client_port)
            POLICY_CLIENT_PORT="$2"
            shift 2
            ;;
        --n_envs)
            N_ENVS="$2"
            shift 2
            ;;
        --n_action_steps)
            N_ACTION_STEPS="$2"
            shift 2
            ;;
        --max_episode_steps)
            MAX_EPISODE_STEPS="$2"
            shift 2
            ;;
        --output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--n_episodes N] [--policy_client_host HOST] [--policy_client_port PORT] [--n_envs N] [--n_action_steps N] [--max_episode_steps N] [--output_dir DIR]"
            exit 1
            ;;
    esac
done

# Get the script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# All 50 BEHAVIOR tasks
TASKS=(
    "sim_behavior_r1_pro/turning_on_radio"
    "sim_behavior_r1_pro/hanging_pictures"
    "sim_behavior_r1_pro/make_microwave_popcorn"
    "sim_behavior_r1_pro/attach_a_camera_to_a_tripod"
    "sim_behavior_r1_pro/picking_up_trash"
    "sim_behavior_r1_pro/clean_a_trumpet"
    "sim_behavior_r1_pro/set_up_a_coffee_station_in_your_kitchen"
    "sim_behavior_r1_pro/chop_an_onion"
    "sim_behavior_r1_pro/spraying_for_bugs"
    "sim_behavior_r1_pro/hiding_Easter_eggs"
    "sim_behavior_r1_pro/cook_bacon"
    "sim_behavior_r1_pro/putting_shoes_on_rack"
    "sim_behavior_r1_pro/clean_boxing_gloves"
    "sim_behavior_r1_pro/preparing_lunch_box"
    "sim_behavior_r1_pro/spraying_fruit_trees"
    "sim_behavior_r1_pro/wash_a_baseball_cap"
    "sim_behavior_r1_pro/rearranging_kitchen_furniture"
    "sim_behavior_r1_pro/setting_the_fire"
    "sim_behavior_r1_pro/bringing_water"
    "sim_behavior_r1_pro/cook_hot_dogs"
    "sim_behavior_r1_pro/setting_mousetraps"
    "sim_behavior_r1_pro/outfit_a_basic_toolbox"
    "sim_behavior_r1_pro/chopping_wood"
    "sim_behavior_r1_pro/putting_dishes_away_after_cleaning"
    "sim_behavior_r1_pro/tidying_bedroom"
    "sim_behavior_r1_pro/wash_dog_toys"
    "sim_behavior_r1_pro/can_meat"
    "sim_behavior_r1_pro/sorting_vegetables"
    "sim_behavior_r1_pro/clean_a_patio"
    "sim_behavior_r1_pro/freeze_pies"
    "sim_behavior_r1_pro/clearing_food_from_table_into_fridge"
    "sim_behavior_r1_pro/bringing_in_wood"
    "sim_behavior_r1_pro/cleaning_up_plates_and_food"
    "sim_behavior_r1_pro/putting_up_Christmas_decorations_inside"
    "sim_behavior_r1_pro/putting_away_Halloween_decorations"
    "sim_behavior_r1_pro/cook_cabbage"
    "sim_behavior_r1_pro/carrying_in_groceries"
    "sim_behavior_r1_pro/moving_boxes_to_storage"
    "sim_behavior_r1_pro/getting_organized_for_work"
    "sim_behavior_r1_pro/sorting_household_items"
    "sim_behavior_r1_pro/picking_up_toys"
    "sim_behavior_r1_pro/collecting_childrens_toys"
    "sim_behavior_r1_pro/make_pizza"
    "sim_behavior_r1_pro/loading_the_car"
    "sim_behavior_r1_pro/storing_food"
    "sim_behavior_r1_pro/clean_up_your_desk"
    "sim_behavior_r1_pro/canning_food"
    "sim_behavior_r1_pro/boxing_books_up_for_storage"
    "sim_behavior_r1_pro/assembling_gift_baskets"
    "sim_behavior_r1_pro/slicing_vegetables"
)

echo "=========================================="
echo "BEHAVIOR Benchmark Evaluation Script"
echo "=========================================="
echo "Number of episodes per task: $N_EPISODES"
echo "Policy server: $POLICY_CLIENT_HOST:$POLICY_CLIENT_PORT"
echo "Number of parallel envs: $N_ENVS"
echo "Action steps: $N_ACTION_STEPS"
echo "Max episode steps: $MAX_EPISODE_STEPS"
echo "Total tasks: ${#TASKS[@]}"
echo "=========================================="
echo ""

# Check if server is reachable
echo "Checking if policy server is reachable..."
if ! nc -z "$POLICY_CLIENT_HOST" "$POLICY_CLIENT_PORT" 2>/dev/null; then
    echo "WARNING: Cannot connect to policy server at $POLICY_CLIENT_HOST:$POLICY_CLIENT_PORT"
    echo "Make sure the server is running with:"
    echo "  uv run gr00t/eval/run_gr00t_server.py \\"
    echo "    --model-path nvidia/GR00T-N1.6-BEHAVIOR1k \\"
    echo "    --embodiment-tag BEHAVIOR_R1_PRO \\"
    echo "    --use-sim-policy-wrapper"
    echo ""
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Create output directory if specified
if [ -n "$OUTPUT_DIR" ]; then
    mkdir -p "$OUTPUT_DIR"
    RESULTS_FILE="$OUTPUT_DIR/results_$(date +%Y%m%d_%H%M%S).txt"
    echo "Results will be saved to: $RESULTS_FILE"
    echo ""
fi

# Change to project root
cd "$PROJECT_ROOT"

# Track statistics
TOTAL_TASKS=${#TASKS[@]}
COMPLETED_TASKS=0
FAILED_TASKS=()

# Evaluate each task
for i in "${!TASKS[@]}"; do
    task="${TASKS[$i]}"
    task_num=$((i + 1))
    
    echo "=========================================="
    echo "[$task_num/$TOTAL_TASKS] Evaluating: $task"
    echo "=========================================="
    
    # Run evaluation
    if uv run python gr00t/eval/rollout_policy.py \
        --n_episodes "$N_EPISODES" \
        --policy_client_host "$POLICY_CLIENT_HOST" \
        --policy_client_port "$POLICY_CLIENT_PORT" \
        --max_episode_steps="$MAX_EPISODE_STEPS" \
        --env_name "$task" \
        --n_action_steps "$N_ACTION_STEPS" \
        --n_envs "$N_ENVS" 2>&1 | tee -a "${RESULTS_FILE:-/dev/null}"; then
        echo "✓ Successfully completed: $task"
        ((COMPLETED_TASKS++))
    else
        echo "✗ Failed: $task"
        FAILED_TASKS+=("$task")
    fi
    
    echo ""
done

# Print summary
echo "=========================================="
echo "Evaluation Summary"
echo "=========================================="
echo "Total tasks: $TOTAL_TASKS"
echo "Completed: $COMPLETED_TASKS"
echo "Failed: ${#FAILED_TASKS[@]}"
echo ""

if [ ${#FAILED_TASKS[@]} -gt 0 ]; then
    echo "Failed tasks:"
    for task in "${FAILED_TASKS[@]}"; do
        echo "  - $task"
    done
    echo ""
fi

if [ -n "$OUTPUT_DIR" ]; then
    echo "Results saved to: $RESULTS_FILE"
fi

echo "=========================================="

