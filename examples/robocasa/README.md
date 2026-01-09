# RoboCasa Evaluation Benchmark

[RoboCasa](https://robocasa.ai/) is a large-scale simulation framework for training generally capable robots to perform everyday tasks, featuring realistic kitchen environments with over 2,500 3D assets and 100 diverse manipulation tasks. This evaluation benchmark uses RoboCasa with the Panda robot equipped with an Omron gripper to test household manipulation tasks including operating kitchen appliances, pick-and-place operations, and interacting with doors, drawers, and various objects.

---

# RoboCasa evaluation benchmark result
Checkpoint: [nvidia/GR00T-N1.6-3B](https://huggingface.co/nvidia/GR00T-N1.6-3B)

| Task | Success rate |
| ---- | ------------ |
| `robocasa_panda_omron/CoffeeSetupMug_PandaOmron_Env` | 31.0% |
| `robocasa_panda_omron/CoffeeServeMug_PandaOmron_Env` | 63.5% |
| `robocasa_panda_omron/CoffeePressButton_PandaOmron_Env` | 98.5% |
| `robocasa_panda_omron/OpenSingleDoor_PandaOmron_Env` | 81.5% |
| `robocasa_panda_omron/OpenDoubleDoor_PandaOmron_Env` | 39.0% |
| `robocasa_panda_omron/CloseSingleDoor_PandaOmron_Env` | 96.0% |
| `robocasa_panda_omron/CloseDoubleDoor_PandaOmron_Env` | 88.5% |
| `robocasa_panda_omron/OpenDrawer_PandaOmron_Env` | 81.1% |
| `robocasa_panda_omron/CloseDrawer_PandaOmron_Env` | 100.0% |
| `robocasa_panda_omron/TurnOnMicrowave_PandaOmron_Env` | 91.5% |
| `robocasa_panda_omron/TurnOffMicrowave_PandaOmron_Env` | 96.0% |
| `robocasa_panda_omron/PnPCounterToCab_PandaOmron_Env` | 47.5% |
| `robocasa_panda_omron/PnPCabToCounter_PandaOmron_Env` | 41.0% |
| `robocasa_panda_omron/PnPCounterToSink_PandaOmron_Env` | 46.0% |
| `robocasa_panda_omron/PnPSinkToCounter_PandaOmron_Env` | 50.0% |
| `robocasa_panda_omron/PnPCounterToMicrowave_PandaOmron_Env` | 19.0% |
| `robocasa_panda_omron/PnPMicrowaveToCounter_PandaOmron_Env` | 24.5% |
| `robocasa_panda_omron/PnPCounterToStove_PandaOmron_Env` | 63.2% |
| `robocasa_panda_omron/PnPStoveToCounter_PandaOmron_Env` | 54.5% |
| `robocasa_panda_omron/TurnOnSinkFaucet_PandaOmron_Env` | 89.0% |
| `robocasa_panda_omron/TurnOffSinkFaucet_PandaOmron_Env` | 93.5% |
| `robocasa_panda_omron/TurnSinkSpout_PandaOmron_Env` | 87.0% |
| `robocasa_panda_omron/TurnOnStove_PandaOmron_Env` | 76.5% |
| `robocasa_panda_omron/TurnOffStove_PandaOmron_Env` | 31.0% |
| **Average** | 66.22% |

# Evaluate checkpoint

## Setup

First, setup the evaluation simulation environment. This only needs to run once for each simulation benchmark. After it's done, we only need to launch server and client.

```bash
sudo apt update
sudo apt install libegl1-mesa-dev libglu1-mesa
bash gr00t/eval/sim/robocasa/setup_RoboCasa.sh
```

## Evaluate All Tasks

To reproduce the evaluation results and get the average success rate, run the evaluation script that evaluates all 24 tasks:

**Terminal 1 - Server:**
```bash
uv run python gr00t/eval/run_gr00t_server.py \
    --model-path nvidia/GR00T-N1.6-3B \
    --embodiment-tag ROBOCASA_PANDA_OMRON \
    --use-sim-policy-wrapper
```

**Terminal 2 - Client (run from project root):**
```bash
cd examples/robocasa
./evaluate_all_tasks.sh --output_dir ./results
```

The script will:
- Evaluate all 24 RoboCasa tasks sequentially
- Save results to a timestamped file in the output directory
- Display progress and summary

After evaluation completes, calculate the average success rate:
```bash
./calc_average_success.sh ./results/results_*.txt
```

Or specify the results file directly:
```bash
./calc_average_success.sh ./results/results_YYYYMMDD_HHMMSS.txt
```

## Evaluate Single Task

To evaluate a single task, run client server evaluation under the project root directory in separate terminals:

**Terminal 1 - Server:**
```bash
uv run python gr00t/eval/run_gr00t_server.py \
    --model-path nvidia/GR00T-N1.6-3B \
    --embodiment-tag ROBOCASA_PANDA_OMRON \
    --use-sim-policy-wrapper
```

**Terminal 2 - Client:**
```bash
gr00t/eval/sim/robocasa/robocasa_uv/.venv/bin/python gr00t/eval/rollout_policy.py \
    --n_episodes 10 \
    --policy_client_host 127.0.0.1 \
    --policy_client_port 5555 \
    --max_episode_steps=720 \
    --env_name robocasa_panda_omron/OpenDrawer_PandaOmron_Env \
    --n_action_steps 8 \
    --n_envs 5
```

## Script Options

The `evaluate_all_tasks.sh` script supports the following options:
- `--n_episodes N`: Number of episodes per task (default: 10)
- `--policy_client_host HOST`: Policy server host (default: 127.0.0.1)
- `--policy_client_port PORT`: Policy server port (default: 5555)
- `--n_envs N`: Number of parallel environments (default: 5)
- `--n_action_steps N`: Number of action steps (default: 8)
- `--max_episode_steps N`: Maximum episode steps (default: 720)
- `--output_dir DIR`: Directory to save results (default: no saving)
