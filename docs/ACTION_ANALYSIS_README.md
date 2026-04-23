# GR00T Action Analysis: Understanding States, Actions, and Similarity

This document explains how to use the action logging and similarity analysis tools, and provides a comprehensive explanation of the state-action relationship in the GR00T model.

## Table of Contents
1. [Quick Start](#quick-start)
2. [State-Action Relationship](#state-action-relationship)
3. [Action Generation Pipeline](#action-generation-pipeline)
4. [Similarity Analysis](#similarity-analysis)
5. [Understanding the Outputs](#understanding-the-outputs)

---

## Quick Start

### Step 1: Run the Server with Action Logging

**Terminal 1 - Server with logging:**
```bash
CUDA_VISIBLE_DEVICES=4 uv run python gr00t/eval/run_gr00t_server_with_logging.py \
    --model-path nvidia/GR00T-N1.6-3B \
    --embodiment-tag ROBOCASA_PANDA_OMRON \
    --use-sim-policy-wrapper \
    --action-log-dir /tmp/gr00t_action_logs
```

### Step 2: Run the Client (Single Episode)

**Terminal 2 - Client:**
```bash
CUDA_VISIBLE_DEVICES=4 gr00t/eval/sim/robocasa/robocasa_uv/.venv/bin/python gr00t/eval/rollout_policy.py \
    --n_episodes 1 \
    --policy_client_host 127.0.0.1 \
    --policy_client_port 5555 \
    --max_episode_steps=720 \
    --env_name robocasa_panda_omron/CoffeeServeMug_PandaOmron_Env \
    --n_action_steps 8 \
    --n_envs 1
```

### Step 3: Analyze the Actions

After the episode completes, run the analysis:

```bash
python gr00t/analysis/action_similarity_analysis.py \
    --action-file /tmp/gr00t_action_logs/episode_0_embedded_actions.npy \
    --output-dir /tmp/gr00t_analysis \
    --metric cosine \
    --n-action-steps 8
```

This generates:
- `step_similarity_heatmap.png` - Full step-wise similarity matrix
- `consecutive_analysis_consecutive_line.png` - Similarity between consecutive steps
- `consecutive_analysis_position_similarity.png` - Position-wise K×Q style matrices
- `action_analysis_trajectory.png` - Action values across steps

---

## State-Action Relationship

### What is the State?

The **state** represents the current configuration/pose of the robot. For the ROBOCASA_PANDA_OMRON embodiment, this typically includes:

| State Component | Description | Dimension |
|-----------------|-------------|-----------|
| `arm_joints` | Joint angles of the Panda arm (7 DoF) | 7 |
| `gripper` | Gripper opening state | 1 |
| `mobile_base` | Omron mobile base position/orientation | 3+ |

The state is:
1. **Normalized** using min-max or mean-std normalization
2. **Encoded** through the StateEncoder network
3. Used as conditioning for the diffusion-based action generation

### What is the Action?

The **action** represents the target configuration for the robot at future timesteps. The GR00T model uses **Action Chunking** - predicting multiple future actions at once.

**Key parameters:**
- `action_horizon`: Number of future actions predicted per inference (e.g., 50)
- `n_action_steps`: Number of actions actually executed before re-querying the model (e.g., 8)

### Absolute vs Relative Control

GR00T supports **both absolute and relative control**, configurable per joint group:

#### Absolute Control (ActionRepresentation.ABSOLUTE)
- Actions are **target joint positions/poses**
- The robot moves directly to these positions
- Example: `gripper = 0.5` means set gripper to 50% open

#### Relative Control (ActionRepresentation.RELATIVE)
- Actions are **deltas from current state**
- Requires current state as reference
- Example: `arm_delta = [0.01, 0.02, ...]` means move joints by these amounts

The configuration is defined in `ActionConfig`:
```python
ActionConfig(
    rep=ActionRepresentation.RELATIVE,  # or ABSOLUTE
    type=ActionType.NON_EEF,  # NON_EEF for joint space, EEF for end-effector space
    format=ActionFormat.DEFAULT,
    state_key="arm_joints",  # Which state to use as reference for relative actions
)
```

### For ROBOCASA_PANDA_OMRON

The model configuration is loaded from the checkpoint's `processor_config.json` and `statistics.json`. The action format typically includes:

1. **Joint positions** - Usually absolute or relative depending on config
2. **Gripper command** - Usually absolute (binary open/close or continuous)
3. **Mobile base** - Navigation commands if applicable

The model outputs normalized actions in range [-1, 1], which are then:
1. **Unnormalized** using stored dataset statistics
2. **Converted to absolute** if relative representation was used
3. **Sent to robot controller**

---

## Action Generation Pipeline

### Flow Matching Diffusion

GR00T uses a **Flow Matching Diffusion** policy for action generation:

```
Input: Vision + Language + State → VLM Backbone → Features
                                         ↓
              State Features ← State Encoder ← Current State
                                         ↓
              Noisy Actions ← Sample from N(0,1)
                                         ↓
              For t = 0 to T_inference:  ← Denoising Loop
                  Action Features ← Action Encoder(noisy_actions, t)
                  Combined ← [State Features, Action Features]
                  Velocity ← DiT(Combined, VL_Features)
                  noisy_actions = noisy_actions + dt * velocity
                                         ↓
              Normalized Actions ← Action Decoder(denoised_actions)
                                         ↓
              Physical Actions ← Decode & Unnormalize
```

### What Gets Saved (Embedded Actions)

Our logging captures **Normalized Actions** - the output after the diffusion process but **before** decoding to physical units. This gives you:

- Shape: `(num_steps, action_horizon, action_dim)`
- Range: Approximately [-1, 1] (normalized space)
- Contains: All predicted future actions, not just the executed ones

### Action Chunking and Overlap

```
Step 0: Predict [a0, a1, a2, ..., a49]  ← Execute a0-a7
Step 1: Predict [a0', a1', a2', ..., a49']  ← Execute a0'-a7'

Ideally: a8, a9, ..., a49 from Step 0 ≈ a0', a1', ..., a41' from Step 1
```

The overlap analysis measures how consistent the model's predictions are across steps.

---

## Similarity Analysis

### Step-wise Similarity Matrix

This is analogous to an attention K×Q matrix, but for actions across time:

```
         Step 0  Step 1  Step 2  ...
Step 0   [1.00   0.85    0.72    ...]
Step 1   [0.85   1.00    0.88    ...]
Step 2   [0.72   0.88    1.00    ...]
...
```

**Interpretation:**
- Diagonal = 1.0 (self-similarity)
- High off-diagonal values = similar action patterns at different steps
- Low values indicate distinct actions being predicted

### Position-wise Similarity (K×Q Style)

For two consecutive steps (i, i+1), we create a matrix:

```
         Step i+1 Position
         0    1    2    ...  49
Step i   
Pos 0    [sim  sim  sim ...]
Pos 1    [sim  sim  sim ...]
...
Pos 49   [...]
```

**What to look for:**
- Strong diagonal = predicted actions are shifting correctly
- Uniform pattern = actions are very similar regardless of position
- Random pattern = predictions are inconsistent

### Metrics

| Metric | Best For | Range |
|--------|----------|-------|
| `cosine` | Direction similarity | [-1, 1] |
| `euclidean` | Magnitude + direction | [0, 1] |
| `pearson` | Linear correlation | [-1, 1] |

---

## Understanding the Outputs

### Heatmap Interpretation

**Step Similarity Heatmap:**
- Dark blue band around diagonal = consecutive steps are similar
- Uniform color = repetitive actions (e.g., reaching motion)
- Distinct blocks = different phases of the task

**Consecutive Similarity Plot:**
- Steady high values = smooth, consistent policy
- Sudden drops = task phase transitions
- Oscillations = unstable predictions

### Action Overlap Analysis

The output reports MSE between overlapping predictions:
```
=== Action Overlap Analysis ===
Action horizon: 50
Actions executed per step: 8
Overlap: actions 8:50 should match 0:42 of next step

Overlap MSE statistics:
  Mean: 0.001234
  Std:  0.000567
```

**Low MSE (< 0.01):** Consistent predictions
**High MSE (> 0.1):** Model revises predictions significantly

---

## Example Analysis Code

```python
import numpy as np
from gr00t.analysis import (
    compute_step_similarity_matrix,
    plot_step_similarity_heatmap,
    analyze_action_overlap,
)

# Load saved actions
actions = np.load("/tmp/gr00t_action_logs/episode_0_embedded_actions.npy")
print(f"Loaded actions: {actions.shape}")  # (num_steps, action_horizon, action_dim)

# Compute similarity
similarity_matrix = compute_step_similarity_matrix(actions, metric="cosine")

# Analyze overlap with n_action_steps=8
overlap_mse = analyze_action_overlap(actions, n_action_steps=8)

# Custom visualization
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 8))
plt.imshow(similarity_matrix, cmap='RdYlBu_r', vmin=-1, vmax=1)
plt.colorbar(label='Cosine Similarity')
plt.xlabel('Step Index')
plt.ylabel('Step Index')
plt.title('Action Similarity Across Steps')
plt.savefig('custom_analysis.png')
```

---

## Summary

| Concept | Description |
|---------|-------------|
| **State** | Current robot configuration (joints, gripper, base) |
| **Action** | Target configuration for future timesteps |
| **Action Horizon** | Number of future actions predicted (e.g., 50) |
| **N Action Steps** | Actions executed before re-query (e.g., 8) |
| **Absolute Control** | Actions are target positions |
| **Relative Control** | Actions are deltas from current state |
| **Embedded Action** | Normalized action in model's internal space [-1, 1] |
| **Physical Action** | Unnormalized action in real units (radians, meters) |

The similarity analysis helps understand:
1. How consistent the model's predictions are over time
2. Whether the action chunking overlap is consistent
3. Task phase transitions (where action patterns change)
