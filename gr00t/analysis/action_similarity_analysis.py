"""
Action Similarity Analysis for GR00T Model

This script analyzes the similarity between actions generated across different 
inference steps. It creates heatmaps similar to attention k×q correlation matrices.

For each pair of steps (i, j), we compute the average similarity between 
corresponding actions in the action horizon:
    - step_i produces actions: [a_i^1, a_i^2, ..., a_i^H] (H = action_horizon)
    - step_j produces actions: [a_j^1, a_j^2, ..., a_j^H]
    - Similarity(i, j) = mean([sim(a_i^1, a_j^1), sim(a_i^2, a_j^2), ..., sim(a_i^H, a_j^H)])

IMPORTANT: GR00T has two different action horizons:
    - Model's action_horizon (e.g., 50): Internal diffusion model output dimension
    - Effective action_horizon (e.g., 16): Actual actions used, defined by embodiment's delta_indices
    
The model outputs 50 actions, but only the first N (e.g., 16) are actually decoded 
and used for robot control. Use --effective-action-horizon to specify this.

Supported similarity metrics:
- Cosine similarity
- Euclidean distance (converted to similarity)
- Pearson correlation

Usage:
    python gr00t/analysis/action_similarity_analysis.py \
        --action-file /tmp/gr00t_action_logs/episode_0_embedded_actions.npy \
        --output-dir /tmp/gr00t_analysis \
        --metric cosine \
        --effective-action-horizon 16
"""

import argparse
import json
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot_product / (norm_a * norm_b)


def euclidean_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute similarity based on Euclidean distance (1 / (1 + distance))."""
    distance = np.linalg.norm(a - b)
    return 1.0 / (1.0 + distance)


def pearson_correlation(a: np.ndarray, b: np.ndarray) -> float:
    """Compute Pearson correlation coefficient."""
    if np.std(a) == 0 or np.std(b) == 0:
        return 0.0
    corr, _ = stats.pearsonr(a, b)
    return corr


def compute_step_pair_similarity(
    actions_step_i: np.ndarray,
    actions_step_j: np.ndarray,
    metric: Literal["cosine", "euclidean", "pearson"] = "cosine",
) -> float:
    """
    Compute similarity between actions from two different steps.
    
    Args:
        actions_step_i: Actions from step i, shape (action_horizon, action_dim)
        actions_step_j: Actions from step j, shape (action_horizon, action_dim)
        metric: Similarity metric to use
        
    Returns:
        Average similarity between corresponding action positions
    """
    assert actions_step_i.shape == actions_step_j.shape, \
        f"Shape mismatch: {actions_step_i.shape} vs {actions_step_j.shape}"
    
    action_horizon = actions_step_i.shape[0]
    
    # Select similarity function
    if metric == "cosine":
        sim_fn = cosine_similarity
    elif metric == "euclidean":
        sim_fn = euclidean_similarity
    elif metric == "pearson":
        sim_fn = pearson_correlation
    else:
        raise ValueError(f"Unknown metric: {metric}")
    
    # Compute similarity for each position in the action horizon
    similarities = []
    for h in range(action_horizon):
        sim = sim_fn(actions_step_i[h], actions_step_j[h])
        similarities.append(sim)
    
    return np.mean(similarities)


def compute_position_wise_similarity_matrix(
    actions_step_i: np.ndarray,
    actions_step_j: np.ndarray,
    metric: Literal["cosine", "euclidean", "pearson"] = "cosine",
) -> np.ndarray:
    """
    Compute full position-wise similarity matrix between two steps.
    
    This creates a matrix where entry (h1, h2) is the similarity between
    action at position h1 in step i and action at position h2 in step j.
    
    Args:
        actions_step_i: Actions from step i, shape (action_horizon, action_dim)
        actions_step_j: Actions from step j, shape (action_horizon, action_dim)
        metric: Similarity metric to use
        
    Returns:
        Similarity matrix of shape (action_horizon, action_horizon)
    """
    action_horizon = actions_step_i.shape[0]
    
    # Select similarity function
    if metric == "cosine":
        sim_fn = cosine_similarity
    elif metric == "euclidean":
        sim_fn = euclidean_similarity
    elif metric == "pearson":
        sim_fn = pearson_correlation
    else:
        raise ValueError(f"Unknown metric: {metric}")
    
    similarity_matrix = np.zeros((action_horizon, action_horizon))
    for h1 in range(action_horizon):
        for h2 in range(action_horizon):
            similarity_matrix[h1, h2] = sim_fn(actions_step_i[h1], actions_step_j[h2])
    
    return similarity_matrix


def compute_step_similarity_matrix(
    all_actions: np.ndarray,
    metric: Literal["cosine", "euclidean", "pearson"] = "cosine",
) -> np.ndarray:
    """
    Compute pairwise similarity matrix between all steps.
    
    Args:
        all_actions: All actions from episode, shape (num_steps, action_horizon, action_dim)
        metric: Similarity metric to use
        
    Returns:
        Similarity matrix of shape (num_steps, num_steps)
    """
    num_steps = all_actions.shape[0]
    similarity_matrix = np.zeros((num_steps, num_steps))
    
    for i in range(num_steps):
        for j in range(num_steps):
            similarity_matrix[i, j] = compute_step_pair_similarity(
                all_actions[i], all_actions[j], metric
            )
    
    return similarity_matrix


def plot_step_similarity_heatmap(
    similarity_matrix: np.ndarray,
    output_path: Path,
    title: str = "Step-wise Action Similarity",
    metric: str = "cosine",
):
    """
    Plot and save a heatmap of step-wise similarity.
    
    Args:
        similarity_matrix: Similarity matrix of shape (num_steps, num_steps)
        output_path: Path to save the figure
        title: Plot title
        metric: Metric name for labeling
    """
    fig, ax = plt.subplots(figsize=(12, 10))
    
    im = ax.imshow(similarity_matrix, cmap='RdYlBu_r', aspect='auto', vmin=-1, vmax=1)
    
    ax.set_xlabel('Step Index', fontsize=12)
    ax.set_ylabel('Step Index', fontsize=12)
    ax.set_title(f'{title}\n(Metric: {metric})', fontsize=14)
    
    # Add colorbar
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(f'{metric.capitalize()} Similarity', fontsize=10)
    
    # Add grid
    ax.set_xticks(np.arange(0, similarity_matrix.shape[1], max(1, similarity_matrix.shape[1]//10)))
    ax.set_yticks(np.arange(0, similarity_matrix.shape[0], max(1, similarity_matrix.shape[0]//10)))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved step similarity heatmap to {output_path}")


def plot_consecutive_step_similarity(
    all_actions: np.ndarray,
    output_path: Path,
    metric: Literal["cosine", "euclidean", "pearson"] = "cosine",
):
    """
    Plot similarity between consecutive steps and position-wise details.
    
    Creates a subplot for each consecutive step pair showing the 
    position-wise similarity matrix (like K×Q attention).
    """
    num_steps = all_actions.shape[0]
    
    # Compute consecutive step similarities
    consecutive_similarities = []
    for i in range(num_steps - 1):
        sim = compute_step_pair_similarity(all_actions[i], all_actions[i+1], metric)
        consecutive_similarities.append(sim)
    
    # Plot 1: Line plot of consecutive similarities
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(range(len(consecutive_similarities)), consecutive_similarities, 'b-o', markersize=4)
    ax.set_xlabel('Step Index (i → i+1)', fontsize=12)
    ax.set_ylabel(f'{metric.capitalize()} Similarity', fontsize=12)
    ax.set_title(f'Consecutive Step Action Similarity\n(Metric: {metric})', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-1.1, 1.1)
    
    plt.tight_layout()
    line_plot_path = output_path.parent / f"{output_path.stem}_consecutive_line.png"
    plt.savefig(line_plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved consecutive similarity line plot to {line_plot_path}")
    
    # Plot 2: Position-wise similarity matrices for a few step pairs
    num_examples = min(6, num_steps - 1)
    if num_examples > 0:
        step_indices = np.linspace(0, num_steps - 2, num_examples, dtype=int)
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for idx, step_i in enumerate(step_indices):
            if idx >= len(axes):
                break
            pos_sim_matrix = compute_position_wise_similarity_matrix(
                all_actions[step_i], all_actions[step_i + 1], metric
            )
            im = axes[idx].imshow(pos_sim_matrix, cmap='RdYlBu_r', aspect='auto', vmin=-1, vmax=1)
            axes[idx].set_xlabel('Step i+1 Action Position', fontsize=9)
            axes[idx].set_ylabel('Step i Action Position', fontsize=9)
            axes[idx].set_title(f'Step {step_i} → {step_i+1}\nAvg Sim: {consecutive_similarities[step_i]:.3f}', fontsize=10)
        
        # Hide unused subplots
        for idx in range(num_examples, len(axes)):
            axes[idx].axis('off')
        
        # Add shared colorbar
        fig.colorbar(im, ax=axes, fraction=0.02, pad=0.04, label=f'{metric.capitalize()} Similarity')
        
        plt.suptitle(f'Position-wise Action Similarity (K×Q style)\nMetric: {metric}', fontsize=14, y=1.02)
        plt.tight_layout()
        
        pos_sim_path = output_path.parent / f"{output_path.stem}_position_similarity.png"
        plt.savefig(pos_sim_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved position-wise similarity matrices to {pos_sim_path}")


def plot_action_trajectory(
    all_actions: np.ndarray,
    output_path: Path,
    num_dims_to_plot: int = 10,
):
    """
    Plot action values across steps for selected dimensions.
    
    Args:
        all_actions: All actions, shape (num_steps, action_horizon, action_dim)
        output_path: Path to save the figure
        num_dims_to_plot: Number of action dimensions to plot
    """
    num_steps, action_horizon, action_dim = all_actions.shape
    dims_to_plot = min(num_dims_to_plot, action_dim)
    
    # Plot first action in each step's horizon (the immediate next action)
    fig, axes = plt.subplots(dims_to_plot, 1, figsize=(14, dims_to_plot * 2), sharex=True)
    if dims_to_plot == 1:
        axes = [axes]
    
    for dim_idx in range(dims_to_plot):
        # Extract the first predicted action at each step
        first_actions = all_actions[:, 0, dim_idx]
        axes[dim_idx].plot(range(num_steps), first_actions, 'b-', linewidth=1)
        axes[dim_idx].set_ylabel(f'Dim {dim_idx}', fontsize=9)
        axes[dim_idx].grid(True, alpha=0.3)
    
    axes[-1].set_xlabel('Step Index', fontsize=12)
    plt.suptitle('First Action in Horizon Across Steps\n(Immediate next action)', fontsize=14)
    plt.tight_layout()
    
    traj_path = output_path.parent / f"{output_path.stem}_trajectory.png"
    plt.savefig(traj_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved action trajectory plot to {traj_path}")


def analyze_action_overlap(all_actions: np.ndarray, n_action_steps: int = 8):
    """
    Analyze how predicted actions overlap between consecutive steps.
    
    In typical action chunking, only the first n_action_steps are executed,
    then new actions are predicted. This analyzes the overlap.
    
    Args:
        all_actions: All actions, shape (num_steps, action_horizon, action_dim)
        n_action_steps: Number of actions actually executed per step
    """
    num_steps, action_horizon, action_dim = all_actions.shape
    
    print(f"\n=== Action Overlap Analysis ===")
    print(f"Action horizon: {action_horizon}")
    print(f"Actions executed per step: {n_action_steps}")
    print(f"Overlap: actions {n_action_steps}:{action_horizon} should match 0:{action_horizon-n_action_steps} of next step")
    
    if action_horizon <= n_action_steps:
        print("No overlap possible: action_horizon <= n_action_steps")
        return
    
    overlap_size = action_horizon - n_action_steps
    overlap_errors = []
    
    for i in range(num_steps - 1):
        # Actions from step i that were not yet executed
        predicted_future = all_actions[i, n_action_steps:action_horizon]  # (overlap_size, action_dim)
        # Actions from step i+1 that correspond to the same time
        repredicted = all_actions[i + 1, :overlap_size]  # (overlap_size, action_dim)
        
        # Compute MSE between overlapping predictions
        mse = np.mean((predicted_future - repredicted) ** 2)
        overlap_errors.append(mse)
    
    print(f"\nOverlap MSE statistics:")
    print(f"  Mean: {np.mean(overlap_errors):.6f}")
    print(f"  Std:  {np.std(overlap_errors):.6f}")
    print(f"  Min:  {np.min(overlap_errors):.6f}")
    print(f"  Max:  {np.max(overlap_errors):.6f}")
    
    return np.array(overlap_errors)


def main():
    parser = argparse.ArgumentParser(description="Analyze action similarity across steps")
    parser.add_argument(
        "--action-file",
        type=str,
        required=True,
        help="Path to the .npy file containing embedded actions (num_steps, action_horizon, action_dim)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/tmp/gr00t_analysis",
        help="Directory to save analysis outputs",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="cosine",
        choices=["cosine", "euclidean", "pearson"],
        help="Similarity metric to use",
    )
    parser.add_argument(
        "--n-action-steps",
        type=int,
        default=8,
        help="Number of actions executed per step (for overlap analysis)",
    )
    parser.add_argument(
        "--effective-action-horizon",
        type=int,
        default=None,
        help="Effective action horizon used by the embodiment (from delta_indices). "
             "If None, uses full model output. Common values: 16 (libero), 8 (oxe), 30 (unitree_g1)",
    )
    
    args = parser.parse_args()
    
    # Load actions
    action_path = Path(args.action_file)
    if not action_path.exists():
        raise FileNotFoundError(f"Action file not found: {action_path}")
    
    all_actions = np.load(action_path)
    model_action_horizon = all_actions.shape[1]
    model_action_dim = all_actions.shape[2]
    
    print(f"Loaded actions with shape: {all_actions.shape}")
    print(f"  - Number of steps: {all_actions.shape[0]}")
    print(f"  - Model action horizon: {model_action_horizon}")
    print(f"  - Action dimension: {model_action_dim}")
    
    # Try to auto-detect effective action horizon from metadata file
    effective_horizon = args.effective_action_horizon
    metadata_path = action_path.parent / f"{action_path.stem}_metadata.json"
    if effective_horizon is None and metadata_path.exists():
        try:
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
            if "effective_action_horizon" in metadata:
                effective_horizon = metadata["effective_action_horizon"]
                embodiment = metadata.get("embodiment_tag", "unknown")
                print(f"\n*** Auto-detected effective_action_horizon={effective_horizon} from metadata ***")
                print(f"    Embodiment: {embodiment}")
        except Exception as e:
            print(f"Warning: Could not read metadata file: {e}")
    if effective_horizon is not None:
        if effective_horizon > model_action_horizon:
            print(f"Warning: effective_action_horizon ({effective_horizon}) > model horizon ({model_action_horizon})")
            print(f"Using model horizon instead.")
            effective_horizon = model_action_horizon
        else:
            print(f"\n*** Using effective action horizon: {effective_horizon} (out of {model_action_horizon}) ***")
            print(f"    Only the first {effective_horizon} actions are actually used for robot control.")
            all_actions = all_actions[:, :effective_horizon, :]
            print(f"    Sliced actions shape: {all_actions.shape}")
    else:
        effective_horizon = model_action_horizon
        print(f"\nNote: Using full model horizon ({model_action_horizon}). Consider using --effective-action-horizon")
        print(f"      to analyze only the actions actually used by your embodiment.")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Compute step-wise similarity matrix
    print(f"\nComputing step similarity matrix using {args.metric} metric...")
    similarity_matrix = compute_step_similarity_matrix(all_actions, args.metric)
    
    # Save similarity matrix
    sim_matrix_path = output_dir / "step_similarity_matrix.npy"
    np.save(sim_matrix_path, similarity_matrix)
    print(f"Saved similarity matrix to {sim_matrix_path}")
    
    # Plot heatmap
    heatmap_path = output_dir / "step_similarity_heatmap.png"
    plot_step_similarity_heatmap(similarity_matrix, heatmap_path, metric=args.metric)
    
    # Plot consecutive step analysis
    consecutive_path = output_dir / "consecutive_analysis"
    plot_consecutive_step_similarity(all_actions, consecutive_path, args.metric)
    
    # Plot action trajectory
    traj_path = output_dir / "action_analysis"
    plot_action_trajectory(all_actions, traj_path)
    
    # Analyze action overlap
    analyze_action_overlap(all_actions, args.n_action_steps)
    
    # Print summary statistics
    print(f"\n=== Similarity Statistics ===")
    print(f"Metric: {args.metric}")
    print(f"Model action horizon: {model_action_horizon}")
    print(f"Effective action horizon (analyzed): {effective_horizon}")
    
    # Diagonal (self-similarity, should be 1.0)
    diag_sim = np.diag(similarity_matrix)
    print(f"\nSelf-similarity (diagonal):")
    print(f"  Mean: {np.mean(diag_sim):.4f}")
    
    # Off-diagonal (inter-step similarity)
    mask = ~np.eye(similarity_matrix.shape[0], dtype=bool)
    off_diag = similarity_matrix[mask]
    print(f"\nInter-step similarity (off-diagonal):")
    print(f"  Mean: {np.mean(off_diag):.4f}")
    print(f"  Std:  {np.std(off_diag):.4f}")
    print(f"  Min:  {np.min(off_diag):.4f}")
    print(f"  Max:  {np.max(off_diag):.4f}")
    
    # Consecutive step similarity
    consecutive = np.array([similarity_matrix[i, i+1] for i in range(len(similarity_matrix)-1)])
    print(f"\nConsecutive step similarity:")
    print(f"  Mean: {np.mean(consecutive):.4f}")
    print(f"  Std:  {np.std(consecutive):.4f}")
    
    print(f"\n=== Analysis Complete ===")
    print(f"All outputs saved to: {output_dir}")
    
    # Print usage hint
    if args.effective_action_horizon is None:
        print(f"\nHint: To analyze only the actions actually used by your embodiment,")
        print(f"      rerun with --effective-action-horizon <N>")
        print(f"      Common values by embodiment:")
        print(f"        - libero_panda: 16")
        print(f"        - oxe_widowx/oxe_google: 8")
        print(f"        - unitree_g1: 30")
        print(f"        - behavior_r1_pro: 32")


if __name__ == "__main__":
    main()
