"""Gr00t Policy with Action Logging for analysis.

This module provides a wrapper that saves embedded actions (before decoding)
for analyzing action similarity across steps.
"""

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch

from gr00t.policy.gr00t_policy import Gr00tPolicy, Gr00tSimPolicyWrapper


class Gr00tPolicyWithActionLogging(Gr00tPolicy):
    """Gr00t Policy that logs embedded actions before decoding.
    
    This policy saves the raw action predictions from the model (in the normalized/embedded
    space) before they are decoded back to physical units. This allows for analysis of
    action similarity across different inference steps.
    """

    def __init__(
        self,
        *args,
        save_dir: str = "/tmp/gr00t_action_logs",
        **kwargs,
    ):
        """Initialize the policy with action logging.
        
        Args:
            save_dir: Directory to save action logs
            *args, **kwargs: Arguments passed to Gr00tPolicy
        """
        super().__init__(*args, **kwargs)
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Storage for embedded actions
        self.embedded_actions_history = []  # List of numpy arrays (action_horizon, action_dim)
        self.step_count = 0
        self.episode_id = 0
        
        print(f"Action logging enabled. Saving to: {self.save_dir}")

    def _get_action(
        self, observation: dict[str, Any], options: dict[str, Any] | None = None
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Internal method to compute actions from observations, with logging.

        This overrides the parent method to capture and save embedded actions
        before they are decoded.
        """
        from gr00t.data.types import MessageType, VLAStepData
        
        def _rec_to_dtype(x: Any, dtype: torch.dtype) -> Any:
            if isinstance(x, torch.Tensor) and torch.is_floating_point(x):
                return x.to(dtype=dtype)
            elif isinstance(x, dict) or hasattr(x, "items"):
                return {k: _rec_to_dtype(v, dtype) for k, v in x.items()}
            elif isinstance(x, list):
                return [_rec_to_dtype(v, dtype) for v in x]
            else:
                return x

        # Step 1: Split batched observation into individual observations
        unbatched_observations = self._unbatch_observation(observation)
        processed_inputs = []

        # Step 2: Process each observation through the VLA processor
        states = []
        for obs in unbatched_observations:
            vla_step_data = self._to_vla_step_data(obs)
            states.append(vla_step_data.states)
            messages = [{"type": MessageType.EPISODE_STEP.value, "content": vla_step_data}]
            processed_inputs.append(self.processor(messages))

        # Step 3: Collate processed inputs into a single batch for model
        collated_inputs = self.collate_fn(processed_inputs)
        collated_inputs = _rec_to_dtype(collated_inputs, dtype=torch.bfloat16)

        # Step 4: Run model inference to predict actions
        with torch.inference_mode():
            model_pred = self.model.get_action(**collated_inputs)
        normalized_action = model_pred["action_pred"].float()

        # ===== ACTION LOGGING: Save embedded actions before decoding =====
        # normalized_action shape: [B, action_horizon, action_dim]
        embedded_action_np = normalized_action.cpu().numpy()
        
        # Save for each batch item
        for batch_idx in range(embedded_action_np.shape[0]):
            action_data = {
                "step": self.step_count,
                "batch_idx": batch_idx,
                "episode_id": self.episode_id,
                "embedded_action": embedded_action_np[batch_idx].tolist(),  # (action_horizon, action_dim)
                "action_shape": list(embedded_action_np[batch_idx].shape),
            }
            self.embedded_actions_history.append(embedded_action_np[batch_idx])
        
        self.step_count += 1
        # ===== END ACTION LOGGING =====

        # Step 5: Decode actions from normalized space back to physical units
        batched_states = {}
        for k in self.modality_configs["state"].modality_keys:
            batched_states[k] = np.stack([s[k] for s in states], axis=0)
        unnormalized_action = self.processor.decode_action(
            normalized_action.cpu().numpy(), self.embodiment_tag, batched_states
        )

        # Cast all actions to float32 for consistency
        casted_action = {
            key: value.astype(np.float32) for key, value in unnormalized_action.items()
        }
        
        # Get sparsity stats if Focus is enabled
        info = {}
        if hasattr(self, '_focus_enabled') and self._focus_enabled:
            info["sparsity"] = self._get_focus_sparsity_stats()
        
        return casted_action, info

    def save_action_logs(self, filename: str | None = None):
        """Save all logged embedded actions to disk.
        
        Args:
            filename: Optional custom filename (without extension)
        """
        if not self.embedded_actions_history:
            print("No actions logged yet")
            return None
            
        if filename is None:
            filename = f"episode_{self.episode_id}_embedded_actions"
        
        # Convert to numpy array: (num_steps, action_horizon, action_dim)
        actions_array = np.stack(self.embedded_actions_history, axis=0)
        
        # Save as numpy file
        save_path = self.save_dir / f"{filename}.npy"
        np.save(save_path, actions_array)
        
        # Get effective action horizon from embodiment's delta_indices
        effective_action_horizon = len(self.modality_configs["action"].delta_indices)
        
        # Also save metadata as JSON
        metadata = {
            "episode_id": self.episode_id,
            "num_steps": len(self.embedded_actions_history),
            "action_shape": list(actions_array.shape),
            "model_action_horizon": actions_array.shape[1] if len(actions_array.shape) > 1 else None,
            "effective_action_horizon": effective_action_horizon,
            "action_dim": actions_array.shape[2] if len(actions_array.shape) > 2 else None,
            "embodiment_tag": self.embodiment_tag.value,
            "note": f"Model outputs {actions_array.shape[1]} actions, but only first {effective_action_horizon} are used",
        }
        metadata_path = self.save_dir / f"{filename}_metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Saved {len(self.embedded_actions_history)} action steps to {save_path}")
        print(f"Action array shape: {actions_array.shape}")
        
        return save_path

    def reset(self, options: dict[str, Any] | None = None) -> dict[str, Any]:
        """Reset the policy and optionally save logs from previous episode."""
        # Save current episode actions if any
        if self.embedded_actions_history:
            self.save_action_logs()
        
        # Clear history for new episode
        self.embedded_actions_history = []
        self.step_count = 0
        self.episode_id += 1
        
        return super().reset(options)

    def get_action_history(self) -> np.ndarray:
        """Get the current episode's action history as numpy array.
        
        Returns:
            numpy array of shape (num_steps, action_horizon, action_dim)
        """
        if not self.embedded_actions_history:
            return np.array([])
        return np.stack(self.embedded_actions_history, axis=0)


class Gr00tSimPolicyWrapperWithLogging(Gr00tSimPolicyWrapper):
    """Wrapper for Gr00tPolicyWithActionLogging for simulation environments."""

    def __init__(self, policy: Gr00tPolicyWithActionLogging, *, strict: bool = True):
        super().__init__(policy, strict=strict)
        self.logging_policy = policy

    def save_action_logs(self, filename: str | None = None):
        """Delegate to underlying policy."""
        return self.logging_policy.save_action_logs(filename)

    def get_action_history(self) -> np.ndarray:
        """Delegate to underlying policy."""
        return self.logging_policy.get_action_history()

    def reset(self, options: dict[str, Any] | None = None) -> dict[str, Any]:
        """Reset both wrapper and underlying policy."""
        return self.logging_policy.reset(options)
