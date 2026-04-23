"""GR00T inference server with action logging for similarity analysis.

Usage:
    uv run python gr00t/eval/run_gr00t_server_with_logging.py \
        --model-path nvidia/GR00T-N1.6-3B \
        --embodiment-tag ROBOCASA_PANDA_OMRON \
        --use-sim-policy-wrapper \
        --action-log-dir /tmp/gr00t_action_logs
"""

from dataclasses import dataclass
import os

from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.policy.gr00t_policy_with_action_logging import (
    Gr00tPolicyWithActionLogging,
    Gr00tSimPolicyWrapperWithLogging,
)
from gr00t.policy.server_client import PolicyServer
import tyro


DEFAULT_MODEL_SERVER_PORT = 5555


@dataclass
class ServerConfigWithLogging:
    """Configuration for running the Groot inference server with action logging."""

    # Gr00t policy configs
    model_path: str = "nvidia/GR00T-N1.6-3B"
    """Path to the model checkpoint directory"""

    embodiment_tag: EmbodimentTag = EmbodimentTag.NEW_EMBODIMENT
    """Embodiment tag"""

    device: str = "cuda"
    """Device to run the model on"""

    # Server configs
    host: str = "127.0.0.1"
    """Host address for the server"""

    port: int = DEFAULT_MODEL_SERVER_PORT
    """Port number for the server"""

    strict: bool = True
    """Whether to enforce strict input and output validation"""

    use_sim_policy_wrapper: bool = False
    """Whether to use the sim policy wrapper"""

    # Action logging configs
    action_log_dir: str = "/tmp/gr00t_action_logs"
    """Directory to save action logs"""


def main(config: ServerConfigWithLogging):
    print("Starting GR00T inference server with action logging...")
    print(f"  Embodiment tag: {config.embodiment_tag}")
    print(f"  Model path: {config.model_path}")
    print(f"  Device: {config.device}")
    print(f"  Host: {config.host}")
    print(f"  Port: {config.port}")
    print(f"  Action log dir: {config.action_log_dir}")

    # Check if the model path exists (for local paths)
    if config.model_path.startswith("/") and not os.path.exists(config.model_path):
        raise FileNotFoundError(f"Model path {config.model_path} does not exist")

    # Create policy with action logging
    policy = Gr00tPolicyWithActionLogging(
        embodiment_tag=config.embodiment_tag,
        model_path=config.model_path,
        device=config.device,
        strict=config.strict,
        save_dir=config.action_log_dir,
    )

    # Apply sim policy wrapper if needed
    if config.use_sim_policy_wrapper:
        policy = Gr00tSimPolicyWrapperWithLogging(policy)

    # Create custom endpoint to save logs on demand
    def save_logs_handler():
        if hasattr(policy, 'save_action_logs'):
            path = policy.save_action_logs()
            return {"status": "ok", "path": str(path) if path else None}
        elif hasattr(policy, 'logging_policy'):
            path = policy.logging_policy.save_action_logs()
            return {"status": "ok", "path": str(path) if path else None}
        return {"status": "error", "message": "No logging policy found"}

    def get_action_history_handler():
        if hasattr(policy, 'get_action_history'):
            history = policy.get_action_history()
        elif hasattr(policy, 'logging_policy'):
            history = policy.logging_policy.get_action_history()
        else:
            return {"status": "error", "message": "No logging policy found"}
        return {
            "status": "ok",
            "action_history": history.tolist() if len(history) > 0 else [],
            "shape": list(history.shape) if len(history) > 0 else [],
        }

    server = PolicyServer(
        policy=policy,
        host=config.host,
        port=config.port,
    )
    
    # Register custom endpoints
    server.register_endpoint("save_action_logs", save_logs_handler, requires_input=False)
    server.register_endpoint("get_action_history", get_action_history_handler, requires_input=False)

    try:
        server.run()
    except KeyboardInterrupt:
        print("\nShutting down server...")
        # Save final action logs
        if hasattr(policy, 'save_action_logs'):
            policy.save_action_logs("final_episode")
        elif hasattr(policy, 'logging_policy'):
            policy.logging_policy.save_action_logs("final_episode")


if __name__ == "__main__":
    config = tyro.cli(ServerConfigWithLogging)
    main(config)
