"""Algorithm registry — add new algos here."""

from algos.ppo_trainer import PPOTrainer
from algos.dqn_trainer import DQNTrainer
from algos.qrdqn_trainer import QRDQNTrainer

ALGO_REGISTRY = {
    "ppo": PPOTrainer,
    "dqn": DQNTrainer,
    "qrdqn": QRDQNTrainer,
}

DEFAULT_CONFIGS = {
    "ppo": {
        "learning_rate": 1e-4,
        "n_steps": 2048,
        "batch_size": 64,
        "n_epochs": 10,
        "ent_coef": 0.01,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_range": 0.2,
        "policy_kwargs": None,
    },
    "dqn": {
        "learning_rate": 1e-4,
        "buffer_size": 100000,
        "exploration_fraction": 0.3,
        "exploration_final_eps": 0.02,
        "target_update_interval": 1000,
        "batch_size": 64,
        "learning_starts": 5000,
        "gamma": 0.99,
        "policy_kwargs": None,
    },
    "qrdqn": {
        "learning_rate": 1e-4,
        "buffer_size": 100000,
        "exploration_fraction": 0.3,
        "target_update_interval": 1000,
        "batch_size": 64,
        "learning_starts": 5000,
        "gamma": 0.99,
        "policy_kwargs": None,
    },
}


def create_trainer(algo_name: str, **kwargs):
    """Factory: create a trainer by name. Passes all kwargs to trainer __init__."""
    if algo_name not in ALGO_REGISTRY:
        raise ValueError(
            f"Unknown algo '{algo_name}'. Available: {list(ALGO_REGISTRY.keys())}"
        )
    return ALGO_REGISTRY[algo_name](**kwargs)


__all__ = ["ALGO_REGISTRY", "DEFAULT_CONFIGS", "create_trainer"]
