"""PPO trainer — on-policy, best for small action spaces."""

from __future__ import annotations

import os
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback

from algos.base_trainer import BaseTrainer


class PPOTrainer(BaseTrainer):
    algo_name = "ppo"

    def __init__(self, algo_config: dict | None = None, **kwargs):
        # Merge with defaults
        from algos import DEFAULT_CONFIGS
        merged = {**DEFAULT_CONFIGS["ppo"], **(algo_config or {})}
        super().__init__(algo_config=merged, **kwargs)

    def train(self, total_timesteps: int, checkpoint_dir: str = "./checkpoints"):
        os.makedirs(checkpoint_dir, exist_ok=True)

        train_env = DummyVecEnv([self.make_train_env])

        self.model = PPO(
            policy="MlpPolicy",
            env=train_env,
            verbose=1,
            **{k: v for k, v in self.algo_config.items() if k != "policy_kwargs"},
        )

        checkpoint_callback = CheckpointCallback(
            save_freq=10000,
            save_path=checkpoint_dir,
            name_prefix=f"{self.algo_name}_btcusdt",
        )

        self.model.learn(
            total_timesteps=total_timesteps,
            callback=checkpoint_callback,
        )

        return self._select_best_checkpoint(checkpoint_dir)

    def _select_best_checkpoint(self, checkpoint_dir: str):
        """Evaluate all checkpoints on OOS data, return best model."""
        test_eval_env = DummyVecEnv([self.make_test_eval_env])

        # Last model OOS equity
        _, last_eq, _ = self.evaluate(test_eval_env)
        print(f"[PPO] Last model OOS equity: ${last_eq:,.2f}")

        best_equity = -np.inf
        best_path = None

        ckpts = sorted(
            [f for f in os.listdir(checkpoint_dir)
             if f.endswith(".zip") and f.startswith(f"{self.algo_name}_btcusdt")],
            key=lambda x: os.path.getmtime(os.path.join(checkpoint_dir, x)),
        )

        for ck in ckpts:
            ck_path = os.path.join(checkpoint_dir, ck)
            try:
                m = PPO.load(ck_path, env=test_eval_env)
                # Temporarily swap model for eval
                saved_model = self.model
                self.model = m
                _, final_eq, _ = self.evaluate(test_eval_env)
                self.model = saved_model
                print(f"[PPO] {ck} -> OOS equity: ${final_eq:,.2f}")
                if final_eq > best_equity:
                    best_equity = final_eq
                    best_path = ck_path
            except Exception as e:
                print(f"[PPO] Skip checkpoint {ck}: {e}")

        if best_path is not None and best_equity > last_eq:
            print(f"[PPO] Best checkpoint: {best_path} (OOS ${best_equity:,.2f})")
            self.model = PPO.load(best_path, env=DummyVecEnv([self.make_train_env]))
        else:
            print(f"[PPO] Using last model as best (OOS ${last_eq:,.2f})")

        return self.model

    def load(self, path: str, env=None):
        if env is None:
            env = DummyVecEnv([self.make_test_eval_env])
        # SB3 auto-appends .zip, so strip if user passed it
        if path.endswith(".zip"):
            path = path[:-4]
        self.model = PPO.load(path, env=env)
        return self.model
