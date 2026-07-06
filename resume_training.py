#!/usr/bin/env python
"""
Resume training from checkpoint using modular algo system.

Usage:
    python resume_training.py --algo ppo
    python resume_training.py --algo dqn --additional-timesteps 200000
    python resume_training.py --algo qrdqn --checkpoint checkpoints/ppo_btcusdt_100000_steps.zip
"""

from __future__ import annotations

import argparse
import os
import re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback

from algos import create_trainer, ALGO_REGISTRY


def get_steps(filename: str) -> int:
    """Extract step count from checkpoint filename."""
    m = re.search(r"(\d+)_steps", filename)
    return int(m.group(1)) if m else 0


def plot_equity_curves(algo_name: str, is_curve, oos_curve):
    """Plot IS vs OOS equity."""
    plt.figure(figsize=(12, 6))
    plt.plot(is_curve, label="Train (in-sample) equity")
    plt.plot(oos_curve, label="Test (out-of-sample) equity")
    plt.title(f"Equity Curves: IS vs OOS ({algo_name.upper()}, resumed)")
    plt.xlabel("Steps")
    plt.ylabel("Equity ($)")
    plt.legend()
    plt.tight_layout()
    fname = f"equity_{algo_name}_resumed.png"
    plt.savefig(fname, dpi=150)
    print(f"Chart saved: {fname}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Resume RL training from checkpoint")
    parser.add_argument(
        "--algo", type=str, default="ppo",
        choices=list(ALGO_REGISTRY.keys()),
        help="Algorithm to resume",
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="Specific checkpoint to resume from (auto-find latest if omitted)",
    )
    parser.add_argument(
        "--additional-timesteps", type=int, default=None,
        help="Additional timesteps to train (default: up to 600000 total)",
    )
    parser.add_argument(
        "--data-path", type=str,
        default="data/BTCUSDT_Perpetual_1H_2025-10-01_2026-07-06.csv",
        help="Path to CSV data file",
    )
    parser.add_argument(
        "--target-total", type=int, default=600000,
        help="Target total timesteps (default: 600000)",
    )
    args = parser.parse_args()

    algo_name = args.algo
    ckpt_dir = "./checkpoints"
    os.makedirs(ckpt_dir, exist_ok=True)

    # Find checkpoint
    if args.checkpoint:
        ckpt_path = args.checkpoint
        steps_done = get_steps(os.path.basename(ckpt_path))
    else:
        ckpts = [
            f for f in os.listdir(ckpt_dir)
            if f.endswith(".zip") and f.startswith(f"{algo_name}_btcusdt")
        ]
        if not ckpts:
            print(f"No checkpoints found for {algo_name} in {ckpt_dir}/")
            print("Start fresh training with: python train.py --algo", algo_name)
            return

        latest = max(ckpts, key=get_steps)
        ckpt_path = os.path.join(ckpt_dir, latest)
        steps_done = get_steps(latest)
        print(f"Found latest checkpoint: {latest} ({steps_done} steps)")

    # Calculate remaining steps
    target_total = args.target_total
    if args.additional_timesteps is not None:
        target_total = steps_done + args.additional_timesteps

    remaining = max(0, target_total - steps_done)
    print(f"Resuming {algo_name.upper()} from {steps_done} steps, "
          f"training {remaining} more (target: {target_total})")

    # Create trainer
    trainer = create_trainer(algo_name, data_path=args.data_path)

    if remaining <= 0:
        print("Target already reached. Loading model for evaluation only.")
    else:
        # Load and resume
        trainer.load(ckpt_path, env=DummyVecEnv([trainer.make_train_env]))

        checkpoint_callback = CheckpointCallback(
            save_freq=10000,
            save_path=ckpt_dir,
            name_prefix=f"{algo_name}_btcusdt",
        )

        trainer.model.set_env(DummyVecEnv([trainer.make_train_env]))
        trainer.model.learn(
            total_timesteps=remaining,
            callback=checkpoint_callback,
            reset_num_timesteps=False,
        )

        # Save resumed model
        trainer.save(f"model_{algo_name}_last")

    # Evaluate on OOS to select best
    test_eval_env = DummyVecEnv([trainer.make_test_eval_env])
    _, last_eq, _ = trainer.evaluate(test_eval_env)
    print(f"[{algo_name.upper()}] Last model OOS equity: ${last_eq:,.2f}")

    best_equity = -np.inf
    best_path = None

    all_ckpts = sorted(
        [
            f for f in os.listdir(ckpt_dir)
            if f.endswith(".zip") and f.startswith(f"{algo_name}_btcusdt")
        ],
        key=lambda x: os.path.getmtime(os.path.join(ckpt_dir, x)),
    )

    for ck in all_ckpts:
        ck_fpath = os.path.join(ckpt_dir, ck)
        try:
            trainer.load(ck_fpath, env=test_eval_env)
            _, final_eq, _ = trainer.evaluate(test_eval_env)
            print(f"[{algo_name.upper()}] {ck} -> OOS equity: ${final_eq:,.2f}")
            if final_eq > best_equity:
                best_equity = final_eq
                best_path = ck_fpath
        except Exception as e:
            print(f"[{algo_name.upper()}] Skip checkpoint {ck}: {e}")

    if best_path is None or last_eq >= best_equity:
        print(f"Using last model as best (OOS ${last_eq:,.2f})")
    else:
        print(f"Best checkpoint: {best_path} (OOS ${best_equity:,.2f})")
        trainer.load(best_path, env=DummyVecEnv([trainer.make_train_env]))

    trainer.save(f"model_{algo_name}_best")
    print(f"Best model saved: model_{algo_name}_best.zip")

    # Plot equity curves
    is_env = DummyVecEnv([trainer.make_train_eval_env])
    is_curve, is_eq, _ = trainer.evaluate(is_env)
    oos_env = DummyVecEnv([trainer.make_test_eval_env])
    oos_curve, oos_eq, _ = trainer.evaluate(oos_env)

    print(f"[IS]  Final equity: ${is_eq:,.2f}")
    print(f"[OOS] Final equity: ${oos_eq:,.2f}")

    plot_equity_curves(algo_name, is_curve, oos_curve)

    print("\nDone.")


if __name__ == "__main__":
    main()
