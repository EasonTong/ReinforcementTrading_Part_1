#!/usr/bin/env python
"""
Unified training CLI for modular RL trading system.

Usage:
    python train.py --algo ppo              # Train PPO (default)
    python train.py --algo dqn              # Train DQN
    python train.py --algo qrdqn            # Train QR-DQN
    python train.py --algo all              # Train all three sequentially
    python train.py --algo all --parallel   # Train all three in parallel
    python train.py --algo ppo --timesteps 100000
    python train.py --algo ppo --data-path data/my_data.csv
"""

from __future__ import annotations

import argparse
import glob
import os
import subprocess
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from stable_baselines3.common.vec_env import DummyVecEnv

from algos import create_trainer, ALGO_REGISTRY


def print_metrics(label: str, metrics: dict):
    """Print formatted metrics block."""
    print(f"\n{'=' * 60}")
    print(f"  {label}")
    print(f"{'=' * 60}")
    print(f"  Initial equity:   ${metrics['initial_equity']:,.0f}")
    print(f"  Final equity:     ${metrics['final_equity']:,.0f}")
    print(f"  Total return:     {metrics['total_return_pct']:+.1f}%")
    print(f"  Max drawdown:     {metrics['max_drawdown_pct']:.1f}%")
    print(f"  Sharpe ratio:     {metrics['sharpe']:.2f}")
    print(f"  Sortino ratio:    {metrics['sortino']:.2f}")
    print(f"  Calmar ratio:     {metrics['calmar']:.2f}")
    print(f"  {'─' * 50}")
    print(f"  Total trades:     {metrics['n_trades']}")
    print(f"  Win rate:         {metrics['win_rate_pct']:.1f}%")
    print(f"  Avg win:          ${metrics['avg_win']:,.0f}")
    print(f"  Avg loss:         ${metrics['avg_loss']:,.0f}")
    print(f"  Profit factor:    {metrics['profit_factor']:.2f}")


def plot_equity_curves(algo_name: str, is_curve, oos_curve, is_metrics, oos_metrics):
    """Save IS vs OOS equity chart."""
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=False)

    # Equity
    axes[0].plot(is_curve, label="IS (train)", color="#1f77b4", linewidth=1)
    axes[0].plot(oos_curve, label="OOS (test)", color="#ff7f0e", linewidth=1)
    axes[0].axhline(y=is_curve[0], color="gray", linestyle="--", linewidth=0.8)
    axes[0].set_ylabel("Equity ($)")
    axes[0].set_title(f"{algo_name.upper()} — Equity Curves")
    axes[0].legend(loc="upper left")
    axes[0].grid(True, alpha=0.3)

    # Drawdown
    for label, curve, ax_idx in [("IS", is_curve, 1), ("OOS", oos_curve, 1)]:
        eq = np.array(curve)
        max_eq = np.maximum.accumulate(eq)
        dd = (eq - max_eq) / max_eq * 100
        color = "#1f77b4" if label == "IS" else "#ff7f0e"
        axes[ax_idx].fill_between(range(len(dd)), 0, dd, color=color, alpha=0.4, label=label)
    axes[1].set_ylabel("Drawdown (%)")
    axes[1].set_title("Underwater Plot")
    axes[1].legend(loc="lower left")
    axes[1].grid(True, alpha=0.3)
    axes[1].axhline(y=0, color="gray", linewidth=0.5)

    # Rolling Sharpe (OOS only, 200-bar window)
    oos_eq = np.array(oos_curve)
    oos_returns = np.diff(oos_eq) / oos_eq[:-1]
    window = 200
    if len(oos_returns) > window:
        roll_sharpe = (
            pd.Series(oos_returns).rolling(window).mean()
            / pd.Series(oos_returns).rolling(window).std().replace(0, np.nan)
            * np.sqrt(8760)
        )
        axes[2].plot(roll_sharpe, color="#2ca02c", linewidth=1)
        axes[2].axhline(y=0, color="gray", linestyle="--", linewidth=0.8)
        axes[2].set_ylabel("Sharpe")
        axes[2].set_xlabel("Step")
        axes[2].set_title(f"Rolling Sharpe (OOS, {window}-bar window)")
        axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    fname = f"equity_{algo_name}.png"
    plt.savefig(fname, dpi=150)
    print(f"Chart saved: {fname}")
    plt.close()


def train_one(algo_name: str, total_timesteps: int, data_path: str, seed: int | None = None):
    """Train one algorithm, save best model, return results."""
    print(f"\n{'#' * 60}")
    print(f"#  Training: {algo_name.upper()}")
    print(f"{'#' * 60}")

    if seed is not None:
        import random
        random.seed(seed)
        np.random.seed(seed)

    trainer = create_trainer(algo_name, data_path=data_path)

    # Train
    trainer.train(total_timesteps=total_timesteps)

    # Save best model
    model_path = f"model_{algo_name}_best"
    trainer.save(model_path)

    # Evaluate IS
    is_env = DummyVecEnv([trainer.make_train_eval_env])
    is_curve, _, is_trades = trainer.evaluate(is_env)
    is_metrics = trainer.compute_metrics(is_curve, is_trades)

    # Evaluate OOS
    oos_env = DummyVecEnv([trainer.make_test_eval_env])
    oos_curve, _, oos_trades = trainer.evaluate(oos_env)
    oos_metrics = trainer.compute_metrics(oos_curve, oos_trades)

    # Print
    print_metrics(f"{algo_name.upper()} — IS (In-Sample)", is_metrics)
    print_metrics(f"{algo_name.upper()} — OOS (Out-of-Sample)", oos_metrics)

    # Plot
    plot_equity_curves(algo_name, is_curve, oos_curve, is_metrics, oos_metrics)

    return {
        "algo": algo_name,
        "is": is_metrics,
        "oos": oos_metrics,
        "is_curve": is_curve,
        "oos_curve": oos_curve,
    }


def _find_data_file():
    """Auto-detect the most recent BTCUSDT data file."""
    candidates = sorted(glob.glob("data/BTCUSDT_*.csv"), reverse=True)
    if candidates:
        return candidates[0]
    return None


def _find_venv_python():
    """Find venv Python, fall back to sys.executable."""
    venv_path = os.path.join(os.path.dirname(__file__), ".venv", "Scripts", "python.exe")
    if os.path.exists(venv_path):
        return venv_path
    return sys.executable


def _run_single_algo(algo: str, timesteps: int, data_path: str, seed: int):
    """Entry point for subprocess parallel training."""
    result = train_one(algo, timesteps, data_path, seed)
    return result


def main():
    parser = argparse.ArgumentParser(description="Train RL trading agents")
    parser.add_argument(
        "--algo", type=str, default="ppo",
        choices=["ppo", "dqn", "qrdqn", "all"],
        help="Algorithm to train (default: ppo). Use 'all' for all three.",
    )
    parser.add_argument(
        "--timesteps", type=int, default=600000,
        help="Total training timesteps (default: 600000)",
    )
    parser.add_argument(
        "--data-path", type=str, default=None,
        help="Path to CSV data file (auto-detected if omitted)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--parallel", action="store_true",
        help="Run all algos in parallel (only applies with --algo all)",
    )
    args = parser.parse_args()

    # Resolve data path
    if args.data_path is None:
        args.data_path = _find_data_file()
        if args.data_path is None:
            print("ERROR: No BTCUSDT CSV file found in data/.")
            print("Run: python fetch_binance_data.py")
            return
        print(f"Auto-detected data: {args.data_path}")
    elif not os.path.exists(args.data_path):
        print(f"WARNING: Data file not found: {args.data_path}")

    if args.algo == "all":
        algos_to_train = list(ALGO_REGISTRY.keys())
    else:
        algos_to_train = [args.algo]

    # ── Parallel mode: spawn subprocesses ──
    if args.parallel and len(algos_to_train) > 1:
        print(f"\n{'=' * 60}")
        print(f"  PARALLEL TRAINING: {', '.join(algos_to_train).upper()}")
        print(f"  Timesteps: {args.timesteps:,}  |  Seed: {args.seed}")
        print(f"{'=' * 60}\n")

        python_exe = _find_venv_python()
        processes = {}
        for algo in algos_to_train:
            cmd = [
                python_exe, __file__,
                "--algo", algo,
                "--timesteps", str(args.timesteps),
                "--data-path", args.data_path,
                "--seed", str(args.seed),
            ]
            print(f"[PARALLEL] Launching: {' '.join(cmd)}")
            p = subprocess.Popen(cmd)
            processes[algo] = p

        # Wait for all to complete
        for algo, p in processes.items():
            rc = p.wait()
            status = "OK" if rc == 0 else f"FAIL (exit={rc})"
            print(f"[PARALLEL] {algo.upper()} finished: {status}")

        print("\n[PARALLEL] All processes completed.")
        print("Run evaluate.py to compare models:")
        for algo in algos_to_train:
            print(f"  python evaluate.py --model model_{algo}_best")
        return

    # ── Sequential mode ──
    all_results = []
    for algo in algos_to_train:
        result = train_one(algo, args.timesteps, args.data_path, args.seed)
        all_results.append(result)

    # Summary comparison (if multiple algos)
    if len(all_results) > 1:
        print(f"\n{'=' * 70}")
        print(f"  CROSS-ALGO COMPARISON")
        print(f"{'=' * 70}")
        header = f"{'Algo':<8} {'IS Ret%':>8} {'OOS Ret%':>9} {'OOS Sharpe':>10} {'OOS MaxDD%':>10} {'Trades':>7} {'Win%':>6}"
        print(header)
        print("-" * 70)
        for r in all_results:
            print(
                f"{r['algo']:<8} "
                f"{r['is']['total_return_pct']:>+7.1f}% "
                f"{r['oos']['total_return_pct']:>+8.1f}% "
                f"{r['oos']['sharpe']:>9.2f} "
                f"{r['oos']['max_drawdown_pct']:>9.1f}% "
                f"{r['oos']['n_trades']:>6} "
                f"{r['oos']['win_rate_pct']:>5.1f}%"
            )

    print("\nDone.")


if __name__ == "__main__":
    main()
