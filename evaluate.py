#!/usr/bin/env python
"""
Unified evaluation CLI — load any trained model, evaluate IS+OOS, export results.

Usage:
    python evaluate.py --model model_ppo_best.zip
    python evaluate.py --model model_dqn_best.zip --data-path data/my_data.csv
"""

from __future__ import annotations

import argparse
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from stable_baselines3.common.vec_env import DummyVecEnv

from algos import create_trainer, ALGO_REGISTRY


def detect_algo(model_path: str) -> str:
    """Auto-detect algorithm from model filename. Falls back to 'ppo'."""
    basename = os.path.splitext(os.path.basename(model_path))[0]
    for algo in ALGO_REGISTRY:
        if algo in basename:
            return algo
    # Legacy models (eurusd prefix) are all PPO
    if "eurusd" in basename:
        print(f"Note: '{model_path}' looks like a legacy PPO model, using 'ppo'")
        return "ppo"
    # Default fallback
    print(f"WARNING: Cannot detect algo from '{model_path}'. Defaulting to 'ppo'.")
    return "ppo"


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


def plot_full_report(algo_name: str, is_curve, oos_curve, is_metrics, oos_metrics):
    """Generate comprehensive 3-panel report chart."""
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))

    # Panel 1: Equity curve
    axes[0].plot(is_curve, label="IS (train)", color="#1f77b4", linewidth=1)
    axes[0].plot(oos_curve, label="OOS (test)", color="#ff7f0e", linewidth=1)
    axes[0].axhline(y=is_curve[0], color="gray", linestyle="--", linewidth=0.8,
                    label=f"Initial ${is_curve[0]:,.0f}")
    axes[0].set_ylabel("Equity ($)")
    axes[0].set_title(f"{algo_name.upper()} — Equity Curves")
    axes[0].legend(loc="upper left")
    axes[0].grid(True, alpha=0.3)

    # Panel 2: Drawdown
    for label, curve in [("IS", is_curve), ("OOS", oos_curve)]:
        eq = np.array(curve)
        max_eq = np.maximum.accumulate(eq)
        dd = (eq - max_eq) / max_eq * 100
        color = "#1f77b4" if label == "IS" else "#ff7f0e"
        axes[1].fill_between(range(len(dd)), 0, dd, color=color, alpha=0.4, label=label)
    axes[1].set_ylabel("Drawdown (%)")
    axes[1].set_title("Underwater Plot")
    axes[1].legend(loc="lower left")
    axes[1].grid(True, alpha=0.3)
    axes[1].axhline(y=0, color="gray", linewidth=0.5)

    # Panel 3: Rolling Sharpe (OOS, 200-bar)
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
    fname = f"eval_{algo_name}.png"
    plt.savefig(fname, dpi=150)
    print(f"Chart saved: {fname}")
    plt.close()


def _find_data_file():
    """Auto-detect the most recent BTCUSDT data file."""
    import glob
    candidates = sorted(glob.glob("data/BTCUSDT_*.csv"), reverse=True)
    if candidates:
        return candidates[0]
    return None


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained RL trading model")
    parser.add_argument(
        "--model", type=str, required=True,
        help="Path to model zip file",
    )
    parser.add_argument(
        "--data-path", type=str, default=None,
        help="Path to CSV data file (auto-detected if omitted)",
    )
    args = parser.parse_args()

    # Resolve data path
    if args.data_path is None:
        args.data_path = _find_data_file()
        if args.data_path is None:
            print("ERROR: No BTCUSDT CSV file found in data/.")
            print("Run: python fetch_binance_data.py")
            return

    # Auto-detect algo
    algo_name = detect_algo(args.model)
    print(f"Detected algorithm: {algo_name}")

    # Create trainer (just for env setup)
    trainer = create_trainer(algo_name, data_path=args.data_path)

    # Load model using trainer's load method
    trainer.load(args.model)
    print(f"Model loaded: {args.model}")

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
    plot_full_report(algo_name, is_curve, oos_curve, is_metrics, oos_metrics)

    # Save OOS trades to CSV
    if oos_trades:
        trades_df = pd.DataFrame(oos_trades)
        out_csv = f"trades_{algo_name}.csv"
        trades_df.to_csv(out_csv, index=False)
        print(f"OOS trades saved: {out_csv} ({len(oos_trades)} trades)")

    print("\nDone.")


if __name__ == "__main__":
    main()
