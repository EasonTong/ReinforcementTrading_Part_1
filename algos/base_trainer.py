"""Base trainer: shared data loading, env creation, normalization, evaluation."""

from __future__ import annotations

import os
import numpy as np
import pandas as pd

from stable_baselines3.common.vec_env import DummyVecEnv

from indicators import load_and_preprocess_data
from trading_env import ForexTradingEnv


def _find_data_file():
    """Auto-detect the most recent BTCUSDT data file."""
    import glob
    candidates = sorted(glob.glob("data/BTCUSDT_*.csv"), reverse=True)
    if candidates:
        return candidates[0]
    return None


# ── Shared env defaults ───────────────────────────────────────────
SL_OPTS = [200, 400, 600]     # v2.1: 2% / 4% / 6% risk per trade on $10K
TP_OPTS = [200, 400, 600]     # symmetric → 2×3×3 = 18 OPEN + 2 = 20 actions
WINDOW = 30


def _compute_normalization(train_df, feature_cols):
    """Z-score normalization fitted on training data only."""
    market_mean = train_df[feature_cols].mean().values.astype(np.float32)
    market_std = train_df[feature_cols].std().values.astype(np.float32)
    market_std = np.where(market_std == 0, 1.0, market_std)

    # State features already scaled — identity transform
    state_mean = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    state_std = np.array([1.0, 1.0, 1.0], dtype=np.float32)

    feature_mean = np.concatenate([market_mean, state_mean])
    feature_std = np.concatenate([market_std, state_std])
    return feature_mean, feature_std


class BaseTrainer:
    """Shared RL trainer for ForexTradingEnv."""

    algo_name: str = "base"

    def __init__(
        self,
        data_path: str | None = None,
        sl_opts: list | None = None,
        tp_opts: list | None = None,
        window_size: int = WINDOW,
        algo_config: dict | None = None,
        **env_kwargs,
    ):
        # Auto-detect data file if not specified
        if data_path is None:
            data_path = _find_data_file()
            if data_path is None:
                raise FileNotFoundError(
                    "No BTCUSDT CSV file found in data/. Run: python fetch_binance_data.py"
                )
        self.data_path = data_path
        self.sl_opts = sl_opts or SL_OPTS
        self.tp_opts = tp_opts or TP_OPTS
        self.window_size = window_size
        self.algo_config = algo_config or {}

        # Load data
        self.df, self.feature_cols = load_and_preprocess_data(data_path)
        split_idx = int(len(self.df) * 0.7)
        self.train_df = self.df.iloc[:split_idx].copy()
        self.test_df = self.df.iloc[split_idx:].copy()

        # Normalization
        self.feature_mean, self.feature_std = _compute_normalization(
            self.train_df, self.feature_cols
        )

        self.model = None

        print(f"[{self.algo_name.upper()}] Data loaded: {len(self.df)} bars "
              f"(train={len(self.train_df)}, test={len(self.test_df)})")
        print(f"[{self.algo_name.upper()}] Features: {len(self.feature_cols)} market + 3 state")
        print(f"[{self.algo_name.upper()}] Actions: {2 + len(self.sl_opts) * len(self.tp_opts) * 2} "
              f"(SL={self.sl_opts}, TP={self.tp_opts})")

    def _env_kwargs(self, df):
        """Shared env constructor kwargs."""
        return dict(
            df=df,
            window_size=self.window_size,
            sl_options=self.sl_opts,
            tp_options=self.tp_opts,
            pip_value=1.0,
            spread_pips=0.0,
            commission_pips=80,          # v2.2: 0.08% round-trip taker fee (Binance futures, ~$100K BTC)
            max_slippage_pips=0.0,
            lot_size=1.0,
            feature_columns=self.feature_cols,
            feature_mean=self.feature_mean,
            feature_std=self.feature_std,
            hold_reward_weight=0.005,
            unrealized_delta_weight=0.1,
            open_penalty_pips=2.0,
            time_penalty_pips=0.02,
            risk_per_trade_pct=0.03,     # v2.2: 3% base risk, Kelly adjusts between 1-4%
            max_drawdown_pct=0.5,        # v2.1: terminate at -50% drawdown
            kelly_window=20,             # v2.2: rolling 20-trade window
            kelly_max_pct=0.04,          # v2.2: max 4% after wins
            kelly_min_pct=0.01,          # v2.2: min 1% after losses
        )

    def make_train_env(self):
        """Random-start env for training."""
        return ForexTradingEnv(
            **self._env_kwargs(self.train_df),
            random_start=True,
            min_episode_steps=1000,
            episode_max_steps=2000,
        )

    def make_train_eval_env(self):
        """Deterministic env for IS eval."""
        return ForexTradingEnv(
            **self._env_kwargs(self.train_df),
            random_start=False,
            episode_max_steps=None,
        )

    def make_test_eval_env(self):
        """Deterministic env for OOS eval."""
        return ForexTradingEnv(
            **self._env_kwargs(self.test_df),
            random_start=False,
            episode_max_steps=None,
        )

    def evaluate(self, env, deterministic: bool = True):
        """Run one episode. Returns (equity_curve, final_equity, trades_list)."""
        obs = env.reset()
        equity_curve = []
        trades = []

        while True:
            action, _ = self.model.predict(obs, deterministic=deterministic)
            step_out = env.step(action)

            if len(step_out) == 4:
                obs, rewards, dones, infos = step_out
                done = bool(dones[0])
            else:
                obs, rewards, terminated, truncated, infos = step_out
                done = bool(terminated[0] or truncated[0])

            info = infos[0] if isinstance(infos, (list, tuple)) else infos
            eq = info.get("equity_usd", env.get_attr("equity_usd")[0])
            equity_curve.append(eq)

            trade_info = info.get("last_trade_info", None)
            if isinstance(trade_info, dict) and trade_info.get("event") == "CLOSE":
                trades.append(trade_info)

            if done:
                break

        final_equity = float(equity_curve[-1])
        return equity_curve, final_equity, trades

    def compute_metrics(self, equity_curve, trades):
        """Compute standard metrics from equity curve and trades."""
        eq = np.array(equity_curve)
        returns = np.diff(eq) / eq[:-1] if len(eq) > 1 else np.array([])

        initial_eq = eq[0]
        final_eq = eq[-1]
        total_return = (final_eq / initial_eq - 1) * 100

        max_eq = np.maximum.accumulate(eq)
        drawdown = (eq - max_eq) / max_eq * 100
        max_dd = drawdown.min()

        if len(returns) > 1 and returns.std() > 0:
            sharpe = float(returns.mean() / returns.std() * np.sqrt(8760))
        else:
            sharpe = 0.0

        downside = returns[returns < 0] if len(returns) > 0 else np.array([])
        if len(downside) > 1 and downside.std() > 0:
            sortino = float(returns.mean() / downside.std() * np.sqrt(8760))
        else:
            sortino = 0.0

        calmar = (total_return / 100) / abs(max_dd / 100) if max_dd != 0 else 0

        if trades:
            df_tr = pd.DataFrame(trades)
            win_rate = float((df_tr.net_pips > 0).mean() * 100)
            avg_win = float(df_tr[df_tr.net_pips > 0].net_pips.mean()) if (df_tr.net_pips > 0).any() else 0
            avg_loss = float(df_tr[df_tr.net_pips < 0].net_pips.mean()) if (df_tr.net_pips < 0).any() else 0
            total_wins = float(df_tr[df_tr.net_pips > 0].net_pips.sum())
            total_losses = float(abs(df_tr[df_tr.net_pips < 0].net_pips.sum()))
            profit_factor = total_wins / total_losses if total_losses > 0 else float("inf")
            n_trades = len(df_tr)
        else:
            win_rate = avg_win = avg_loss = profit_factor = n_trades = 0

        return {
            "initial_equity": float(initial_eq),
            "final_equity": float(final_eq),
            "total_return_pct": total_return,
            "max_drawdown_pct": float(max_dd),
            "sharpe": sharpe,
            "sortino": sortino,
            "calmar": calmar,
            "n_trades": n_trades,
            "win_rate_pct": win_rate,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "profit_factor": profit_factor,
        }

    def save(self, path: str):
        """Save model to disk."""
        if self.model is not None:
            self.model.save(path)
            print(f"[{self.algo_name.upper()}] Model saved: {path}")

    def load(self, path: str, env=None):
        """Load model from disk. Must be overridden in subclasses for algo-specific load."""
        raise NotImplementedError

    def train(self, total_timesteps: int, checkpoint_dir: str = "./checkpoints"):
        """Train the model. Must be overridden in subclasses."""
        raise NotImplementedError
