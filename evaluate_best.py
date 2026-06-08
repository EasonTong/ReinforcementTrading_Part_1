"""Comprehensive evaluation of best model (10K checkpoint) with metrics & charts."""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from indicators import load_and_preprocess_data
from trading_env import ForexTradingEnv

# ── Load data ──────────────────────────────────────────────────────
df, fc = load_and_preprocess_data('data/BTCUSDT_Perpetual_1H_2025-10-01_2026-06-08.csv')
split_idx = int(len(df) * 0.7)
train_df = df.iloc[:split_idx].copy()
test_df  = df.iloc[split_idx:].copy()

# Normalization (fit on train)
market_mean = train_df[fc].mean().values.astype(np.float32)
market_std  = train_df[fc].std().values.astype(np.float32)
market_std  = np.where(market_std == 0, 1.0, market_std)
state_mean = np.array([0.0, 0.0, 0.0], dtype=np.float32)
state_std  = np.array([1.0, 1.0, 1.0], dtype=np.float32)
feat_mean = np.concatenate([market_mean, state_mean])
feat_std  = np.concatenate([market_std,  state_std])

SL = [200, 500, 1000, 2000]
TP = [200, 500, 1000, 2000]

def make_env(df):
    return ForexTradingEnv(df=df, window_size=30, sl_options=SL, tp_options=TP,
        pip_value=1.0, spread_pips=0.0, commission_pips=0.0, max_slippage_pips=0.0,
        lot_size=1.0, random_start=False, episode_max_steps=None,
        feature_columns=fc, feature_mean=feat_mean, feature_std=feat_std,
        hold_reward_weight=0.005, open_penalty_pips=2.0,
        time_penalty_pips=0.02, unrealized_delta_weight=0.1)

# ── Load model ─────────────────────────────────────────────────────
model = PPO.load("model_eurusd_best")
print("Model loaded: model_eurusd_best.zip")

# ── Evaluate IS & OOS ─────────────────────────────────────────────
for label, data_df in [("IS (训练集)", train_df), ("OOS (测试集)", test_df)]:
    env = DummyVecEnv([lambda df=data_df: make_env(df)])
    obs = env.reset()
    equity_curve = []
    trades = []

    while True:
        action, _ = model.predict(obs, deterministic=True)
        step_out = env.step(action)
        if len(step_out) == 4:
            obs, rewards, dones, infos = step_out
            done = bool(dones[0])
        else:
            obs, rewards, terminated, truncated, infos = step_out
            done = bool(terminated[0] or truncated[0])

        # Capture equity from info BEFORE env auto-resets (DummyVecEnv resets on done)
        info = infos[0] if isinstance(infos, (list, tuple)) else infos
        eq = info.get("equity_usd", env.get_attr("equity_usd")[0])
        equity_curve.append(eq)

        trade_info = info.get("last_trade_info", None)
        if isinstance(trade_info, dict) and trade_info.get("event") == "CLOSE":
            trades.append(trade_info)
        if done:
            break

    eq = np.array(equity_curve)
    returns = np.diff(eq) / eq[:-1]  # step returns

    # ── Metrics ──────────────────────────────────────────────────
    initial_eq = eq[0]
    final_eq = eq[-1]
    total_return = (final_eq / initial_eq - 1) * 100
    max_eq = np.maximum.accumulate(eq)
    drawdown = (eq - max_eq) / max_eq * 100
    max_dd = drawdown.min()
    max_dd_idx = np.argmin(drawdown)

    # Sharpe ratio (annualized, assuming 24*365 = 8760 trading hours/year)
    if len(returns) > 1 and returns.std() > 0:
        sharpe = returns.mean() / returns.std() * np.sqrt(8760)
    else:
        sharpe = 0.0

    # Sortino ratio (downside deviation only)
    downside = returns[returns < 0]
    if len(downside) > 1 and downside.std() > 0:
        sortino = returns.mean() / downside.std() * np.sqrt(8760)
    else:
        sortino = 0.0

    # Calmar ratio
    calmar = (total_return / 100) / abs(max_dd / 100) if max_dd != 0 else 0

    # Trade stats
    if trades:
        df_tr = pd.DataFrame(trades)
        win_rate = (df_tr.net_pips > 0).mean() * 100
        avg_win = df_tr[df_tr.net_pips > 0].net_pips.mean() if (df_tr.net_pips > 0).any() else 0
        avg_loss = df_tr[df_tr.net_pips < 0].net_pips.mean() if (df_tr.net_pips < 0).any() else 0
        total_wins = df_tr[df_tr.net_pips > 0].net_pips.sum()
        total_losses = abs(df_tr[df_tr.net_pips < 0].net_pips.sum())
        profit_factor = total_wins / total_losses if total_losses > 0 else float('inf')
        avg_time = df_tr.time_in_trade.mean()
        n_trades = len(df_tr)
    else:
        win_rate = avg_win = avg_loss = profit_factor = avg_time = n_trades = 0

    print(f"\n{'='*60}")
    print(f"  {label} — Best Model (10K checkpoint)")
    print(f"{'='*60}")
    print(f"  Initial equity:   ${initial_eq:,.0f}")
    print(f"  Final equity:     ${final_eq:,.0f}")
    print(f"  Total return:     {total_return:+.1f}%")
    print(f"  Max drawdown:     {max_dd:.1f}%")
    print(f"  Sharpe ratio:     {sharpe:.2f}")
    print(f"  Sortino ratio:    {sortino:.2f}")
    print(f"  Calmar ratio:     {calmar:.2f}")
    print(f"  ─────────────────────────────────")
    print(f"  Total trades:     {n_trades}")
    print(f"  Win rate:         {win_rate:.1f}%")
    print(f"  Avg win:          ${avg_win:,.0f}")
    print(f"  Avg loss:         ${avg_loss:,.0f}")
    print(f"  Profit factor:    {profit_factor:.2f}")
    print(f"  Avg time/trade:   {avg_time:.0f} bars")

    # ── Charts ───────────────────────────────────────────────────
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    # Equity curve
    axes[0].plot(eq, color='#1f77b4', linewidth=1)
    axes[0].fill_between(range(len(eq)), initial_eq, eq, where=(eq >= initial_eq),
                         color='#1f77b4', alpha=0.15)
    axes[0].fill_between(range(len(eq)), initial_eq, eq, where=(eq < initial_eq),
                         color='#d62728', alpha=0.15)
    axes[0].axhline(y=initial_eq, color='gray', linestyle='--', linewidth=0.8)
    axes[0].set_ylabel('Equity ($)')
    axes[0].set_title(f'{label} — Equity Curve')
    axes[0].legend(['Equity', 'Initial $10,000'], loc='upper left')
    axes[0].grid(True, alpha=0.3)

    # Drawdown (underwater)
    axes[1].fill_between(range(len(drawdown)), 0, drawdown, color='#d62728', alpha=0.6)
    axes[1].set_ylabel('Drawdown (%)')
    axes[1].set_title('Underwater Plot (Drawdown)')
    axes[1].grid(True, alpha=0.3)
    axes[1].axhline(y=0, color='gray', linewidth=0.5)

    # Rolling Sharpe (500-bar window)
    window = 500
    if len(returns) > window:
        roll_returns = pd.Series(returns)
        roll_mean = roll_returns.rolling(window).mean()
        roll_std  = roll_returns.rolling(window).std()
        roll_sharpe = roll_mean / roll_std.replace(0, np.nan) * np.sqrt(8760)
        axes[2].plot(roll_sharpe, color='#2ca02c', linewidth=1)
        axes[2].axhline(y=0, color='gray', linestyle='--', linewidth=0.8)
        axes[2].set_ylabel('Sharpe')
        axes[2].set_xlabel('Step')
        axes[2].set_title(f'Rolling Sharpe (500-bar window)')
        axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    fname = f"eval_{'is' if 'IS' in label else 'oos'}.png"
    plt.savefig(fname, dpi=150)
    print(f"  Chart saved: {fname}")
    plt.close()

print("\nDone.")
