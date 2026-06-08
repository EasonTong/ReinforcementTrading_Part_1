import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from indicators import load_and_preprocess_data
from trading_env import ForexTradingEnv


def run_one_episode(model, vec_env, deterministic=True):
    obs = vec_env.reset()
    equity_curve = []
    closed_trades = []

    while True:
        action, _ = model.predict(obs, deterministic=deterministic)
        step_out = vec_env.step(action)

        if len(step_out) == 4:
            obs, rewards, dones, infos = step_out
            done = bool(dones[0])
        else:
            obs, rewards, terminated, truncated, infos = step_out
            done = bool(terminated[0] or truncated[0])

        equity_curve.append(vec_env.get_attr("equity_usd")[0])

        trade_info = vec_env.get_attr("last_trade_info")[0]
        if isinstance(trade_info, dict) and trade_info.get("event") == "CLOSE":
            closed_trades.append(trade_info)

        if done:
            break

    return equity_curve, closed_trades


def main():
    # Load BTCUSDT data
    file_path = "data/BTCUSDT_Perpetual_1H_2025-10-01_2026-06-08.csv"
    df, feature_cols = load_and_preprocess_data(file_path)

    # Use test portion (last 30%)
    split_idx = int(len(df) * 0.7)
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()
    print(f"OOS dataset: {len(test_df)} bars")

    # Feature normalization — must match training (fit on train data only)
    market_mean = train_df[feature_cols].mean().values.astype(np.float32)
    market_std  = train_df[feature_cols].std().values.astype(np.float32)
    market_std  = np.where(market_std == 0, 1.0, market_std)
    state_mean = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    state_std  = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    feature_mean = np.concatenate([market_mean, state_mean])
    feature_std  = np.concatenate([market_std,  state_std])

    # Must match training params
    SL_OPTS = [200, 500, 1000, 2000]
    TP_OPTS = [200, 500, 1000, 2000]
    WIN = 30

    test_env = ForexTradingEnv(
        df=test_df,
            window_size=WIN,
            sl_options=SL_OPTS,
            tp_options=TP_OPTS,
            pip_value=1.0,
            spread_pips=0.0,
            commission_pips=0.0,
            max_slippage_pips=0.0,
            lot_size=1.0,
            random_start=False,
            episode_max_steps=None,
            feature_columns=feature_cols,
            feature_mean=feature_mean,
            feature_std=feature_std,
            hold_reward_weight=0.005,
            open_penalty_pips=2.0,
            time_penalty_pips=0.02,
            unrealized_delta_weight=0.1
    )

    vec_test_env = DummyVecEnv([lambda: test_env])

    # Load best model
    model = PPO.load("model_eurusd_best", env=vec_test_env)

    equity_curve, closed_trades = run_one_episode(model, vec_test_env, deterministic=True)

    # Save trades
    if closed_trades:
        trades_df = pd.DataFrame(closed_trades)
        out_csv = "trade_history_output.csv"
        trades_df.to_csv(out_csv, index=False)
        print(f"Closed trade history saved to {out_csv}")
    else:
        print("No closed trades recorded.")

    # Plot equity
    plt.figure(figsize=(10, 6))
    plt.plot(equity_curve, label="Equity (Test)")
    plt.title("Equity Curve - Evaluation")
    plt.xlabel("Steps")
    plt.ylabel("Equity ($)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("test_equity_curve.png", dpi=150)
    print("Chart saved to test_equity_curve.png")
    plt.close()


if __name__ == "__main__":
    main()
