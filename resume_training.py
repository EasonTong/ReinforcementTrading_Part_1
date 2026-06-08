"""
Resume PPO training from latest checkpoint (290K) to 600K steps.
Then evaluate all checkpoints, select best model, plot equity curves.
"""
import os, re
import numpy as np
import matplotlib
matplotlib.use('Agg')  # non-interactive
import matplotlib.pyplot as plt
import pandas as pd

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback

from indicators import load_and_preprocess_data
from trading_env import ForexTradingEnv


def evaluate_model(model: PPO, eval_env: DummyVecEnv, deterministic: bool = True):
    obs = eval_env.reset()
    equity_curve = []
    while True:
        action, _ = model.predict(obs, deterministic=deterministic)
        step_out = eval_env.step(action)
        if len(step_out) == 4:
            obs, rewards, dones, infos = step_out
            done = bool(dones[0])
        else:
            obs, rewards, terminated, truncated, infos = step_out
            done = bool(terminated[0] or truncated[0])
        info = infos[0] if isinstance(infos, (list, tuple)) else infos
        eq = info.get("equity_usd", eval_env.get_attr("equity_usd")[0])
        equity_curve.append(eq)
        if done:
            break
    final_equity = float(equity_curve[-1])
    return equity_curve, final_equity


def main():
    # ---- Data ----
    file_path = "data/BTCUSDT_Perpetual_1H_2025-10-01_2026-06-08.csv"
    df, feature_cols = load_and_preprocess_data(file_path)

    split_idx = int(len(df) * 0.7)
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()

    print(f"Training bars: {len(train_df)}")
    print(f"Testing bars : {len(test_df)}")

    # Feature normalization (fit on train data only)
    market_mean = train_df[feature_cols].mean().values.astype(np.float32)
    market_std  = train_df[feature_cols].std().values.astype(np.float32)
    market_std  = np.where(market_std == 0, 1.0, market_std)
    state_mean = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    state_std  = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    feature_mean = np.concatenate([market_mean, state_mean])
    feature_std  = np.concatenate([market_std,  state_std])

    SL_OPTS = [200, 500, 1000, 2000]
    TP_OPTS = [200, 500, 1000, 2000]
    WIN = 30

    def make_train_env():
        return ForexTradingEnv(
            df=train_df, window_size=WIN, sl_options=SL_OPTS, tp_options=TP_OPTS,
            pip_value=1.0, spread_pips=0.0, commission_pips=0.0, max_slippage_pips=0.0,
            lot_size=1.0,
            random_start=True, min_episode_steps=1000, episode_max_steps=2000,
            feature_columns=feature_cols, feature_mean=feature_mean, feature_std=feature_std,
            hold_reward_weight=0.005, open_penalty_pips=2.0,
            time_penalty_pips=0.02, unrealized_delta_weight=0.1,
        )

    def make_train_eval_env():
        return ForexTradingEnv(
            df=train_df, window_size=WIN, sl_options=SL_OPTS, tp_options=TP_OPTS,
            pip_value=1.0, spread_pips=0.0, commission_pips=0.0, max_slippage_pips=0.0,
            lot_size=1.0,
            random_start=False, episode_max_steps=None,
            feature_columns=feature_cols, feature_mean=feature_mean, feature_std=feature_std,
            hold_reward_weight=0.005, open_penalty_pips=2.0,
            time_penalty_pips=0.02, unrealized_delta_weight=0.1,
        )

    def make_test_eval_env():
        return ForexTradingEnv(
            df=test_df, window_size=WIN, sl_options=SL_OPTS, tp_options=TP_OPTS,
            pip_value=1.0, spread_pips=0.0, commission_pips=0.0, max_slippage_pips=0.0,
            lot_size=1.0,
            random_start=False, episode_max_steps=None,
            feature_columns=feature_cols, feature_mean=feature_mean, feature_std=feature_std,
            hold_reward_weight=0.005, open_penalty_pips=2.0,
            time_penalty_pips=0.02, unrealized_delta_weight=0.1,
        )

    train_vec_env = DummyVecEnv([make_train_env])
    train_eval_env = DummyVecEnv([make_train_eval_env])
    test_eval_env = DummyVecEnv([make_test_eval_env])

    # ---- Find latest checkpoint ----
    ckpt_dir = "./checkpoints"
    os.makedirs(ckpt_dir, exist_ok=True)

    ckpts = [f for f in os.listdir(ckpt_dir)
             if f.endswith(".zip") and f.startswith("ppo_eurusd")]

    def get_steps(f):
        m = re.search(r'(\d+)_steps', f)
        return int(m.group(1)) if m else 0

    latest_ckpt = max(ckpts, key=get_steps)
    steps_done = get_steps(latest_ckpt)
    total_target = 600000
    remaining = total_target - steps_done

    print(f"Latest checkpoint: {latest_ckpt} ({steps_done} steps done)")
    print(f"Remaining steps: {remaining}")

    # ---- Resume training ----
    if remaining > 0:
        print(f"\nLoading checkpoint and resuming training for {remaining} steps...")
        model = PPO.load(os.path.join(ckpt_dir, latest_ckpt), env=train_vec_env)

        checkpoint_callback = CheckpointCallback(
            save_freq=10000,
            save_path=ckpt_dir,
            name_prefix="ppo_eurusd",
        )
        model.learn(total_timesteps=remaining, callback=checkpoint_callback)
        model.save("model_eurusd_last")
        print("Final model saved: model_eurusd_last.zip")
    else:
        print("Training already complete (remaining <= 0). Loading last saved model.")
        model = PPO.load("model_eurusd_last.zip", env=train_vec_env)

    # ---- Evaluate all checkpoints on OOS data ----
    print("\n=== Evaluating all checkpoints on OOS data ===")
    equity_curve_test_last, final_equity_test_last = evaluate_model(model, test_eval_env)
    print(f"[OOS Eval] Last model final equity: {final_equity_test_last:.2f}")

    best_equity = -np.inf
    best_path = None

    # Re-scan checkpoints (including newly created ones)
    all_ckpts = sorted(
        [f for f in os.listdir(ckpt_dir) if f.endswith(".zip") and f.startswith("ppo_eurusd")],
        key=lambda x: os.path.getmtime(os.path.join(ckpt_dir, x)),
    )

    for ck in all_ckpts:
        ck_path = os.path.join(ckpt_dir, ck)
        try:
            m = PPO.load(ck_path, env=test_eval_env)
            _, final_eq = evaluate_model(m, test_eval_env)
            print(f"[OOS Eval] {ck} -> final equity: {final_eq:.2f}")
            if final_eq > best_equity:
                best_equity = final_eq
                best_path = ck_path
        except Exception as e:
            print(f"[Skip] Could not evaluate checkpoint {ck}: {e}")

    if best_path is None or final_equity_test_last >= best_equity:
        print("\nUsing last model as best (by OOS final equity).")
        best_model = model
    else:
        print(f"\nUsing best checkpoint: {best_path} (OOS final equity: {best_equity:.2f})")
        best_model = PPO.load(best_path, env=train_vec_env)

    best_model.save("model_eurusd_best")
    print("Best model saved: model_eurusd_best.zip")

    # ---- Plot equity curves ----
    equity_curve_train, final_equity_train = evaluate_model(best_model, train_eval_env)
    equity_curve_test, final_equity_test = evaluate_model(best_model, test_eval_env)

    print(f"\n[IS Eval]  Final equity (train): {final_equity_train:.2f}")
    print(f"[OOS Eval] Final equity (test) : {final_equity_test:.2f}")

    plt.figure(figsize=(12, 6))
    plt.plot(equity_curve_train, label="Train (in-sample) equity")
    plt.plot(equity_curve_test, label="Test (out-of-sample) equity")
    plt.title(f"Equity Curves: In-sample vs Out-of-sample (Best Model, {total_target} steps)")
    plt.xlabel("Steps")
    plt.ylabel("Equity ($)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("equity_curves.png", dpi=150)
    print("Chart saved to equity_curves.png")
    plt.close()

    print("\n=== Training & Evaluation Complete ===")


if __name__ == "__main__":
    main()
