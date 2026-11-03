from pathlib import Path
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor

from sensor_navigation_env import SensorNavigationEnv


# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------
TOTAL_TIMESTEPS = 200_000
SEED = 42

PROJECT_DIR = Path(__file__).resolve().parent
MODEL_DIR = PROJECT_DIR / "models"
CHECKPOINT_DIR = MODEL_DIR / "checkpoints_sensor"
RESULTS_DIR = PROJECT_DIR / "results"

FINAL_MODEL_PATH = MODEL_DIR / "ppo_sensor_navigation"
RESULTS_PATH = RESULTS_DIR / "sensor_ppo_evaluation.txt"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def evaluate_model(model, episodes=30):
    """Evaluate the trained policy on fixed, reproducible episode seeds."""
    eval_env = SensorNavigationEnv()

    successes = 0
    collisions = 0
    episode_lengths = []
    episode_rewards = []

    for episode in range(episodes):
        obs, info = eval_env.reset(seed=1000 + episode)
        done = False
        total_reward = 0.0
        steps = 0
        final_info = {}

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = eval_env.step(action)

            total_reward += float(reward)
            steps += 1
            done = terminated or truncated
            final_info = info

        successes += int(final_info.get("success", False))
        collisions += int(final_info.get("collision", False))
        episode_lengths.append(steps)
        episode_rewards.append(total_reward)

        print(
            f"Evaluation episode {episode + 1:02d}/{episodes}: "
            f"success={final_info.get('success', False)}, "
            f"collision={final_info.get('collision', False)}, "
            f"steps={steps}, reward={total_reward:.2f}"
        )

    eval_env.close()

    success_rate = 100.0 * successes / episodes
    collision_rate = 100.0 * collisions / episodes
    mean_steps = float(np.mean(episode_lengths))
    mean_reward = float(np.mean(episode_rewards))

    return {
        "episodes": episodes,
        "successes": successes,
        "collisions": collisions,
        "success_rate": success_rate,
        "collision_rate": collision_rate,
        "mean_steps": mean_steps,
        "mean_reward": mean_reward,
    }


def main():
    print("=" * 60)
    print("SENSOR-BASED PPO NAVIGATION TRAINING")
    print("=" * 60)
    print(f"Training timesteps: {TOTAL_TIMESTEPS}")
    print("Observation: 2 goal-relative values + 32 LiDAR readings = 34")
    print("Action: 2 continuous movement values")
    print("Device: CPU")
    print()

    # A fresh environment is used. No existing PPO model is loaded or changed.
    train_env = Monitor(SensorNavigationEnv())

    # Save a checkpoint every 25,000 environment steps.
    checkpoint_callback = CheckpointCallback(
        save_freq=25_000,
        save_path=str(CHECKPOINT_DIR),
        name_prefix="ppo_sensor_navigation",
        save_replay_buffer=False,
        save_vecnormalize=False,
    )

    model = PPO(
        policy="MlpPolicy",
        env=train_env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        policy_kwargs=dict(net_arch=[128, 128]),
        verbose=1,
        seed=SEED,
        device="cpu",
    )

    print("Starting training...")
    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=checkpoint_callback,
        progress_bar=False,
    )

    # Save to a NEW filename; existing models are not overwritten.
    model.save(str(FINAL_MODEL_PATH))
    train_env.close()

    print()
    print(f"Training finished. Model saved to: {FINAL_MODEL_PATH}.zip")
    print("Starting evaluation over 30 episodes...")
    print()

    results = evaluate_model(model, episodes=30)

    report = (
        "SENSOR-BASED PPO EVALUATION\n"
        "===========================\n"
        f"Episodes: {results['episodes']}\n"
        f"Successful episodes: {results['successes']}\n"
        f"Collision episodes: {results['collisions']}\n"
        f"Success rate: {results['success_rate']:.2f}%\n"
        f"Collision rate: {results['collision_rate']:.2f}%\n"
        f"Mean episode length: {results['mean_steps']:.2f} steps\n"
        f"Mean episode reward: {results['mean_reward']:.2f}\n"
    )

    RESULTS_PATH.write_text(report, encoding="utf-8")

    print()
    print(report)
    print(f"Evaluation results saved to: {RESULTS_PATH}")
    print(f"Final model: {FINAL_MODEL_PATH}.zip")
    print(f"Checkpoints: {CHECKPOINT_DIR}")


if __name__ == "__main__":
    main()
