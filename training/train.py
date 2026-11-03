from pathlib import Path
import sys
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from environment.navigation_env import RobotNavigationEnv

MODEL_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"
MODEL_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)


def evaluate(model, episodes=20):
    env = RobotNavigationEnv()

    successes = []
    collisions = []
    episode_lengths = []
    episode_returns = []

    for episode in range(episodes):
        obs, _ = env.reset(seed=1000 + episode)
        done = False
        total_reward = 0.0
        length = 0
        success = False
        collision = False

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)

            total_reward += reward
            length += 1
            success = success or info["success"]
            collision = collision or info["collision"]
            done = terminated or truncated

        successes.append(int(success))
        collisions.append(int(collision))
        episode_lengths.append(length)
        episode_returns.append(total_reward)

    success_rate = 100.0 * np.mean(successes)
    collision_rate = 100.0 * np.mean(collisions)
    mean_length = float(np.mean(episode_lengths))
    mean_reward = float(np.mean(episode_returns))

    print(f"\nEvaluation on {episodes} unseen obstacle layouts")
    print(f"Success rate:   {success_rate:.1f}%")
    print(f"Collision rate: {collision_rate:.1f}%")
    print(f"Mean episode length: {mean_length:.1f}")
    print(f"Mean episode reward: {mean_reward:.2f}")

    metrics_path = RESULTS_DIR / "evaluation_metrics.txt"
    with open(metrics_path, "w", encoding="utf-8") as f:
        f.write(f"Episodes: {episodes}\n")
        f.write(f"Success rate: {success_rate:.1f}%\n")
        f.write(f"Collision rate: {collision_rate:.1f}%\n")
        f.write(f"Mean episode length: {mean_length:.1f}\n")
        f.write(f"Mean episode reward: {mean_reward:.2f}\n")

    plt.figure(figsize=(7, 4))
    plt.bar(["Success", "Collision"], [success_rate, collision_rate])
    plt.ylabel("Episodes (%)")
    plt.ylim(0, 100)
    plt.title("PPO Navigation Evaluation")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "evaluation_plot.png", dpi=150)
    plt.close()

    env.close()


def main():
    # CPU-friendly PPO configuration. Training time depends on your laptop.
    env = make_vec_env(RobotNavigationEnv, n_envs=1, seed=42)

    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=0.0003,
        n_steps=2048,
        batch_size=64,
        gamma=0.99,
        gae_lambda=0.95,
        ent_coef=0.01,
        verbose=1,
        policy_kwargs=dict(net_arch=[128, 128]),
        device="cpu",
    )

    total_timesteps = 200000
    print(f"Starting PPO training for {total_timesteps} timesteps...")
    model.learn(total_timesteps=total_timesteps, progress_bar=False)

    model_path = MODEL_DIR / "ppo_navigation"
    model.save(str(model_path))
    env.close()

    print(f"\nTraining complete. Model saved to {model_path}.zip")
    evaluate(model, episodes=20)


if __name__ == "__main__":
    main()
