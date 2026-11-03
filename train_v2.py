
import os
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback

from navigation_env_v2 import RobotNavigationEnvV2


ORIGINAL_MODEL = "models/ppo_navigation.zip"
NEW_MODEL = "models/ppo_navigation_v2.zip"

os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)

if not os.path.exists(ORIGINAL_MODEL):
    raise FileNotFoundError(
        "Original PPO model not found: " + ORIGINAL_MODEL
    )


def evaluate_model(model, episodes=30):

    env = RobotNavigationEnvV2()

    successes = 0
    collisions = 0
    episode_lengths = []
    episode_rewards = []

    for episode in range(episodes):

        obs, info = env.reset(seed=1000 + episode)

        total_reward = 0.0
        done = False
        steps = 0

        while not done:

            action, _ = model.predict(
                obs,
                deterministic=True
            )

            obs, reward, terminated, truncated, info = env.step(
                action
            )

            total_reward += reward
            steps += 1

            done = terminated or truncated

        successes += int(info["success"])
        collisions += int(info["collision"])

        episode_lengths.append(steps)
        episode_rewards.append(total_reward)

    env.close()

    return {
        "success_rate": 100 * successes / episodes,
        "collision_rate": 100 * collisions / episodes,
        "mean_steps": np.mean(episode_lengths),
        "mean_reward": np.mean(episode_rewards)
    }


def print_results(name, result):

    print("\n", name)

    print("-------------------------------")
    print("Success rate:", round(result["success_rate"], 2), "%")
    print("Collision rate:", round(result["collision_rate"], 2), "%")
    print("Mean episode length:", round(result["mean_steps"], 2))
    print("Mean reward:", round(result["mean_reward"], 2))


# ---------------------------------------
# 1. LOAD AND EVALUATE ORIGINAL MODEL
# ---------------------------------------

print("\nEvaluating original PPO model...")

original_model = PPO.load(
    ORIGINAL_MODEL,
    device="cpu"
)

original_results = evaluate_model(original_model)

print_results("ORIGINAL PPO MODEL", original_results)


# ---------------------------------------
# 2. CREATE NEW ENVIRONMENT
# ---------------------------------------

env = RobotNavigationEnvV2()


# ---------------------------------------
# 3. FINE-TUNE EXISTING PPO MODEL
# ---------------------------------------

print("\nLoading original model for fine-tuning...")

model = PPO.load(
    ORIGINAL_MODEL,
    env=env,
    device="cpu"
)

checkpoint_callback = CheckpointCallback(
    save_freq=50000,
    save_path="models/checkpoints_v2",
    name_prefix="ppo_navigation_v2"
)

print("\nStarting PPO fine-tuning...")

model.learn(
    total_timesteps=300000,
    callback=checkpoint_callback,
    reset_num_timesteps=False,
    progress_bar=False
)

# Save under a NEW filename
model.save(NEW_MODEL)

env.close()

print("\nNew model saved:", NEW_MODEL)


# ---------------------------------------
# 4. EVALUATE IMPROVED MODEL
# ---------------------------------------

print("\nEvaluating improved PPO model...")

new_model = PPO.load(
    NEW_MODEL,
    device="cpu"
)

new_results = evaluate_model(new_model)

print_results("IMPROVED PPO MODEL", new_results)


# ---------------------------------------
# 5. SAVE RESULTS
# ---------------------------------------

with open(
    "results/ppo_comparison.txt",
    "w"
) as file:

    file.write("PPO MODEL COMPARISON\n")
    file.write("====================\n\n")

    file.write("Original model:\n")
    for key, value in original_results.items():
        file.write(f"{key}: {value}\n")

    file.write("\nImproved model:\n")
    for key, value in new_results.items():
        file.write(f"{key}: {value}\n")

print("\nComparison saved in results/ppo_comparison.txt")
