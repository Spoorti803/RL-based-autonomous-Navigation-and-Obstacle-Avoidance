from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Circle
from stable_baselines3 import PPO

from sensor_navigation_env import SensorNavigationEnv


# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------
PROJECT_DIR = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_DIR / "models" / "ppo_sensor_navigation.zip"
OUTPUT_DIR = PROJECT_DIR / "results"
GIF_PATH = OUTPUT_DIR / "sensor_lidar_navigation.gif"

EPISODE_SEED = 1000       # First evaluation seed used by train_sensor.py
MAX_FRAMES = 250
FRAME_INTERVAL_MS = 120
SAVE_GIF = True           # Change to False if you only want the live window


def collect_episode(model, seed):
    """Run one deterministic episode and record robot and LiDAR data."""
    env = SensorNavigationEnv()
    obs, _ = env.reset(seed=seed)

    frames = []
    done = False
    total_reward = 0.0

    while not done and len(frames) < MAX_FRAMES:
        # Record the actual LiDAR measurements used by the environment.
        lidar_readings = env.get_lidar_readings().copy()

        frames.append({
            "robot": env.robot.copy(),
            "goal": env.goal.copy(),
            "obstacles": env.obstacles.copy(),
            "lidar": lidar_readings,
            "step": env.steps,
        })

        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += float(reward)
        done = terminated or truncated

    # Record the final position and final sensor scan as the last frame.
    frames.append({
        "robot": env.robot.copy(),
        "goal": env.goal.copy(),
        "obstacles": env.obstacles.copy(),
        "lidar": env.get_lidar_readings().copy(),
        "step": env.steps,
    })

    result = {
        "success": bool(info.get("success", False)),
        "collision": bool(info.get("collision", False)),
        "steps": env.steps,
        "reward": total_reward,
    }
    env.close()
    return frames, result


def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Trained model not found: {MODEL_PATH}\n"
            "Run train_sensor.py first and check the models folder."
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading sensor-based PPO model...")
    model = PPO.load(str(MODEL_PATH), device="cpu")

    print(f"Running evaluation episode with seed {EPISODE_SEED}...")
    frames, result = collect_episode(model, EPISODE_SEED)

    print("\nEpisode result")
    print("------------------------------")
    print(f"Success:   {result['success']}")
    print(f"Collision: {result['collision']}")
    print(f"Steps:     {result['steps']}")
    print(f"Reward:    {result['reward']:.2f}")

    # Create figure and axes.
    fig, ax = plt.subplots(figsize=(8, 8))
    world_size = SensorNavigationEnv.WORLD_SIZE
    ax.set_xlim(0, world_size)
    ax.set_ylim(0, world_size)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("X position")
    ax.set_ylabel("Y position")
    ax.set_title("Sensor-Based PPO Navigation with 32-Ray LiDAR")
    ax.grid(True, alpha=0.25)

    # Draw world boundary.
    ax.plot(
        [0, world_size, world_size, 0, 0],
        [0, 0, world_size, world_size, 0],
        linewidth=2,
    )

    # Static goal marker.
    goal = frames[0]["goal"]
    goal_patch = Circle(
        goal, SensorNavigationEnv.GOAL_RADIUS,
        facecolor="green", edgecolor="darkgreen", alpha=0.65,
        label="Goal"
    )
    ax.add_patch(goal_patch)
    ax.plot(goal[0], goal[1], marker="*", markersize=14, color="green")

    # Static obstacles.
    for index, obstacle in enumerate(frames[0]["obstacles"]):
        patch = Circle(
            obstacle[:2], obstacle[2],
            facecolor="red", edgecolor="darkred", alpha=0.65,
            label="Obstacle" if index == 0 else None,
        )
        ax.add_patch(patch)

    # Robot and path.
    robot_patch = Circle(
        frames[0]["robot"], SensorNavigationEnv.ROBOT_RADIUS,
        facecolor="royalblue", edgecolor="navy", zorder=5,
        label="Robot"
    )
    ax.add_patch(robot_patch)
    path_line, = ax.plot([], [], linewidth=2, label="Robot path", zorder=3)

    # LiDAR rays: one line per actual sensor measurement.
    lidar_lines = []
    lidar_points, = ax.plot([], [], "o", markersize=3, zorder=4)
    for _ in range(SensorNavigationEnv.LIDAR_RAYS):
        line, = ax.plot([], [], linewidth=0.8, alpha=0.65)
        lidar_lines.append(line)

    status_text = ax.text(
        0.02, 0.98, "", transform=ax.transAxes,
        va="top", ha="left",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85)
    )
    ax.legend(loc="lower right")

    path_x = []
    path_y = []

    def update(frame_index):
        frame = frames[frame_index]
        robot = frame["robot"]
        readings = frame["lidar"]

        robot_patch.center = (float(robot[0]), float(robot[1]))

        path_x.append(float(robot[0]))
        path_y.append(float(robot[1]))
        path_line.set_data(path_x, path_y)

        hit_x = []
        hit_y = []

        for ray_index, distance in enumerate(readings):
            angle = 2.0 * np.pi * ray_index / SensorNavigationEnv.LIDAR_RAYS
            end_x = float(robot[0] + distance * np.cos(angle))
            end_y = float(robot[1] + distance * np.sin(angle))

            lidar_lines[ray_index].set_data(
                [float(robot[0]), end_x],
                [float(robot[1]), end_y],
            )
            hit_x.append(end_x)
            hit_y.append(end_y)

        lidar_points.set_data(hit_x, hit_y)

        status_text.set_text(
            f"Step: {frame['step']}\n"
            f"LiDAR rays: {len(readings)}\n"
            f"Min range: {float(np.min(readings)):.2f}\n"
            f"Max range: {float(np.max(readings)):.2f}"
        )

        return [robot_patch, path_line, lidar_points, status_text, *lidar_lines]

    animation = FuncAnimation(
        fig,
        update,
        frames=len(frames),
        interval=FRAME_INTERVAL_MS,
        blit=False,
        repeat=False,
    )

    if SAVE_GIF:
        try:
            print(f"\nSaving animation to: {GIF_PATH}")
            animation.save(
                str(GIF_PATH),
                writer=PillowWriter(fps=8),
            )
            print("GIF saved successfully.")
        except Exception as exc:
            print(f"Could not save GIF: {exc}")
            print("The animation will still open in a window.")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
