
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from pathlib import Path

from stable_baselines3 import PPO
from environment.navigation_env import RobotNavigationEnv

# Project paths
ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "ppo_navigation.zip"
RESULTS_DIR = ROOT / "results"

RESULTS_DIR.mkdir(exist_ok=True)


def main():

    if not MODEL_PATH.exists():
        print("Model not found. Train the PPO agent first.")
        return

    # Load your already-trained PPO model
    model = PPO.load(str(MODEL_PATH))

    env = RobotNavigationEnv()

    # Try different layouts to find a successful demonstration
    selected_frames = None
    selected_info = None

    for seed in range(2026, 2056):

        obs, _ = env.reset(seed=seed)

        frames = [{
            "robot": env.robot.copy(),
            "obstacles": [
                (c.copy(), r) for c, r in env.obstacles
            ],
            "goal": env.goal.copy()
        }]

        done = False
        success = False
        collision = False

        while not done:

            action, _ = model.predict(
                obs,
                deterministic=True
            )

            obs, reward, terminated, truncated, info = env.step(action)

            success = success or info["success"]
            collision = collision or info["collision"]

            frames.append({
                "robot": env.robot.copy(),
                "obstacles": [
                    (c.copy(), r) for c, r in env.obstacles
                ],
                "goal": env.goal.copy()
            })

            done = terminated or truncated

        selected_frames = frames
        selected_info = {
            "success": success,
            "collision": collision,
            "seed": seed
        }

        if success:
            break

    env.close()

    # -----------------------------------
    # Create live robot navigation animation
    # -----------------------------------

    fig, ax = plt.subplots(figsize=(8, 8))

    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_aspect("equal")

    ax.set_xlabel("X Position")
    ax.set_ylabel("Y Position")
    ax.grid(True, alpha=0.3)

    first_frame = selected_frames[0]

    # Draw obstacles
    for center, radius in first_frame["obstacles"]:

        obstacle = plt.Circle(
            center,
            radius,
            color="red",
            alpha=0.6
        )

        ax.add_patch(obstacle)

    # Draw goal
    ax.scatter(
        9, 9,
        marker="*",
        s=250,
        color="green",
        label="Goal"
    )

    # Robot starting position
    robot, = ax.plot(
        [], [],
        "bo",
        markersize=12,
        label="Robot"
    )

    # Robot movement trail
    path, = ax.plot(
        [], [],
        "b-",
        linewidth=2,
        label="Robot Path"
    )

    step_label = ax.text(
        0.02, 0.96, "",
        transform=ax.transAxes
    )

    status_label = ax.text(
        0.02, 0.91, "",
        transform=ax.transAxes
    )

    ax.legend(loc="upper right")

    x_positions = []
    y_positions = []

    def update(frame_number):

        position = selected_frames[frame_number]["robot"]

        x = float(position[0])
        y = float(position[1])

        x_positions.append(x)
        y_positions.append(y)

        # Move robot
        robot.set_data([x], [y])

        # Update trajectory
        path.set_data(
            x_positions,
            y_positions
        )

        step_label.set_text(
            f"Step: {frame_number}/{len(selected_frames)-1}"
        )

        if frame_number == len(selected_frames) - 1:

            if selected_info["success"]:
                status = "GOAL REACHED"

            elif selected_info["collision"]:
                status = "COLLISION"

            else:
                status = "TIME LIMIT"

            status_label.set_text(status)

        ax.set_title(
            "RL-Based Autonomous Robot Navigation"
        )

        return robot, path, step_label, status_label

    # Create animation
    animation = FuncAnimation(
        fig,
        update,
        frames=len(selected_frames),
        interval=80,
        repeat=False,
        blit=False
    )

    # Save animated GIF
    gif_path = RESULTS_DIR / "robot_navigation.gif"

    animation.save(
        gif_path,
        writer=PillowWriter(fps=12)
    )

    print("Animation saved:", gif_path)
    print("Demonstration seed:", selected_info["seed"])
    print("Success:", selected_info["success"])
    print("Collision:", selected_info["collision"])

    # Display live animation
    plt.show()

    plt.close(fig)


if __name__ == "__main__":
    main()
