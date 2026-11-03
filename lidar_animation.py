
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Circle, Rectangle
from matplotlib.transforms import Affine2D
from stable_baselines3 import PPO
from environment.navigation_env import RobotNavigationEnv

# ---------------------------------------
# LOAD TRAINED PPO MODEL
# ---------------------------------------
model = PPO.load("models/ppo_navigation.zip", device="cpu")
env = RobotNavigationEnv()

# ---------------------------------------
# FIND A SUCCESSFUL EPISODE
# ---------------------------------------
selected_frames = None
selected_success = False

for seed in range(2026, 2056):
    obs, info = env.reset(seed=seed)
    frames = []

    for step in range(250):
        frames.append(np.array(obs).flatten().copy())

        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            frames.append(np.array(obs).flatten().copy())
            break

    success = bool(info.get("success", False))

    if selected_frames is None:
        selected_frames = frames
        selected_success = success

    if success:
        selected_frames = frames
        selected_success = True
        print("Successful episode found. Seed:", seed)
        break

if selected_frames is None:
    raise RuntimeError("No animation frames generated.")

# ---------------------------------------
# EXTRACT ENVIRONMENT INFORMATION
# ---------------------------------------
first = selected_frames[0]

if len(first) < 16:
    raise ValueError(
        "Expected 16 observation values. "
        "Check the observation layout in navigation_env.py."
    )

goal = first[2:4]

# Each obstacle contains x, y and radius
obstacles = first[4:16].reshape(4, 3)

centers = obstacles[:, :2]
radii = obstacles[:, 2]

print("Obstacle positions:", centers)
print("Obstacle radii:", radii)
# ---------------------------------------
# LIDAR CONFIGURATION
# ---------------------------------------
N_RAYS = 32
MAX_RANGE = 3.0
angles = np.linspace(0, 2*np.pi, N_RAYS, endpoint=False)

def lidar_scan(robot):
    distances = np.full(N_RAYS, MAX_RANGE)

    for i, angle in enumerate(angles):
        direction = np.array([np.cos(angle), np.sin(angle)])

        for center, radius in zip(centers, radii):
            relative = center - robot
            projection = np.dot(relative, direction)
            perpendicular = np.dot(relative, relative) - projection**2
            discriminant = radius**2 - perpendicular

            if discriminant >= 0:
                hit = projection - np.sqrt(discriminant)

                if 0 <= hit < distances[i]:
                    distances[i] = hit

    return distances

# ---------------------------------------
# DRAW ENVIRONMENT
# ---------------------------------------
fig, ax = plt.subplots(figsize=(8, 8))
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.set_aspect("equal")
ax.set_facecolor("#edf1f5")
ax.set_title("RL Autonomous Navigation with LiDAR")
ax.set_xlabel("X Position")
ax.set_ylabel("Y Position")
ax.grid(alpha=0.2)

# Goal
ax.scatter(*goal, marker="*", s=350, color="green",
           label="Goal", zorder=5)

# Obstacles
for center, radius in zip(centers, radii):
    ax.add_patch(Circle(center, radius, color="red",
                        alpha=0.65, zorder=3))

# Robot body
robot_body = Circle((1, 1), 0.23, color="#2878d0",
                    ec="black", linewidth=1.5, zorder=8)
ax.add_patch(robot_body)

# Wheels
left_wheel = Rectangle((0, 0), 0.12, 0.28,
                       color="black", zorder=9)
right_wheel = Rectangle((0, 0), 0.12, 0.28,
                        color="black", zorder=9)
ax.add_patch(left_wheel)
ax.add_patch(right_wheel)

# Direction
heading_line, = ax.plot([], [], color="yellow",
                        linewidth=3, zorder=10)

# LiDAR rays
ray_lines = [
    ax.plot([], [], color="orange", alpha=0.7,
            linewidth=0.8)[0]
    for _ in range(N_RAYS)
]

hit_points = ax.scatter([], [], color="orange",
                        s=12, zorder=7)

# Path
path_line, = ax.plot([], [], color="blue",
                     linewidth=1.8, label="Robot path")

status = ax.text(
    0.02, 0.97, "", transform=ax.transAxes,
    verticalalignment="top",
    bbox=dict(facecolor="white", alpha=0.85)
)

ax.legend(loc="lower right")

# ---------------------------------------
# ANIMATION UPDATE
# ---------------------------------------
def update(frame_no):
    frame = selected_frames[frame_no]
    robot = frame[:2]

    positions = np.array([
        f[:2] for f in selected_frames[:frame_no + 1]
    ])

    path_line.set_data(positions[:, 0], positions[:, 1])
    robot_body.center = robot

    # Calculate robot heading
    heading = 0.0
    if len(positions) > 1:
        movement = positions[-1] - positions[-2]
        if np.linalg.norm(movement) > 0.0001:
            heading = np.arctan2(movement[1], movement[0])

    direction = np.array([np.cos(heading), np.sin(heading)])

    heading_line.set_data(
        [robot[0], robot[0] + 0.38 * direction[0]],
        [robot[1], robot[1] + 0.38 * direction[1]]
    )

    # Position wheels
    for wheel, side in [
        (left_wheel, -1),
        (right_wheel, 1)
    ]:
        offset = np.array([
            -np.sin(heading), np.cos(heading)
        ]) * side * 0.19

        x, y = robot + offset

        wheel.set_xy((x - 0.06, y - 0.14))
        wheel.set_transform(
            Affine2D().rotate_around(
                robot[0], robot[1], heading
            ) + ax.transData
        )

    # LiDAR measurements
    distances = lidar_scan(robot)
    hits = []

    for i, angle in enumerate(angles):
        end = robot + distances[i] * np.array([
            np.cos(angle), np.sin(angle)
        ])

        ray_lines[i].set_data(
            [robot[0], end[0]],
            [robot[1], end[1]]
        )

        if distances[i] < MAX_RANGE:
            hits.append(end)

    if hits:
        hit_points.set_offsets(np.array(hits))
    else:
        hit_points.set_offsets(np.empty((0, 2)))

    result = "GOAL REACHED" if selected_success else "EPISODE ENDED"

    status.set_text(
        f"Step: {frame_no + 1}/{len(selected_frames)}\n"
        f"LiDAR rays: {N_RAYS}\n"
        f"Result: {result}"
    )

    return (
        robot_body, left_wheel, right_wheel,
        heading_line, path_line, status,
        hit_points, *ray_lines
    )

# ---------------------------------------
# SAVE AND DISPLAY ANIMATION
# ---------------------------------------
animation = FuncAnimation(
    fig,
    update,
    frames=len(selected_frames),
    interval=80,
    blit=False,
    repeat=True
)

os.makedirs("results", exist_ok=True)

output = "results/robot_lidar_animation.gif"

animation.save(output, writer=PillowWriter(fps=12))

print("\nAnimation saved successfully:", output)
print("Episode result:",
      "GOAL REACHED" if selected_success else "EPISODE ENDED")

plt.show()
