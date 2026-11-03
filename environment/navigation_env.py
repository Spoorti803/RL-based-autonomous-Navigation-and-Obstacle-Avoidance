import numpy as np
import gymnasium as gym
from gymnasium import spaces


class RobotNavigationEnv(gym.Env):
    """
    Lightweight 2D point-robot navigation environment.

    Observation:
      [robot_x, robot_y, goal_x, goal_y,
       obstacle_1_x, obstacle_1_y, obstacle_1_radius, ...]
    Action:
      [horizontal movement, vertical movement], each in [-1, 1].
    """
    metadata = {"render_modes": ["human"], "render_fps": 20}

    WORLD_SIZE = 10.0
    ROBOT_RADIUS = 0.18
    GOAL_RADIUS = 0.45
    NUM_OBSTACLES = 4
    MAX_STEPS = 250
    SPEED = 0.22

    def __init__(self, render_mode=None):
        super().__init__()
        self.render_mode = render_mode

        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32
        )

        obs_low = np.array(
            [0, 0, 0, 0] + [0, 0, 0] * self.NUM_OBSTACLES,
            dtype=np.float32
        )
        obs_high = np.array(
            [10, 10, 10, 10] + [10, 10, 1.2] * self.NUM_OBSTACLES,
            dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=obs_low, high=obs_high, dtype=np.float32
        )

        self.rng = np.random.default_rng()
        self.robot = np.array([1.0, 1.0], dtype=np.float32)
        self.goal = np.array([9.0, 9.0], dtype=np.float32)
        self.obstacles = []
        self.steps = 0

    def _sample_obstacles(self):
        obstacles = []
        attempts = 0

        while len(obstacles) < self.NUM_OBSTACLES and attempts < 1000:
            attempts += 1
            center = self.rng.uniform(1.5, 8.5, size=2)
            radius = float(self.rng.uniform(0.45, 0.85))

            if np.linalg.norm(center - self.robot) < 1.8:
                continue
            if np.linalg.norm(center - self.goal) < 1.8:
                continue

            overlaps = any(
                np.linalg.norm(center - old_center) < radius + old_radius + 0.6
                for old_center, old_radius in obstacles
            )
            if not overlaps:
                obstacles.append((center.astype(np.float32), radius))

        return obstacles

    def _get_obs(self):
        values = [
            self.robot[0], self.robot[1],
            self.goal[0], self.goal[1]
        ]
        for center, radius in self.obstacles:
            values.extend([center[0], center[1], radius])
        return np.asarray(values, dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.rng = np.random.default_rng(seed)

        self.robot = np.array([1.0, 1.0], dtype=np.float32)
        self.goal = np.array([9.0, 9.0], dtype=np.float32)
        self.obstacles = self._sample_obstacles()
        self.steps = 0

        return self._get_obs(), {}

    def step(self, action):
        self.steps += 1

        action = np.asarray(action, dtype=np.float32)
        action = np.clip(action, -1.0, 1.0)

        old_distance = float(np.linalg.norm(self.goal - self.robot))

        # Move the robot according to the policy action.
        self.robot += action * self.SPEED
        self.robot = np.clip(self.robot, 0.0, self.WORLD_SIZE)

        new_distance = float(np.linalg.norm(self.goal - self.robot))

        # Reward for progress toward the goal and a small time penalty.
        reward = 8.0 * (old_distance - new_distance)
        reward -= 0.02

        terminated = False
        truncated = self.steps >= self.MAX_STEPS
        info = {"success": False, "collision": False}

        # Find the nearest obstacle surface and detect collisions.
        min_clearance = float("inf")

        for center, radius in self.obstacles:
            distance = float(np.linalg.norm(self.robot - center))
            clearance = distance - radius - self.ROBOT_RADIUS
            min_clearance = min(min_clearance, clearance)

            if clearance <= 0.0:
                reward -= 30.0
                terminated = True
                info["collision"] = True
                break

        # A gentle penalty discourages the robot from scraping obstacles.
        if not terminated and min_clearance < 1.0:
            reward -= 0.2 * (1.0 - min_clearance)

        # Reward successful arrival at the target.
        if not terminated and new_distance <= self.GOAL_RADIUS:
            reward += 50.0
            terminated = True
            info["success"] = True

        # Discourage repeatedly pushing against the world boundary.
        if np.any(self.robot <= 0.0) or np.any(self.robot >= self.WORLD_SIZE):
            reward -= 0.5

        return self._get_obs(), float(reward), terminated, truncated, info

    def render(self):
        import matplotlib.pyplot as plt

        if not hasattr(self, "_fig"):
            self._fig, self._ax = plt.subplots(figsize=(6, 6))
            plt.ion()

        self._ax.clear()
        self._ax.set_xlim(0, self.WORLD_SIZE)
        self._ax.set_ylim(0, self.WORLD_SIZE)
        self._ax.set_aspect("equal")
        self._ax.grid(True, alpha=0.25)

        self._ax.scatter(*self.goal, marker="*", s=220, label="Goal")
        self._ax.scatter(*self.robot, marker="o", s=100, label="Robot")

        for center, radius in self.obstacles:
            self._ax.add_patch(
                plt.Circle(center, radius, color="tomato", alpha=0.65)
            )

        self._ax.set_title("2D Robot Navigation")
        self._ax.legend(loc="upper left")
        self._fig.canvas.draw()
        self._fig.canvas.flush_events()
        plt.pause(0.001)

    def close(self):
        if hasattr(self, "_fig"):
            import matplotlib.pyplot as plt
            plt.close(self._fig)
            del self._fig
