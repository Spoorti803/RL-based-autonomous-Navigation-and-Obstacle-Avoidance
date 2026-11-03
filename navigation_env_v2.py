
import numpy as np
import gymnasium as gym
from gymnasium import spaces


class RobotNavigationEnvV2(gym.Env):
    """
    Improved PPO navigation environment.

    Observation:
    [robot_x, robot_y, goal_x, goal_y,
     obstacle_1_x, obstacle_1_y, obstacle_1_radius, ...]

    Action:
    [horizontal movement, vertical movement]
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
            low=-1.0,
            high=1.0,
            shape=(2,),
            dtype=np.float32
        )

        obs_low = np.array(
            [0, 0, 0, 0] +
            [0, 0, 0] * self.NUM_OBSTACLES,
            dtype=np.float32
        )

        obs_high = np.array(
            [10, 10, 10, 10] +
            [10, 10, 1.2] * self.NUM_OBSTACLES,
            dtype=np.float32
        )

        self.observation_space = spaces.Box(
            low=obs_low,
            high=obs_high,
            dtype=np.float32
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
                np.linalg.norm(center - old_center)
                < radius + old_radius + 0.6
                for old_center, old_radius in obstacles
            )

            if not overlaps:
                obstacles.append(
                    (center.astype(np.float32), radius)
                )

        if len(obstacles) != self.NUM_OBSTACLES:
            raise RuntimeError("Could not generate enough obstacles.")

        return obstacles

    def _get_obs(self):
        values = [
            self.robot[0],
            self.robot[1],
            self.goal[0],
            self.goal[1]
        ]

        for center, radius in self.obstacles:
            values.extend([
                center[0],
                center[1],
                radius
            ])

        return np.asarray(values, dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.rng = np.random.default_rng(seed)

        self.robot = np.array(
            [1.0, 1.0],
            dtype=np.float32
        )

        self.goal = np.array(
            [9.0, 9.0],
            dtype=np.float32
        )

        self.obstacles = self._sample_obstacles()
        self.steps = 0

        return self._get_obs(), {}

    def _segment_clearance(self, start, end, center, radius):
        """
        Minimum clearance between a movement segment
        and an obstacle, including robot radius.
        """

        segment = end - start
        length_sq = float(np.dot(segment, segment))

        if length_sq < 1e-12:
            closest = start
        else:
            t = np.dot(center - start, segment) / length_sq
            t = np.clip(t, 0.0, 1.0)
            closest = start + t * segment

        distance = np.linalg.norm(closest - center)

        return float(
            distance - radius - self.ROBOT_RADIUS
        )

    def step(self, action):

        self.steps += 1

        action = np.asarray(action, dtype=np.float32)
        action = np.clip(action, -1.0, 1.0)

        old_robot = self.robot.copy()

        old_distance = float(
            np.linalg.norm(self.goal - old_robot)
        )

        # Calculate proposed movement
        proposed_robot = old_robot + action * self.SPEED

        proposed_robot = np.clip(
            proposed_robot,
            0.0,
            self.WORLD_SIZE
        )

        # Calculate distance to goal after movement
        new_distance = float(
            np.linalg.norm(self.goal - proposed_robot)
        )

        # Reward for making progress
        reward = 10.0 * (old_distance - new_distance)

        # Small time penalty encourages faster navigation
        reward -= 0.04

        terminated = False
        truncated = self.steps >= self.MAX_STEPS

        info = {
            "success": False,
            "collision": False,
            "clearance": float("inf")
        }

        min_clearance = float("inf")

        # Check the complete movement segment
        # to prevent crossing an obstacle between steps.
        for center, radius in self.obstacles:

            clearance = self._segment_clearance(
                old_robot,
                proposed_robot,
                center,
                radius
            )

            min_clearance = min(
                min_clearance,
                clearance
            )

            if clearance <= 0.0:
                reward -= 45.0
                terminated = True
                info["collision"] = True
                break

        # Update robot position
        self.robot = proposed_robot

        info["clearance"] = min_clearance

        # Smooth safety penalty:
        # Penalize getting too close, without forcing huge detours.
        if not terminated and min_clearance < 0.9:

            safety_penalty = 1.2 * (
                0.9 - min_clearance
            ) ** 2

            reward -= safety_penalty

        # Goal reached
        if not terminated and new_distance <= self.GOAL_RADIUS:

            reward += 100.0

            terminated = True
            info["success"] = True

        # Boundary penalty
        if np.any(self.robot <= 0.0) or np.any(
            self.robot >= self.WORLD_SIZE
        ):
            reward -= 0.5

        return (
            self._get_obs(),
            float(reward),
            terminated,
            truncated,
            info
        )

    def close(self):
        pass
