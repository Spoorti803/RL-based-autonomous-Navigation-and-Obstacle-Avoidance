import math
import numpy as np
import gymnasium as gym
from gymnasium import spaces


class SensorNavigationEnv(gym.Env):
    """
    2D mobile-robot navigation with simulated 360-degree LiDAR.

    PPO observation:
      [goal_dx, goal_dy, lidar_0, ..., lidar_(N-1)]
    Goal offsets and LiDAR distances are normalized to [-1, 1] and [0, 1].

    PPO action:
      [forward_x, forward_y], each in [-1, 1].

    LiDAR readings are ray-cast from the robot to circular obstacles and
    the world boundary. These readings ARE part of the policy observation.
    """

    metadata = {"render_modes": []}

    WORLD_SIZE = 10.0
    ROBOT_RADIUS = 0.18
    GOAL_RADIUS = 0.45
    NUM_OBSTACLES = 4
    MAX_STEPS = 250
    SPEED = 0.22
    LIDAR_RAYS = 32
    LIDAR_MAX_RANGE = 4.0

    def __init__(self):
        super().__init__()

        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32
        )

        # 2 normalized goal-relative values + 32 normalized LiDAR ranges
        self.observation_space = spaces.Box(
            low=np.array([-1.0, -1.0] + [0.0] * self.LIDAR_RAYS, dtype=np.float32),
            high=np.array([1.0, 1.0] + [1.0] * self.LIDAR_RAYS, dtype=np.float32),
            dtype=np.float32,
        )

        self.robot = np.array([1.0, 1.0], dtype=np.float32)
        self.goal = np.array([9.0, 9.0], dtype=np.float32)
        self.obstacles = np.zeros((self.NUM_OBSTACLES, 3), dtype=np.float32)
        self.steps = 0

    def _sample_obstacles(self):
        obstacles = []
        attempts = 0

        while len(obstacles) < self.NUM_OBSTACLES and attempts < 5000:
            attempts += 1
            x, y = self.np_random.uniform(1.5, 8.5, size=2)
            radius = self.np_random.uniform(0.45, 0.85)

            center = np.array([x, y], dtype=np.float32)

            if np.linalg.norm(center - self.robot) < radius + 1.2:
                continue
            if np.linalg.norm(center - self.goal) < radius + 1.0:
                continue

            if any(np.linalg.norm(center - old[:2]) < radius + old[2] + 0.5
                   for old in obstacles):
                continue

            obstacles.append(np.array([x, y, radius], dtype=np.float32))

        if len(obstacles) != self.NUM_OBSTACLES:
            raise RuntimeError("Could not place all obstacles. Try resetting with another seed.")

        self.obstacles = np.asarray(obstacles, dtype=np.float32)

    def _ray_circle_distance(self, origin, direction, center, radius):
        # Solve |origin + t*direction - center|^2 = radius^2.
        offset = origin - center
        b = 2.0 * float(np.dot(offset, direction))
        c = float(np.dot(offset, offset) - radius * radius)
        discriminant = b * b - 4.0 * c

        if discriminant < 0.0:
            return None

        root = math.sqrt(discriminant)
        t1 = (-b - root) / 2.0
        t2 = (-b + root) / 2.0
        valid = [t for t in (t1, t2) if t >= 0.0]
        return min(valid) if valid else None

    def _ray_boundary_distance(self, origin, direction):
        distances = []

        if direction[0] > 1e-9:
            distances.append((self.WORLD_SIZE - origin[0]) / direction[0])
        elif direction[0] < -1e-9:
            distances.append((0.0 - origin[0]) / direction[0])

        if direction[1] > 1e-9:
            distances.append((self.WORLD_SIZE - origin[1]) / direction[1])
        elif direction[1] < -1e-9:
            distances.append((0.0 - origin[1]) / direction[1])

        valid = [d for d in distances if d >= 0.0]
        return min(valid) if valid else self.LIDAR_MAX_RANGE

    def get_lidar_readings(self):
        """Return 32 distances in metres, clipped to the sensor max range."""
        readings = []
        for i in range(self.LIDAR_RAYS):
            angle = 2.0 * math.pi * i / self.LIDAR_RAYS
            direction = np.array([math.cos(angle), math.sin(angle)], dtype=np.float32)

            nearest = min(self.LIDAR_MAX_RANGE,
                          self._ray_boundary_distance(self.robot, direction))

            for obstacle in self.obstacles:
                hit = self._ray_circle_distance(
                    self.robot, direction, obstacle[:2], float(obstacle[2])
                )
                if hit is not None:
                    nearest = min(nearest, hit)

            readings.append(nearest)

        return np.asarray(readings, dtype=np.float32)

    def _get_obs(self):
        goal_delta = (self.goal - self.robot) / self.WORLD_SIZE
        lidar = self.get_lidar_readings() / self.LIDAR_MAX_RANGE
        return np.concatenate([goal_delta, lidar]).astype(np.float32)

    def _segment_clearance(self, start, end, center, radius):
        segment = end - start
        length_sq = float(np.dot(segment, segment))

        if length_sq < 1e-12:
            closest = start
        else:
            t = float(np.dot(center - start, segment) / length_sq)
            t = float(np.clip(t, 0.0, 1.0))
            closest = start + t * segment

        return float(np.linalg.norm(closest - center) - radius - self.ROBOT_RADIUS)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.robot = np.array([1.0, 1.0], dtype=np.float32)
        self.goal = np.array([9.0, 9.0], dtype=np.float32)
        self.steps = 0
        self._sample_obstacles()
        return self._get_obs(), {}

    def step(self, action):
        action = np.asarray(action, dtype=np.float32)
        action = np.clip(action, -1.0, 1.0)

        old_distance = float(np.linalg.norm(self.goal - self.robot))
        proposed = self.robot + action * self.SPEED
        proposed = np.clip(proposed, self.ROBOT_RADIUS,
                           self.WORLD_SIZE - self.ROBOT_RADIUS)

        min_clearance = float("inf")
        collision = False

        for obstacle in self.obstacles:
            clearance = self._segment_clearance(
                self.robot, proposed, obstacle[:2], float(obstacle[2])
            )
            min_clearance = min(min_clearance, clearance)
            if clearance <= 0.0:
                collision = True

        self.robot = proposed
        self.steps += 1

        new_distance = float(np.linalg.norm(self.goal - self.robot))
        reward = 10.0 * (old_distance - new_distance) - 0.04

        terminated = False
        truncated = self.steps >= self.MAX_STEPS

        if collision:
            reward -= 45.0
            terminated = True
        else:
            if min_clearance < 0.9:
                reward -= 1.2 * (0.9 - min_clearance) ** 2

            if new_distance <= self.GOAL_RADIUS:
                reward += 100.0
                terminated = True

        info = {
            "success": bool(new_distance <= self.GOAL_RADIUS and not collision),
            "collision": bool(collision),
            "distance_to_goal": new_distance,
            "min_obstacle_clearance": min_clearance,
        }

        return self._get_obs(), float(reward), terminated, truncated, info
