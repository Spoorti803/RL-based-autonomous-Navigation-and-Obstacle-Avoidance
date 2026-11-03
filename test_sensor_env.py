import numpy as np
from sensor_navigation_env import SensorNavigationEnv

env = SensorNavigationEnv()
obs, info = env.reset(seed=42)

print("Environment reset successful")
print("Observation shape:", obs.shape)
print("Expected shape: (34,)")
print("Goal-relative observation:", obs[:2])
print("LiDAR readings (normalized):", np.round(obs[2:], 3))
print("Number of LiDAR readings:", len(obs[2:]))

assert obs.shape == (34,), f"Expected 34 values, got {obs.shape}"
assert np.all(obs[:2] >= -1.0) and np.all(obs[:2] <= 1.0)
assert np.all(obs[2:] >= 0.0) and np.all(obs[2:] <= 1.0)

# Take one harmless test action.
next_obs, reward, terminated, truncated, step_info = env.step(
    np.array([0.5, 0.5], dtype=np.float32)
)

print("\nOne-step test successful")
print("Next observation shape:", next_obs.shape)
print("Reward:", round(reward, 3))
print("Terminated:", terminated)
print("Truncated:", truncated)
print("Step information:", step_info)

assert next_obs.shape == (34,)
env.close()
print("\nPASS: SensorNavigationEnv is working.")
