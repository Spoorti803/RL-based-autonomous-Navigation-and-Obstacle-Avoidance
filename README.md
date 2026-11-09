# RL-Based Autonomous Navigation and Obstacle Avoidance

## Project overview
This project demonstrates reinforcement-learning-based navigation in a 2D simulated environment. A mobile robot learns to move from a start position to a goal while avoiding randomly positioned circular obstacles. The navigation policy uses Proximal Policy Optimization (PPO) with continuous movement actions.

A simulated LiDAR visualization displays 32 rays around the robot. The rays illustrate obstacle directions and approximate distances.
> 🚧 **Status: Actively under development.** Features and evaluation are being expanded.
## Objectives
- Build a 2D robot navigation environment.
- Train a PPO policy for continuous movement.
- Encourage progress toward the goal and safe obstacle clearance.
- Detect collisions during movement.
- Evaluate success rate, collision rate, episode length, and reward.
- Visualize the robot path and simulated LiDAR rays.

## Technologies
Python, Gymnasium, Stable-Baselines3, PyTorch, NumPy, Matplotlib, and Pillow (for GIF output).

## Environment
The world is 10 × 10 units. The robot starts near (1, 1), and the goal is near (9, 9). Four circular obstacles are generated with randomized positions.

The 16-element observation contains:
1. Robot x and y coordinates.
2. Goal x and y coordinates.
3. Four obstacle records, each containing x, y, and radius.

The action is a continuous two-dimensional vector for horizontal and vertical movement. Collision checking considers the robot's movement segment, not only its final position.

## PPO method
PPO learns a policy by interacting with the environment and updating its action-selection strategy using collected experience. The reward encourages progress toward the goal, rewards goal completion, penalizes collisions, and discourages unsafe proximity to obstacles.

The trained model used for the demonstration is `models/ppo_navigation_v2.zip`.

## LiDAR visualization
The animation displays 32 orange rays, red obstacle circles, a green goal marker, and the robot trajectory in blue.

**Important:** The current LiDAR rays are calculated from known obstacle geometry for visualization. They are not fed into the PPO policy. The policy observations contain robot/goal coordinates and obstacle coordinates/radii.

## Evaluation results
Evaluation was performed over 30 episodes.

| Metric | Result |
|---|---:|
| Success rate | 93.33% (28/30) |
| Collision rate | 3.33% (1/30) |
| Mean episode length | 67.27 steps |
| Mean episode reward | 187.65 |

These results describe the tested simulation and do not guarantee identical performance for every possible obstacle layout.

## Project files
Typical project files include:
- `environment/navigation_env.py` — original environment implementation.
- `navigation_env_v2.py` — improved environment used for the final trained model.
- `training/` — training scripts.
- `models/ppo_navigation_v2.zip` — trained PPO policy.
- `lidar_animation_v2.py` — LiDAR-style animation.
- `results/` — evaluation output and animation.

## Running the animation
From the project root in Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
python lidar_animation_v2.py
```

Ensure that the model path and environment import in the animation script match the files in your project.

## Limitations
- The environment is 2D, not a 3D physics-based simulator.
- The PPO policy uses obstacle coordinates and radii, not raw LiDAR or camera data.
- LiDAR rays are visual-only in this implementation.
- No physical robot, ROS 2 deployment, or Jetson integration has been completed.
- Performance depends on the tested environment and random obstacle layouts.


## Conclusion
The project demonstrates PPO-based autonomous navigation and obstacle avoidance in a 2D simulation. The trained policy achieved a 93.33% success rate and a 3.33% collision rate over 30 evaluation episodes. The visualization illustrates the robot's route and simulated LiDAR rays, providing a foundation for future sensor-driven and real-robot navigation work.
