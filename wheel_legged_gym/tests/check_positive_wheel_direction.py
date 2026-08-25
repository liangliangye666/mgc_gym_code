"""Isaac Gym micro-test for the L5A wheel-axis sign and cylindrical collision.

Run from the repository root with the same Isaac Gym environment used for
training, for example:

    ROBOT_TYPE=l5a_2wheel_upstairs_cp_liang python \
        wheel_legged_gym/tests/check_positive_wheel_direction.py \
        --task l5a_2wheel --headless --sim_device cpu --pipeline cpu
"""

import os

os.environ.setdefault("ROBOT_TYPE", "l5a_2wheel_upstairs_cp_liang")

import isaacgym  # noqa: F401: Isaac Gym must load before torch.
import torch

from wheel_legged_gym.envs import *  # noqa: F401,F403: performs task registration.
from wheel_legged_gym.utils import get_args, task_registry


def main():
    args = get_args()
    cfg, _ = task_registry.get_cfgs(args.task)
    cfg.env.num_envs = 1
    cfg.terrain.num_rows = 1
    cfg.terrain.num_cols = 1
    cfg.terrain.border_size = 2.0
    cfg.terrain.max_init_terrain_level = 0
    cfg.terrain.curriculum = True
    cfg.init_state.pos = [0.0, 0.0, 0.643]
    cfg.noise.add_noise = False

    disabled_randomizations = (
        "push_robots",
        "randomize_friction",
        "randomize_restitution",
        "randomize_base_com",
        "randomize_Kp",
        "randomize_Kd",
        "randomize_motor_torque",
        "randomize_default_dof_pos",
        "randomize_action_delay",
        "randomize_obs_delay",
    )
    for name in disabled_randomizations:
        setattr(cfg.domain_rand, name, False)

    env, _ = task_registry.make_env(args.task, args=args, env_cfg=cfg)
    actions = torch.zeros((1, env.num_actions), device=env.device)
    actions[:, env.wheel_indices] = 2.0

    start_x = env.root_states[0, 0].item()
    env.step(actions)
    measured_speed = (env.root_states[0, 0].item() - start_x) / env.dt
    wheel_speed = torch.mean(env.dof_vel[0, env.wheel_indices]).item()
    rolling_speed = cfg.asset.wheel_radius * wheel_speed
    error = abs(measured_speed - rolling_speed)

    print(
        "positive wheel direction:",
        f"base_vx={measured_speed:.4f} m/s,",
        f"r_omega={rolling_speed:.4f} m/s,",
        f"error={error:.4f} m/s",
    )
    assert measured_speed > 0.0
    assert wheel_speed > 0.0
    assert error < 0.15


if __name__ == "__main__":
    main()
