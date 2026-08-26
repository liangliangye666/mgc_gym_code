import ast
import math
import importlib.util
from pathlib import Path

import torch


MODULE_PATH = Path(__file__).parents[1] / "utils" / "wheel_center_trajectory.py"
REPO_ROOT = Path(__file__).parents[2]
SPEC = importlib.util.spec_from_file_location("wheel_center_trajectory", MODULE_PATH)
wheel_trajectory = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(wheel_trajectory)

curriculum_obstacle_height = wheel_trajectory.curriculum_obstacle_height
histogram_quantile = wheel_trajectory.histogram_quantile
nearest_wheel_center_path_state = wheel_trajectory.nearest_wheel_center_path_state
tangent_rolling_error = wheel_trajectory.tangent_rolling_error
wheel_center_path = wheel_trajectory.wheel_center_path
wheel_trajectory_completion = wheel_trajectory.wheel_trajectory_completion


WHEEL_RADIUS = 0.127


def test_curriculum_levels_cover_two_to_sixteen_centimetres():
    heights = [
        curriculum_obstacle_height(level, 10, 0.02, 0.16)
        for level in range(10)
    ]
    assert heights[0] == 0.02
    assert heights[-1] == 0.16
    assert math.isclose(heights[2], 0.02 + 0.14 * 2.0 / 9.0)


def test_wheel_center_path_has_expected_endpoints_and_unit_tangents():
    heights = torch.tensor([0.02, WHEEL_RADIUS, 0.16])
    points, tangents, _ = wheel_center_path(heights, WHEEL_RADIUS, 33)

    assert torch.allclose(points[:, 0], torch.zeros(3, 2), atol=1e-7)
    assert torch.allclose(points[:, -1, 1], heights, atol=1e-6)
    expected_low_x = math.sqrt(
        2.0 * WHEEL_RADIUS * heights[0].item() - heights[0].item() ** 2
    )
    expected_x = torch.tensor([expected_low_x, WHEEL_RADIUS, WHEEL_RADIUS])
    assert torch.allclose(points[:, -1, 0], expected_x, atol=1e-6)
    assert torch.allclose(
        torch.linalg.vector_norm(tangents, dim=-1),
        torch.ones(3, 33),
        atol=1e-6,
    )
    assert torch.allclose(tangents[:, -1], torch.tensor([[1.0, 0.0]] * 3), atol=1e-6)


def test_path_is_continuous_at_wheel_radius_boundary():
    heights = torch.tensor([WHEEL_RADIUS - 1e-6, WHEEL_RADIUS, WHEEL_RADIUS + 1e-6])
    points, tangents, lengths = wheel_center_path(heights, WHEEL_RADIUS, 257)
    assert torch.max(torch.abs(points[0] - points[1])) < 2e-6
    assert torch.max(torch.abs(points[1] - points[2])) < 2e-6
    assert torch.max(torch.abs(tangents[0] - tangents[1])) < 2e-5
    assert torch.max(torch.abs(tangents[1] - tangents[2])) < 2e-5
    assert torch.max(torch.abs(lengths[1:] - lengths[:-1])) < 2e-6


def test_nearest_path_state_recovers_sample_progress_and_end_error():
    height = torch.tensor([0.05])
    points, tangents, _ = wheel_center_path(height, WHEEL_RADIUS, 33)
    sample_index = 20
    error, progress, tangent, end_error = nearest_wheel_center_path_state(
        points[:, sample_index], points, tangents
    )
    assert torch.allclose(error, torch.zeros(1))
    assert torch.allclose(progress, torch.tensor([sample_index / 32.0]))
    assert torch.allclose(tangent, tangents[:, sample_index])
    assert end_error.item() > 0.0


def test_positive_wheel_rotation_matches_vertical_and_horizontal_motion():
    tangential_speed = torch.tensor([0.2, 0.2])
    raw_angular_speed = torch.full((2,), 0.2 / WHEEL_RADIUS)
    error = tangent_rolling_error(
        tangential_speed,
        raw_angular_speed,
        WHEEL_RADIUS,
        torch.ones(2),
        torch.ones(2),
    )
    assert torch.allclose(error, torch.zeros(2), atol=1e-6)

    reverse_error = tangent_rolling_error(
        tangential_speed,
        -raw_angular_speed,
        WHEEL_RADIUS,
        torch.ones(2),
        torch.ones(2),
    )
    assert torch.allclose(reverse_error, torch.full((2,), 0.4), atol=1e-6)


def test_completion_requires_two_support_steps_and_timeout_is_separate():
    common = dict(
        progress_threshold=0.95,
        end_tolerance=0.04,
        support_force_threshold=20.0,
        required_support_steps=2,
        minimum_duration=0.10,
        timeout_base=0.60,
        timeout_height_gain=5.0,
    )
    support_steps, completed, timed_out = wheel_trajectory_completion(
        progress=torch.tensor([0.96]),
        end_error=torch.tensor([0.03]),
        vertical_force=torch.tensor([25.0]),
        support_steps=torch.tensor([1]),
        elapsed_time=torch.tensor([0.11]),
        step_height=torch.tensor([0.05]),
        **common,
    )
    assert support_steps.item() == 2
    assert completed.item()
    assert not timed_out.item()

    _, completed, timed_out = wheel_trajectory_completion(
        progress=torch.tensor([0.5]),
        end_error=torch.tensor([0.10]),
        vertical_force=torch.tensor([0.0]),
        support_steps=torch.tensor([0]),
        elapsed_time=torch.tensor([0.86]),
        step_height=torch.tensor([0.05]),
        **common,
    )
    assert not completed.item()
    assert timed_out.item()


def test_histogram_quantile_uses_upper_bin_edge():
    histogram = torch.tensor([1.0, 3.0, 6.0])
    assert torch.allclose(histogram_quantile(histogram, 0.5, 0.01), torch.tensor(0.03))


def _nested_class(root, *names):
    current = root
    for name in names:
        current = next(
            node
            for node in current.body
            if isinstance(node, ast.ClassDef) and node.name == name
        )
    return current


def _literal_assignments(class_node):
    values = {}
    for node in class_node.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name):
                try:
                    values[target.id] = ast.literal_eval(node.value)
                except (ValueError, TypeError):
                    pass
    return values


def test_task_config_keeps_actor_abi_and_new_reward_contract():
    config_path = (
        REPO_ROOT
        / "wheel_legged_gym/envs/l5a_2wheel_upstairs_cp_liang/l5a_2wheel_config.py"
    )
    tree = ast.parse(config_path.read_text())
    root = _nested_class(tree, "L5A_2WHEEL_Cfg")

    env_node = _nested_class(root, "env")
    env_values = {"num_actions": 8}
    for node in env_node.body:
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            env_values[node.targets[0].id] = eval(
                compile(ast.Expression(node.value), str(config_path), "eval"),
                {},
                env_values,
            )
    assert env_values["num_actions"] == 8
    assert env_values["num_observations"] == 32
    assert env_values["obs_history_length"] == 10
    assert env_values["num_observations"] * env_values["obs_history_length"] == 320

    terrain_values = _literal_assignments(_nested_class(root, "terrain"))
    assert terrain_values["num_rows"] == 10
    assert terrain_values["max_init_terrain_level"] == 2
    assert terrain_values["obstacle_height_min"] == 0.02
    assert terrain_values["obstacle_height_max"] == 0.16

    trajectory_values = _literal_assignments(
        _nested_class(root, "wheel_trajectory")
    )
    assert trajectory_values["num_samples"] == 33
    assert trajectory_values["wheel_forward_sign"] == [1.0, 1.0]

    reward_values = _literal_assignments(_nested_class(root, "rewards", "scales"))
    for reward_name in (
        "feet_contact_number",
        "feet_clearance",
        "swing_foot_lift",
        "triggered_leg_up_vel",
        "triggered_leg_action_dir",
        "wheel_zero_velocity",
    ):
        assert reward_values[reward_name] == 0
    assert reward_values["wheel_center_trajectory"] == 4.0
    assert reward_values["wheel_center_progress"] == 2.0
    assert reward_values["wheel_tangent_roll"] == 3.0
    assert reward_values["guided_wheel_contact"] == 1.5
    assert reward_values["selected_wheel_excess_force"] == -2.0

    ppo_root = _nested_class(tree, "L5A_2WHEEL_CfgPPO")
    runner_values = _literal_assignments(_nested_class(ppo_root, "runner"))
    assert runner_values["run_name"] == "wheel_center_trajectory"
    assert runner_values["resume"] is False
