import ast
import importlib.util
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import torch


MODULE_PATH = Path(__file__).parents[1] / "utils" / "stair_stability.py"
SPEC = importlib.util.spec_from_file_location("stair_stability", MODULE_PATH)
stair_stability = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(stair_stability)
REPO_ROOT = Path(__file__).parents[2]


def _xml_signature(element):
    return (
        element.tag,
        tuple(sorted(element.attrib.items())),
        (element.text or "").strip(),
        tuple(_xml_signature(child) for child in element),
    )


def test_curriculum_height_has_exact_endpoints():
    heights = [
        stair_stability.curriculum_obstacle_height(i / 10, 10, 0.02, 0.16)
        for i in range(10)
    ]
    assert heights[0] == 0.02
    assert heights[-1] == 0.16
    assert torch.allclose(
        torch.tensor(heights), torch.linspace(0.02, 0.16, 10), atol=1e-7
    )


def test_height_velocity_envelope():
    heights = torch.tensor([0.0, 0.05, 0.10, 0.16])
    expected = torch.tensor([1.0, 1.0, 0.7, 0.4])
    assert torch.allclose(stair_stability.max_forward_velocity(heights), expected)


def test_command_bounds_preserve_reverse_only_through_ten_centimeters():
    heights = torch.tensor([0.05, 0.10, 0.16])
    configured_min = torch.full((3,), -0.6)
    configured_max = torch.ones(3)
    lower, upper = stair_stability.command_velocity_bounds(
        heights, configured_min, configured_max, 0.10, 0.10
    )
    assert torch.allclose(lower, torch.tensor([-0.6, -0.6, 0.1]))
    assert torch.allclose(upper, torch.tensor([1.0, 0.7, 0.4]))


def test_python_action_limiter_matches_deployment_example():
    previous = torch.tensor([[0.0, 0.1, -0.1, 19.5, 0.0, -0.1, 0.1, -19.5]])
    raw = torch.tensor([[1.0, -1.0, 0.0, 30.0, -1.0, 1.0, 0.0, -30.0]])
    limited = stair_stability.limit_l5a_actions(
        raw, previous, [0, 1, 2, 4, 5, 6], [3, 7], 0.2, 1.0, 20.0
    )
    expected = torch.tensor([[0.2, -0.1, 0.0, 20.0, -0.2, 0.1, 0.0, -20.0]])
    assert torch.allclose(limited, expected)


def _make_state_machine():
    return stair_stability.StairStateMachine(
        num_envs=1,
        device=torch.device("cpu"),
        contact_confirm_min_time=0.10,
        contact_confirm_max_time=0.45,
        contact_confirm_stable_steps=3,
        roll_duration_base=0.35,
        roll_duration_height_gain=3.0,
        roll_height_margin=0.01,
        roll_timeout_margin=0.35,
        contact_loss_grace_steps=2,
        crest_height_margin=0.015,
        crest_entry_forward_ratio=0.6,
        crest_entry_support_ratio=0.6,
        crest_entry_support_steps=2,
        crest_min_time=0.12,
        crest_max_time=0.50,
        crest_finish_forward_ratio=1.2,
        crest_finish_vertical_force=20.0,
        crest_finish_stable_steps=3,
        recover_time=0.25,
        cooldown_time=0.25,
        wheel_radius=0.127,
    )


def _advance(
    sm,
    need_roll=False,
    candidate=0,
    stable=False,
    left_position=(0.0, 0.0, 0.0),
    right_position=(0.0, 0.0, 0.0),
    contact=True,
    support_ratio=0.0,
    vertical_force=0.0,
    vz=0.0,
):
    return sm.step(
        need_roll=torch.tensor([need_roll]),
        candidate_leg=torch.tensor([candidate]),
        dynamics_stable=torch.tensor([stable]),
        wheel_positions=torch.tensor([[left_position, right_position]]),
        base_forward_direction=torch.tensor([[1.0, 0.0, 0.0]]),
        contact_active=torch.tensor([[contact, contact]]),
        vertical_support_ratio=torch.tensor(
            [[support_ratio, support_ratio]], dtype=torch.float
        ),
        vertical_contact_force=torch.tensor(
            [[vertical_force, vertical_force]], dtype=torch.float
        ),
        wheel_vertical_velocity=torch.tensor([[vz, vz]], dtype=torch.float),
        step_height=torch.tensor([0.05]),
        dt=0.02,
    )


def _advance_until_roll(sm, candidate=1):
    events = _advance(sm, need_roll=True, candidate=candidate, stable=True)
    assert events["start_contact_confirm"].item()
    for _ in range(8):
        events = _advance(sm, need_roll=True, candidate=1 - candidate, stable=True)
        if events["start_roll"].item():
            return events
    raise AssertionError("state machine did not enter ROLL_UP")


def test_contact_tangent_rotates_from_riser_to_tread():
    forces = torch.tensor(
        [
            [-20.0, 0.0, 0.0],
            [0.0, 0.0, 20.0],
            [-20.0, 0.0, 20.0],
        ]
    )
    tangent = stair_stability.contact_tangent(forces)
    sqrt_half = 2.0**-0.5
    expected = torch.tensor(
        [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [sqrt_half, 0.0, sqrt_half]]
    )
    assert torch.allclose(tangent, expected, atol=1e-6)


def test_rolling_error_uses_signed_positive_wheel_speed():
    wheel_velocity = torch.tensor([[0.2, 0.0, 0.0], [0.0, 0.0, 0.2]])
    contact_force = torch.tensor([[0.0, 0.0, 20.0], [-20.0, 0.0, 0.0]])
    wheel_velocity_rad = torch.tensor([0.2 / 0.127, 0.2 / 0.127])
    _, tangent_speed, error = stair_stability.rolling_contact_kinematics(
        wheel_velocity, contact_force, wheel_velocity_rad, 0.127
    )
    assert torch.allclose(tangent_speed, torch.tensor([0.2, 0.2]))
    assert torch.allclose(error, torch.zeros(2), atol=1e-6)


def test_minimum_jerk_roll_reference_has_zero_endpoint_velocity():
    height = torch.tensor([0.02, 0.16])
    duration = 0.35 + 3.0 * height
    actual_duration, start_height, start_velocity = stair_stability.roll_reference(
        torch.zeros(2), height, 0.01, 0.35, 3.0
    )
    _, end_height, end_velocity = stair_stability.roll_reference(
        duration, height, 0.01, 0.35, 3.0
    )
    assert torch.allclose(actual_duration, torch.tensor([0.41, 0.83]))
    assert torch.allclose(start_height, torch.zeros(2))
    assert torch.allclose(start_velocity, torch.zeros(2))
    assert torch.allclose(end_height, height + 0.01)
    assert torch.allclose(end_velocity, torch.zeros(2), atol=1e-6)


def test_bevel_changes_collision_vertices_but_not_logical_heights():
    height_field = np.array([[0, 0], [10, 10]], dtype=np.int16)
    sharp_vertices, sharp_triangles = (
        stair_stability.heightfield_to_trimesh_with_bevel(
            height_field, 0.05, 0.005, 0.1, np.zeros((2, 2))
        )
    )
    bevel_vertices, bevel_triangles = (
        stair_stability.heightfield_to_trimesh_with_bevel(
            height_field, 0.05, 0.005, 0.1, np.full((2, 2), 0.01)
        )
    )
    sharp_grid = sharp_vertices.reshape(2, 2, 3)
    bevel_grid = bevel_vertices.reshape(2, 2, 3)
    assert np.allclose(sharp_grid[0, :, 0], 0.05)
    assert np.allclose(bevel_grid[0, :, 0], 0.04)
    assert np.allclose(bevel_grid[1, :, 0], 0.05)
    assert np.array_equal(sharp_vertices[:, 2], bevel_vertices[:, 2])
    assert np.array_equal(sharp_triangles, bevel_triangles)


def test_task_specific_wheel_collision_is_cylindrical_and_axes_are_positive_y():
    urdf_path = (
        REPO_ROOT
        / "resources/robots/l5a/urdf/l5aurdf20260521_roll_contact.urdf"
    )
    root = ET.parse(urdf_path).getroot()
    shared_root = ET.parse(
        REPO_ROOT / "resources/robots/l5a/urdf/l5aurdf20260521.urdf"
    ).getroot()
    for side, y_offset in (("left", "0.035"), ("right", "-0.035")):
        link = root.find(f"./link[@name='{side}_wheel_link']")
        shared_link = shared_root.find(f"./link[@name='{side}_wheel_link']")
        assert _xml_signature(link.find("visual")) == _xml_signature(
            shared_link.find("visual")
        )
        assert _xml_signature(link.find("inertial")) == _xml_signature(
            shared_link.find("inertial")
        )
        collisions = link.findall("collision")
        assert len(collisions) == 1
        cylinder = collisions[0].find("geometry/cylinder")
        assert cylinder is not None
        assert float(cylinder.attrib["radius"]) == 0.127
        assert float(cylinder.attrib["length"]) == 0.04
        assert collisions[0].find("geometry/mesh") is None
        assert collisions[0].find("origin").attrib["xyz"] == f"0 {y_offset} 0"

        joint = root.find(f"./joint[@name='{side}_wheel_joint']")
        assert joint.find("axis").attrib["xyz"] == "0 1 0"


def test_actor_observation_layout_keeps_32_by_10_to_8_abi():
    slices = stair_stability.L5A_OBSERVATION_SLICES
    assert stair_stability.L5A_OBSERVATION_DIM == 32
    assert slices == {
        "angular_velocity": slice(0, 3),
        "projected_gravity": slice(3, 6),
        "command": slice(6, 10),
        "leg_joint_position": slice(10, 16),
        "all_joint_velocity": slice(16, 24),
        "previous_action": slice(24, 32),
    }
    covered = []
    for observation_slice in slices.values():
        covered.extend(range(observation_slice.start, observation_slice.stop))
    assert covered == list(range(32))

    config_path = (
        REPO_ROOT
        / "wheel_legged_gym/envs/l5a_2wheel_upstairs_cp_liang/l5a_2wheel_config.py"
    )
    config_tree = ast.parse(config_path.read_text())
    env_class = next(
        node
        for node in ast.walk(config_tree)
        if isinstance(node, ast.ClassDef) and node.name == "env"
    )
    values = {}
    for node in env_class.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name):
                values[target.id] = eval(
                    compile(ast.Expression(node.value), str(config_path), "eval"),
                    {},
                    values,
                )
    assert values["num_actions"] == 8
    assert values["num_observations"] == 32
    assert values["obs_history_length"] == 10
    assert values["num_observations"] * values["obs_history_length"] == 320


def test_state_machine_locks_leg_rolls_over_crest_and_recovers():
    sm = _make_state_machine()
    events = _advance_until_roll(sm, candidate=1)
    assert sm.current_leg.item() == 1
    assert sm.state.item() == stair_stability.STATE_ROLL_UP
    assert events["reference_height"].item() == 0.0
    assert events["reference_forward"].item() == 0.0

    for _ in range(2):
        events = _advance(
            sm,
            need_roll=True,
            candidate=0,
            stable=True,
            right_position=(0.08, 0.0, 0.04),
            support_ratio=0.7,
            vertical_force=25.0,
        )
        assert sm.current_leg.item() == 1
    assert events["start_crest"].item()
    assert sm.state.item() == stair_stability.STATE_CREST

    for _ in range(8):
        events = _advance(
            sm,
            need_roll=True,
            candidate=0,
            stable=True,
            right_position=(0.16, 0.0, 0.05),
            support_ratio=0.9,
            vertical_force=30.0,
        )
        if events["crest_finished"].item():
            break
    assert events["crest_finished"].item()
    assert sm.state.item() == stair_stability.STATE_RECOVER

    for _ in range(13):
        _advance(sm, need_roll=False)
    assert sm.state.item() == stair_stability.STATE_APPROACH


def test_unstable_contact_confirmation_times_out_without_roll():
    sm = _make_state_machine()
    _advance(sm, need_roll=True, stable=False)
    events = None
    for _ in range(24):
        events = _advance(sm, need_roll=True, stable=False)
        if events["contact_confirm_timeout"].item():
            break
    assert events["contact_confirm_timeout"].item()
    assert not events["start_roll"].item()
    assert sm.state.item() == stair_stability.STATE_COOLDOWN


def test_roll_allows_two_lost_contact_steps_then_cools_down():
    sm = _make_state_machine()
    _advance_until_roll(sm)

    for expected_loss_steps in (1, 2):
        events = _advance(sm, need_roll=True, stable=True, contact=False)
        assert events["contact_loss_steps"].item() == expected_loss_steps
        assert not events["roll_contact_lost"].item()
        assert sm.state.item() == stair_stability.STATE_ROLL_UP

    events = _advance(sm, need_roll=True, stable=True, contact=False)
    assert events["roll_contact_lost"].item()
    assert sm.state.item() == stair_stability.STATE_COOLDOWN

    events = _advance(sm, need_roll=True, stable=True)
    assert events["retrigger"].item()
