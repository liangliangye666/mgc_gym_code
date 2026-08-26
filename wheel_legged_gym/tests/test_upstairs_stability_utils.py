import importlib.util
from pathlib import Path

import torch


MODULE_PATH = Path(__file__).parents[1] / "utils" / "stair_stability.py"
SPEC = importlib.util.spec_from_file_location("stair_stability", MODULE_PATH)
stair_stability = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(stair_stability)


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
        settle_min_time=0.10,
        settle_max_time=0.60,
        settle_stable_steps=3,
        swing_min_time=0.16,
        swing_timeout_base=0.40,
        swing_timeout_height_gain=2.0,
        swing_height_margin=0.02,
        swing_finish_vertical_velocity=0.10,
        cooldown_time=0.20,
    )


def _advance(sm, need_swing=False, candidate=0, stable=False, clearance=0.0, vz=0.0):
    return sm.step(
        need_swing=torch.tensor([need_swing]),
        candidate_leg=torch.tensor([candidate]),
        dynamics_stable=torch.tensor([stable]),
        foot_clearance=torch.tensor([[clearance, clearance]]),
        foot_vertical_velocity=torch.tensor([[vz, vz]]),
        step_height=torch.tensor([0.05]),
        dt=0.02,
    )


def test_state_machine_locks_leg_settles_swings_and_cools_down():
    sm = _make_state_machine()
    events = _advance(sm, need_swing=True, candidate=1, stable=True)
    assert events["start_settle"].item()
    assert sm.current_leg.item() == 1

    for _ in range(4):
        _advance(sm, need_swing=True, candidate=0, stable=True)
        assert sm.current_leg.item() == 1
    events = _advance(sm, need_swing=True, candidate=0, stable=True)
    assert events["start_swing"].item()
    assert sm.state.item() == stair_stability.STATE_SWING

    for _ in range(8):
        events = _advance(sm, need_swing=True, candidate=0, stable=True, clearance=0.07)
    assert events["swing_finished"].item()
    assert sm.state.item() == stair_stability.STATE_COOLDOWN

    events = _advance(sm, need_swing=True, candidate=0, stable=True)
    assert events["retrigger"].item()
    for _ in range(9):
        _advance(sm, need_swing=True, candidate=0, stable=True)
    assert sm.state.item() == stair_stability.STATE_APPROACH


def test_unstable_settle_times_out_without_swing():
    sm = _make_state_machine()
    _advance(sm, need_swing=True, stable=False)
    events = None
    for _ in range(30):
        events = _advance(sm, need_swing=True, stable=False)
        if events["settle_timeout"].item():
            break
    assert events["settle_timeout"].item()
    assert not events["start_swing"].item()
    assert sm.state.item() == stair_stability.STATE_COOLDOWN
