"""Pure helpers for the L5A upstairs curriculum and control post-processing.

This module intentionally has no Isaac Gym dependency so the numerical rules can
be unit-tested without creating a simulator.
"""

from __future__ import annotations

from typing import Dict, Iterable

import numpy as np
import torch


STATE_APPROACH = 0
STATE_SETTLE = 1
STATE_SWING = 2
STATE_COOLDOWN = 3
NUM_STAIR_STATES = 4


def curriculum_obstacle_height(
    difficulty: float,
    num_rows: int,
    min_height: float,
    max_height: float,
) -> float:
    """Map the repository's ``row / num_rows`` difficulty to exact endpoints."""
    if num_rows <= 1:
        return float(min_height)
    max_curriculum_difficulty = (num_rows - 1) / num_rows
    level_ratio = np.clip(difficulty / max_curriculum_difficulty, 0.0, 1.0)
    return float(min_height + (max_height - min_height) * level_ratio)


def max_forward_velocity(step_height: torch.Tensor) -> torch.Tensor:
    """Conservative maximum command envelope for 0--16 cm obstacles."""
    height = torch.clamp(step_height, min=0.0, max=0.16)
    low_to_mid = 1.0 - 6.0 * (height - 0.05)
    mid_to_high = 0.7 - 5.0 * (height - 0.10)
    return torch.where(
        height <= 0.05,
        torch.ones_like(height),
        torch.where(height <= 0.10, low_to_mid, mid_to_high),
    )


def command_velocity_bounds(
    step_height: torch.Tensor,
    configured_min: torch.Tensor,
    configured_max: torch.Tensor,
    forward_only_threshold: float,
    min_forward_velocity: float,
) -> tuple:
    """Return per-environment sampling bounds for the height-speed envelope."""
    forward_max = torch.minimum(configured_max, max_forward_velocity(step_height))
    forward_only = step_height > forward_only_threshold
    lower_bound = torch.where(
        forward_only,
        torch.full_like(forward_max, min_forward_velocity),
        configured_min,
    )
    return torch.minimum(lower_bound, forward_max), forward_max


def limit_l5a_actions(
    raw_actions: torch.Tensor,
    previous_actions: torch.Tensor,
    joint_indices: Iterable[int],
    wheel_indices: Iterable[int],
    joint_delta_limit: float,
    wheel_delta_limit: float,
    wheel_abs_limit: float,
) -> torch.Tensor:
    """Apply the same per-policy-step limits used by the C++ deployment path."""
    limited = raw_actions.clone()
    joint_indices = list(joint_indices)
    wheel_indices = list(wheel_indices)

    joint_delta = torch.clamp(
        raw_actions[:, joint_indices] - previous_actions[:, joint_indices],
        -joint_delta_limit,
        joint_delta_limit,
    )
    limited[:, joint_indices] = previous_actions[:, joint_indices] + joint_delta

    wheel_delta = torch.clamp(
        raw_actions[:, wheel_indices] - previous_actions[:, wheel_indices],
        -wheel_delta_limit,
        wheel_delta_limit,
    )
    limited[:, wheel_indices] = previous_actions[:, wheel_indices] + wheel_delta
    limited[:, wheel_indices] = torch.clamp(
        limited[:, wheel_indices], -wheel_abs_limit, wheel_abs_limit
    )
    return limited


class StairStateMachine:
    """Batched APPROACH -> SETTLE -> SWING -> COOLDOWN state machine."""

    def __init__(
        self,
        num_envs: int,
        device: torch.device,
        settle_min_time: float,
        settle_max_time: float,
        settle_stable_steps: int,
        swing_min_time: float,
        swing_timeout_base: float,
        swing_timeout_height_gain: float,
        swing_height_margin: float,
        swing_finish_vertical_velocity: float,
        cooldown_time: float,
    ) -> None:
        self.num_envs = num_envs
        self.device = device
        self.settle_min_time = settle_min_time
        self.settle_max_time = settle_max_time
        self.settle_stable_steps_required = settle_stable_steps
        self.swing_min_time = swing_min_time
        self.swing_timeout_base = swing_timeout_base
        self.swing_timeout_height_gain = swing_timeout_height_gain
        self.swing_height_margin = swing_height_margin
        self.swing_finish_vertical_velocity = swing_finish_vertical_velocity
        self.cooldown_time = cooldown_time

        self.state = torch.full(
            (num_envs,), STATE_APPROACH, dtype=torch.long, device=device
        )
        self.state_time = torch.zeros(num_envs, dtype=torch.float, device=device)
        self.stable_steps = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.current_leg = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.cooldown_retrigger_latched = torch.zeros(
            num_envs, dtype=torch.bool, device=device
        )

    def reset(self, env_ids: torch.Tensor) -> None:
        self.state[env_ids] = STATE_APPROACH
        self.state_time[env_ids] = 0.0
        self.stable_steps[env_ids] = 0
        self.current_leg[env_ids] = 0
        self.cooldown_retrigger_latched[env_ids] = False

    def step(
        self,
        need_swing: torch.Tensor,
        candidate_leg: torch.Tensor,
        dynamics_stable: torch.Tensor,
        foot_clearance: torch.Tensor,
        foot_vertical_velocity: torch.Tensor,
        step_height: torch.Tensor,
        dt: float,
    ) -> Dict[str, torch.Tensor]:
        """Advance all environments once and return one-step transition events."""
        time_epsilon = 1e-6
        self.state_time += dt
        swing_before = self.state == STATE_SWING
        cooldown_before = self.state == STATE_COOLDOWN

        start_settle = (self.state == STATE_APPROACH) & need_swing
        self.current_leg[start_settle] = candidate_leg[start_settle]
        self.state[start_settle] = STATE_SETTLE
        self.state_time[start_settle] = 0.0
        self.stable_steps[start_settle] = 0

        in_settle = self.state == STATE_SETTLE
        self.stable_steps = torch.where(
            in_settle & dynamics_stable,
            self.stable_steps + 1,
            torch.zeros_like(self.stable_steps),
        )
        settle_elapsed = self.state_time.clone()
        start_swing = (
            in_settle
            & (self.state_time + time_epsilon >= self.settle_min_time)
            & (self.stable_steps >= self.settle_stable_steps_required)
        )
        settle_timeout = (
            in_settle
            & ~start_swing
            & (self.state_time + time_epsilon >= self.settle_max_time)
        )

        self.state[start_swing] = STATE_SWING
        self.state_time[start_swing] = 0.0
        self.stable_steps[start_swing] = 0
        self.state[settle_timeout] = STATE_COOLDOWN
        self.state_time[settle_timeout] = 0.0
        self.stable_steps[settle_timeout] = 0
        self.cooldown_retrigger_latched[settle_timeout] = False

        env_ids = torch.arange(self.num_envs, device=self.device)
        selected_clearance = foot_clearance[env_ids, self.current_leg]
        selected_vertical_velocity = foot_vertical_velocity[env_ids, self.current_leg]
        target_clearance = step_height + self.swing_height_margin
        swing_timeout_limit = (
            self.swing_timeout_base + self.swing_timeout_height_gain * step_height
        )

        # Only states that were already swinging accrue swing completion events;
        # a SETTLE -> SWING transition starts at exactly t=0.
        swing_finished = (
            swing_before
            & (self.state_time + time_epsilon >= self.swing_min_time)
            & (selected_clearance >= target_clearance)
            & (
                torch.abs(selected_vertical_velocity)
                <= self.swing_finish_vertical_velocity
            )
        )
        swing_timeout = (
            swing_before
            & ~swing_finished
            & (self.state_time + time_epsilon >= swing_timeout_limit)
        )
        leave_swing = swing_finished | swing_timeout
        self.state[leave_swing] = STATE_COOLDOWN
        self.state_time[leave_swing] = 0.0
        self.cooldown_retrigger_latched[leave_swing] = False

        retrigger = (
            cooldown_before & need_swing & ~self.cooldown_retrigger_latched
        )
        self.cooldown_retrigger_latched[retrigger] = True
        cooldown_finished = cooldown_before & (
            self.state_time + time_epsilon >= self.cooldown_time
        )
        self.state[cooldown_finished] = STATE_APPROACH
        self.state_time[cooldown_finished] = 0.0
        self.cooldown_retrigger_latched[cooldown_finished] = False

        return {
            "start_settle": start_settle,
            "start_swing": start_swing,
            "settle_timeout": settle_timeout,
            "swing_finished": swing_finished,
            "swing_timeout": swing_timeout,
            "retrigger": retrigger,
            "settle_elapsed": settle_elapsed,
            "selected_clearance": selected_clearance,
            "selected_vertical_velocity": selected_vertical_velocity,
            "target_clearance": target_clearance,
        }
