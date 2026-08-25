"""Pure helpers for the L5A upstairs curriculum and control post-processing.

This module intentionally has no Isaac Gym dependency so the numerical rules can
be unit-tested without creating a simulator.
"""

from __future__ import annotations

from typing import Dict, Iterable

import numpy as np
import torch


STATE_APPROACH = 0
STATE_CONTACT_CONFIRM = 1
STATE_ROLL_UP = 2
STATE_CREST = 3
STATE_RECOVER = 4
STATE_COOLDOWN = 5
NUM_STAIR_STATES = 6
L5A_OBSERVATION_DIM = 32
L5A_OBSERVATION_SLICES = {
    "angular_velocity": slice(0, 3),
    "projected_gravity": slice(3, 6),
    "command": slice(6, 10),
    "leg_joint_position": slice(10, 16),
    "all_joint_velocity": slice(16, 24),
    "previous_action": slice(24, 32),
}


def heightfield_to_trimesh_with_bevel(
    height_field_raw: np.ndarray,
    horizontal_scale: float,
    vertical_scale: float,
    slope_threshold: float = None,
    edge_bevel_map: np.ndarray = None,
) -> tuple:
    """Convert a height field to a mesh with an optional steep-edge bevel.

    The logical height samples remain untouched.  For a steep edge, Isaac Gym's
    original conversion moves the lower vertex by one complete horizontal grid
    cell to create a vertical face.  Here that movement is shortened by the
    requested bevel depth, producing a narrow sloped collision face while
    preserving the same tread heights.
    """
    height_field = np.asarray(height_field_raw)
    num_rows, num_cols = height_field.shape
    y = np.linspace(0, (num_cols - 1) * horizontal_scale, num_cols)
    x = np.linspace(0, (num_rows - 1) * horizontal_scale, num_rows)
    yy, xx = np.meshgrid(y, x)

    if slope_threshold is not None:
        threshold_raw = slope_threshold * horizontal_scale / vertical_scale
        move_x = np.zeros((num_rows, num_cols))
        move_y = np.zeros((num_rows, num_cols))
        move_corners = np.zeros((num_rows, num_cols))
        move_x[: num_rows - 1, :] += (
            height_field[1:num_rows, :] - height_field[: num_rows - 1, :]
            > threshold_raw
        )
        move_x[1:num_rows, :] -= (
            height_field[: num_rows - 1, :] - height_field[1:num_rows, :]
            > threshold_raw
        )
        move_y[:, : num_cols - 1] += (
            height_field[:, 1:num_cols] - height_field[:, : num_cols - 1]
            > threshold_raw
        )
        move_y[:, 1:num_cols] -= (
            height_field[:, : num_cols - 1] - height_field[:, 1:num_cols]
            > threshold_raw
        )
        move_corners[: num_rows - 1, : num_cols - 1] += (
            height_field[1:num_rows, 1:num_cols]
            - height_field[: num_rows - 1, : num_cols - 1]
            > threshold_raw
        )
        move_corners[1:num_rows, 1:num_cols] -= (
            height_field[: num_rows - 1, : num_cols - 1]
            - height_field[1:num_rows, 1:num_cols]
            > threshold_raw
        )
        move_x_total = move_x + move_corners * (move_x == 0)
        move_y_total = move_y + move_corners * (move_y == 0)

        if edge_bevel_map is None:
            bevel = np.zeros_like(height_field, dtype=np.float64)
        else:
            bevel = np.asarray(edge_bevel_map, dtype=np.float64)
            if bevel.shape != height_field.shape:
                raise ValueError("edge_bevel_map must match height_field_raw shape")
            bevel = np.clip(bevel, 0.0, max(horizontal_scale - 1e-6, 0.0))
        vertex_shift = horizontal_scale - bevel
        xx += move_x_total * vertex_shift
        yy += move_y_total * vertex_shift

    vertices = np.zeros((num_rows * num_cols, 3), dtype=np.float32)
    vertices[:, 0] = xx.flatten()
    vertices[:, 1] = yy.flatten()
    vertices[:, 2] = height_field.flatten() * vertical_scale
    triangles = -np.ones(
        (2 * (num_rows - 1) * (num_cols - 1), 3), dtype=np.uint32
    )
    for row in range(num_rows - 1):
        ind0 = np.arange(0, num_cols - 1) + row * num_cols
        ind1 = ind0 + 1
        ind2 = ind0 + num_cols
        ind3 = ind2 + 1
        start = 2 * row * (num_cols - 1)
        stop = start + 2 * (num_cols - 1)
        triangles[start:stop:2, 0] = ind0
        triangles[start:stop:2, 1] = ind3
        triangles[start:stop:2, 2] = ind1
        triangles[start + 1 : stop : 2, 0] = ind0
        triangles[start + 1 : stop : 2, 1] = ind2
        triangles[start + 1 : stop : 2, 2] = ind3
    return vertices, triangles


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


def minimum_jerk_profile(progress: torch.Tensor) -> tuple:
    """Return a smooth 0--1 position profile and its derivative in normalized time."""
    progress = torch.clamp(progress, 0.0, 1.0)
    position = 10.0 * progress**3 - 15.0 * progress**4 + 6.0 * progress**5
    velocity = 30.0 * progress**2 * (1.0 - progress) ** 2
    return position, velocity


def roll_reference(
    elapsed_time: torch.Tensor,
    step_height: torch.Tensor,
    height_margin: float,
    duration_base: float,
    duration_height_gain: float,
) -> tuple:
    """Return roll-up duration, wheel-center lift and vertical-speed references."""
    duration = duration_base + duration_height_gain * step_height
    progress = torch.clamp(elapsed_time / torch.clamp(duration, min=1e-6), 0.0, 1.0)
    position_profile, velocity_profile = minimum_jerk_profile(progress)
    target_lift = step_height + height_margin
    height_reference = target_lift * position_profile
    vertical_velocity_reference = (
        target_lift * velocity_profile / torch.clamp(duration, min=1e-6)
    )
    return duration, height_reference, vertical_velocity_reference


def contact_tangent(contact_force_base: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Construct the forward/up tangent implied by an x-z contact force.

    A backward force on a wheel at a stair riser produces a +z tangent. A
    vertical support force on a tread produces a +x tangent.
    """
    tangent = torch.zeros_like(contact_force_base)
    tangent[..., 0] = contact_force_base[..., 2]
    tangent[..., 2] = -contact_force_base[..., 0]
    force_xz = torch.linalg.vector_norm(contact_force_base[..., (0, 2)], dim=-1)
    valid = force_xz > eps
    tangent = tangent / torch.clamp(force_xz.unsqueeze(-1), min=eps)
    return tangent * valid.unsqueeze(-1)


def rolling_contact_kinematics(
    wheel_velocity_base: torch.Tensor,
    contact_force_base: torch.Tensor,
    wheel_angular_velocity: torch.Tensor,
    wheel_radius: float,
    forward_wheel_sign: float = 1.0,
) -> tuple:
    """Return contact tangent, wheel-center tangent speed and no-slip error."""
    tangent = contact_tangent(contact_force_base)
    tangent_speed = torch.sum(wheel_velocity_base * tangent, dim=-1)
    rim_speed = forward_wheel_sign * wheel_radius * wheel_angular_velocity
    return tangent, tangent_speed, tangent_speed - rim_speed


class StairStateMachine:
    """Batched contact-guided stair rolling state machine."""

    def __init__(
        self,
        num_envs: int,
        device: torch.device,
        contact_confirm_min_time: float,
        contact_confirm_max_time: float,
        contact_confirm_stable_steps: int,
        roll_duration_base: float,
        roll_duration_height_gain: float,
        roll_height_margin: float,
        roll_timeout_margin: float,
        contact_loss_grace_steps: int,
        crest_height_margin: float,
        crest_entry_forward_ratio: float,
        crest_entry_support_ratio: float,
        crest_entry_support_steps: int,
        crest_min_time: float,
        crest_max_time: float,
        crest_finish_forward_ratio: float,
        crest_finish_vertical_force: float,
        crest_finish_stable_steps: int,
        recover_time: float,
        cooldown_time: float,
        wheel_radius: float,
    ) -> None:
        self.num_envs = num_envs
        self.device = device
        self.contact_confirm_min_time = contact_confirm_min_time
        self.contact_confirm_max_time = contact_confirm_max_time
        self.contact_confirm_stable_steps_required = contact_confirm_stable_steps
        self.roll_duration_base = roll_duration_base
        self.roll_duration_height_gain = roll_duration_height_gain
        self.roll_height_margin = roll_height_margin
        self.roll_timeout_margin = roll_timeout_margin
        self.contact_loss_grace_steps = contact_loss_grace_steps
        self.crest_height_margin = crest_height_margin
        self.crest_entry_forward_ratio = crest_entry_forward_ratio
        self.crest_entry_support_ratio = crest_entry_support_ratio
        self.crest_entry_support_steps_required = crest_entry_support_steps
        self.crest_min_time = crest_min_time
        self.crest_max_time = crest_max_time
        self.crest_finish_forward_ratio = crest_finish_forward_ratio
        self.crest_finish_vertical_force = crest_finish_vertical_force
        self.crest_finish_stable_steps_required = crest_finish_stable_steps
        self.recover_time = recover_time
        self.cooldown_time = cooldown_time
        self.wheel_radius = wheel_radius

        self.state = torch.full(
            (num_envs,), STATE_APPROACH, dtype=torch.long, device=device
        )
        self.state_time = torch.zeros(num_envs, dtype=torch.float, device=device)
        self.stable_steps = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.contact_loss_steps = torch.zeros(
            num_envs, dtype=torch.long, device=device
        )
        self.crest_entry_steps = torch.zeros(
            num_envs, dtype=torch.long, device=device
        )
        self.crest_finish_steps = torch.zeros(
            num_envs, dtype=torch.long, device=device
        )
        self.current_leg = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.start_wheel_position = torch.zeros(
            num_envs, 3, dtype=torch.float, device=device
        )
        self.start_forward_direction = torch.zeros_like(self.start_wheel_position)
        self.start_forward_direction[:, 0] = 1.0
        self.cooldown_retrigger_latched = torch.zeros(
            num_envs, dtype=torch.bool, device=device
        )

    def reset(self, env_ids: torch.Tensor) -> None:
        self.state[env_ids] = STATE_APPROACH
        self.state_time[env_ids] = 0.0
        self.stable_steps[env_ids] = 0
        self.contact_loss_steps[env_ids] = 0
        self.crest_entry_steps[env_ids] = 0
        self.crest_finish_steps[env_ids] = 0
        self.current_leg[env_ids] = 0
        self.start_wheel_position[env_ids] = 0.0
        self.start_forward_direction[env_ids] = 0.0
        self.start_forward_direction[env_ids, 0] = 1.0
        self.cooldown_retrigger_latched[env_ids] = False

    def step(
        self,
        need_roll: torch.Tensor,
        candidate_leg: torch.Tensor,
        dynamics_stable: torch.Tensor,
        wheel_positions: torch.Tensor,
        base_forward_direction: torch.Tensor,
        contact_active: torch.Tensor,
        vertical_support_ratio: torch.Tensor,
        vertical_contact_force: torch.Tensor,
        wheel_vertical_velocity: torch.Tensor,
        step_height: torch.Tensor,
        dt: float,
    ) -> Dict[str, torch.Tensor]:
        """Advance all environments once and return one-step transition events."""
        time_epsilon = 1e-6
        self.state_time += dt
        state_before = self.state.clone()
        roll_before = state_before == STATE_ROLL_UP
        crest_before = state_before == STATE_CREST
        recover_before = state_before == STATE_RECOVER
        cooldown_before = self.state == STATE_COOLDOWN

        start_contact_confirm = (self.state == STATE_APPROACH) & need_roll
        self.current_leg[start_contact_confirm] = candidate_leg[start_contact_confirm]
        env_ids = torch.arange(self.num_envs, device=self.device)
        selected_position = wheel_positions[env_ids, self.current_leg]
        horizontal_forward = base_forward_direction.clone()
        horizontal_forward[:, 2] = 0.0
        horizontal_forward /= torch.clamp(
            torch.linalg.vector_norm(horizontal_forward, dim=1, keepdim=True),
            min=1e-6,
        )
        self.state[start_contact_confirm] = STATE_CONTACT_CONFIRM
        self.state_time[start_contact_confirm] = 0.0
        self.stable_steps[start_contact_confirm] = 0
        self.contact_loss_steps[start_contact_confirm] = 0

        selected_contact = contact_active[env_ids, self.current_leg]
        selected_support_ratio = vertical_support_ratio[env_ids, self.current_leg]
        selected_vertical_force = vertical_contact_force[env_ids, self.current_leg]
        selected_vertical_velocity = wheel_vertical_velocity[env_ids, self.current_leg]

        in_contact_confirm = self.state == STATE_CONTACT_CONFIRM
        self.stable_steps = torch.where(
            in_contact_confirm & dynamics_stable & selected_contact,
            self.stable_steps + 1,
            torch.zeros_like(self.stable_steps),
        )
        contact_confirm_elapsed = self.state_time.clone()
        start_roll = (
            in_contact_confirm
            & (self.state_time + time_epsilon >= self.contact_confirm_min_time)
            & (
                self.stable_steps
                >= self.contact_confirm_stable_steps_required
            )
        )
        contact_confirm_timeout = (
            in_contact_confirm
            & ~start_roll
            & (self.state_time + time_epsilon >= self.contact_confirm_max_time)
        )

        # The candidate wheel is locked at first contact; the roll trajectory's
        # spatial origin and forward direction are locked when ROLL_UP starts.
        self.start_wheel_position[start_roll] = selected_position[start_roll]
        self.start_forward_direction[start_roll] = horizontal_forward[start_roll]
        self.state[start_roll] = STATE_ROLL_UP
        self.state_time[start_roll] = 0.0
        self.stable_steps[start_roll] = 0
        self.contact_loss_steps[start_roll] = 0
        self.crest_entry_steps[start_roll] = 0

        self.state[contact_confirm_timeout] = STATE_COOLDOWN
        self.state_time[contact_confirm_timeout] = 0.0
        self.stable_steps[contact_confirm_timeout] = 0
        self.cooldown_retrigger_latched[contact_confirm_timeout] = False

        selected_position = wheel_positions[env_ids, self.current_leg]
        displacement = selected_position - self.start_wheel_position
        height_progress = displacement[:, 2]
        forward_progress = torch.sum(
            displacement * self.start_forward_direction, dim=1
        )
        (
            roll_reference_duration,
            reference_height,
            reference_vertical_velocity,
        ) = roll_reference(
            self.state_time,
            step_height,
            self.roll_height_margin,
            self.roll_duration_base,
            self.roll_duration_height_gain,
        )
        normalized_roll_time = torch.clamp(
            self.state_time / torch.clamp(roll_reference_duration, min=1e-6),
            0.0,
            1.0,
        )
        reference_forward_profile, _ = minimum_jerk_profile(normalized_roll_time)
        reference_forward = (
            self.crest_entry_forward_ratio
            * self.wheel_radius
            * reference_forward_profile
        )
        roll_elapsed = self.state_time.clone()
        crest_elapsed = self.state_time.clone()

        active_roll_contact = roll_before | crest_before
        self.contact_loss_steps = torch.where(
            active_roll_contact & ~selected_contact,
            self.contact_loss_steps + 1,
            torch.where(
                active_roll_contact,
                torch.zeros_like(self.contact_loss_steps),
                self.contact_loss_steps,
            ),
        )

        crest_entry_ready = (
            roll_before
            & (
                height_progress
                >= torch.clamp(step_height - self.crest_height_margin, min=0.0)
            )
            & (forward_progress >= self.crest_entry_forward_ratio * self.wheel_radius)
            & (selected_support_ratio >= self.crest_entry_support_ratio)
        )
        self.crest_entry_steps = torch.where(
            crest_entry_ready,
            self.crest_entry_steps + 1,
            torch.where(
                roll_before,
                torch.zeros_like(self.crest_entry_steps),
                self.crest_entry_steps,
            ),
        )
        start_crest = roll_before & (
            self.crest_entry_steps >= self.crest_entry_support_steps_required
        )
        roll_contact_lost = (
            roll_before
            & ~start_crest
            & (self.contact_loss_steps > self.contact_loss_grace_steps)
        )
        roll_timeout = (
            roll_before
            & ~start_crest
            & ~roll_contact_lost
            & (
                self.state_time + time_epsilon
                >= roll_reference_duration + self.roll_timeout_margin
            )
        )

        self.state[start_crest] = STATE_CREST
        self.state_time[start_crest] = 0.0
        self.crest_finish_steps[start_crest] = 0
        self.contact_loss_steps[start_crest] = 0

        roll_failed = roll_contact_lost | roll_timeout
        self.state[roll_failed] = STATE_COOLDOWN
        self.state_time[roll_failed] = 0.0
        self.cooldown_retrigger_latched[roll_failed] = False

        crest_finish_ready = (
            crest_before
            & (forward_progress >= self.crest_finish_forward_ratio * self.wheel_radius)
            & (selected_vertical_force >= self.crest_finish_vertical_force)
        )
        self.crest_finish_steps = torch.where(
            crest_finish_ready,
            self.crest_finish_steps + 1,
            torch.where(
                crest_before,
                torch.zeros_like(self.crest_finish_steps),
                self.crest_finish_steps,
            ),
        )
        crest_finished = (
            crest_before
            & (self.state_time + time_epsilon >= self.crest_min_time)
            & (
                self.crest_finish_steps
                >= self.crest_finish_stable_steps_required
            )
        )
        crest_contact_lost = (
            crest_before
            & ~crest_finished
            & (self.contact_loss_steps > self.contact_loss_grace_steps)
        )
        crest_timeout = (
            crest_before
            & ~crest_finished
            & ~crest_contact_lost
            & (self.state_time + time_epsilon >= self.crest_max_time)
        )

        self.state[crest_finished] = STATE_RECOVER
        self.state_time[crest_finished] = 0.0
        crest_failed = crest_contact_lost | crest_timeout
        self.state[crest_failed] = STATE_COOLDOWN
        self.state_time[crest_failed] = 0.0
        self.cooldown_retrigger_latched[crest_failed] = False

        recover_finished = recover_before & (
            self.state_time + time_epsilon >= self.recover_time
        )
        self.state[recover_finished] = STATE_APPROACH
        self.state_time[recover_finished] = 0.0

        retrigger = (
            cooldown_before & need_roll & ~self.cooldown_retrigger_latched
        )
        self.cooldown_retrigger_latched[retrigger] = True
        cooldown_finished = cooldown_before & (
            self.state_time + time_epsilon >= self.cooldown_time
        )
        self.state[cooldown_finished] = STATE_APPROACH
        self.state_time[cooldown_finished] = 0.0
        self.cooldown_retrigger_latched[cooldown_finished] = False

        return {
            "start_contact_confirm": start_contact_confirm,
            "start_roll": start_roll,
            "contact_confirm_timeout": contact_confirm_timeout,
            "start_crest": start_crest,
            "roll_contact_lost": roll_contact_lost,
            "roll_timeout": roll_timeout,
            "crest_finished": crest_finished,
            "crest_contact_lost": crest_contact_lost,
            "crest_timeout": crest_timeout,
            "recover_finished": recover_finished,
            "retrigger": retrigger,
            "contact_confirm_elapsed": contact_confirm_elapsed,
            "roll_elapsed": roll_elapsed,
            "crest_elapsed": crest_elapsed,
            "roll_reference_duration": roll_reference_duration,
            "reference_height": reference_height,
            "reference_vertical_velocity": reference_vertical_velocity,
            "reference_forward": reference_forward,
            "height_progress": height_progress,
            "forward_progress": forward_progress,
            "selected_contact": selected_contact,
            "selected_support_ratio": selected_support_ratio,
            "selected_vertical_force": selected_vertical_force,
            "selected_vertical_velocity": selected_vertical_velocity,
            "contact_loss_steps": self.contact_loss_steps.clone(),
        }
