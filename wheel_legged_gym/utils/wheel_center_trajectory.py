"""Geometry helpers for contact-guided wheel-center stair trajectories.

The functions in this module are deliberately independent from Isaac Gym so
that the trajectory geometry and completion rules can be tested on CPU.
"""

import math

import torch


def curriculum_obstacle_height(level, num_levels, min_height, max_height):
    """Return a curriculum height whose first and last levels are exact."""
    if num_levels < 2:
        raise ValueError("num_levels must be at least 2")
    if not 0 <= level < num_levels:
        raise ValueError("level must be in [0, num_levels)")
    return min_height + (max_height - min_height) * level / (num_levels - 1)


def wheel_center_path(step_height, wheel_radius, num_samples=33):
    """Build equal-arc-length wheel-center paths in the forward-z plane.

    Args:
        step_height: Tensor of shape ``[N]`` in metres.
        wheel_radius: Positive wheel radius in metres.
        num_samples: Number of path samples, including both endpoints.

    Returns:
        points: Reference positions of shape ``[N, num_samples, 2]``.
        tangents: Unit path tangents with the same shape as ``points``.
        path_length: Total path length of shape ``[N]``.
    """
    if wheel_radius <= 0:
        raise ValueError("wheel_radius must be positive")
    if num_samples < 2:
        raise ValueError("num_samples must be at least 2")
    if step_height.ndim != 1:
        raise ValueError("step_height must be a one-dimensional tensor")

    height = torch.clamp(step_height, min=0.0)
    radius = torch.as_tensor(
        wheel_radius, dtype=height.dtype, device=height.device
    )
    q = torch.linspace(
        0.0, 1.0, num_samples, dtype=height.dtype, device=height.device
    ).unsqueeze(0)

    # Low steps: the wheel is already above the corner and follows only the
    # circular part of the path.
    theta_start = torch.asin(torch.clamp((radius - height) / radius, 0.0, 1.0))
    theta_range = 0.5 * math.pi - theta_start
    theta = theta_start.unsqueeze(1) + q * theta_range.unsqueeze(1)
    low_x = radius * (torch.cos(theta_start).unsqueeze(1) - torch.cos(theta))
    low_z = radius * (torch.sin(theta) - torch.sin(theta_start).unsqueeze(1))
    low_tangent_x = torch.sin(theta)
    low_tangent_z = torch.cos(theta)
    low_length = radius * theta_range

    # High steps: lift along the riser until the wheel centre reaches the
    # corner height, then traverse a quarter circle around the corner.
    vertical_length = torch.relu(height - radius)
    arc_length = 0.5 * math.pi * radius
    high_length = vertical_length + arc_length
    travelled = q * high_length.unsqueeze(1)
    on_riser = travelled <= vertical_length.unsqueeze(1)
    phi = torch.clamp(
        (travelled - vertical_length.unsqueeze(1)) / radius,
        min=0.0,
        max=0.5 * math.pi,
    )
    high_x = torch.where(
        on_riser,
        torch.zeros_like(travelled),
        radius * (1.0 - torch.cos(phi)),
    )
    high_z = torch.where(
        on_riser,
        travelled,
        vertical_length.unsqueeze(1) + radius * torch.sin(phi),
    )
    high_tangent_x = torch.where(on_riser, torch.zeros_like(phi), torch.sin(phi))
    high_tangent_z = torch.where(on_riser, torch.ones_like(phi), torch.cos(phi))

    low_step = (height <= radius).unsqueeze(1)
    points = torch.stack(
        (
            torch.where(low_step, low_x, high_x),
            torch.where(low_step, low_z, high_z),
        ),
        dim=-1,
    )
    tangents = torch.stack(
        (
            torch.where(low_step, low_tangent_x, high_tangent_x),
            torch.where(low_step, low_tangent_z, high_tangent_z),
        ),
        dim=-1,
    )
    path_length = torch.where(height <= radius, low_length, high_length)
    return points, tangents, path_length


def nearest_wheel_center_path_state(actual_position, points, tangents):
    """Find the nearest sampled path point and its progress/tangent."""
    if actual_position.ndim != 2 or actual_position.shape[-1] != 2:
        raise ValueError("actual_position must have shape [N, 2]")
    if points.ndim != 3 or points.shape[-1] != 2:
        raise ValueError("points must have shape [N, S, 2]")
    if points.shape != tangents.shape:
        raise ValueError("points and tangents must have identical shapes")
    if points.shape[0] != actual_position.shape[0]:
        raise ValueError("batch dimensions must match")

    squared_distance = torch.sum(
        torch.square(points - actual_position.unsqueeze(1)), dim=-1
    )
    nearest_index = torch.argmin(squared_distance, dim=1)
    batch_index = torch.arange(points.shape[0], device=points.device)
    path_error = torch.sqrt(
        torch.clamp(squared_distance[batch_index, nearest_index], min=0.0)
    )
    progress = nearest_index.to(points.dtype) / float(points.shape[1] - 1)
    tangent = tangents[batch_index, nearest_index]
    end_error = torch.linalg.vector_norm(actual_position - points[:, -1], dim=-1)
    return path_error, progress, tangent, end_error


def tangent_rolling_error(
    tangential_speed,
    wheel_angular_velocity,
    wheel_radius,
    command_direction,
    wheel_forward_sign,
):
    """Return signed no-slip error along the trajectory tangent."""
    forward_angular_velocity = (
        command_direction * wheel_forward_sign * wheel_angular_velocity
    )
    return tangential_speed - wheel_radius * forward_angular_velocity


def histogram_quantile(histogram, quantile, bin_width):
    """Return an upper-bin-edge quantile from a one-dimensional histogram."""
    if histogram.ndim != 1:
        raise ValueError("histogram must be one-dimensional")
    if not 0.0 <= quantile <= 1.0:
        raise ValueError("quantile must be in [0, 1]")
    total = torch.sum(histogram)
    if total.item() <= 0:
        return torch.zeros((), dtype=histogram.dtype, device=histogram.device)
    target = quantile * total
    index = torch.argmax((torch.cumsum(histogram, dim=0) >= target).to(torch.int64))
    return (index.to(histogram.dtype) + 1.0) * bin_width


def wheel_trajectory_completion(
    progress,
    end_error,
    vertical_force,
    support_steps,
    elapsed_time,
    step_height,
    progress_threshold,
    end_tolerance,
    support_force_threshold,
    required_support_steps,
    minimum_duration,
    timeout_base,
    timeout_height_gain,
):
    """Update support counters and evaluate success/timeout conditions."""
    supporting = vertical_force >= support_force_threshold
    updated_support_steps = torch.where(
        supporting, support_steps + 1, torch.zeros_like(support_steps)
    )
    completed = (
        (progress >= progress_threshold)
        & (end_error <= end_tolerance)
        & (updated_support_steps >= required_support_steps)
        & (elapsed_time >= minimum_duration)
    )
    timeout = elapsed_time >= timeout_base + timeout_height_gain * step_height
    timed_out = timeout & (~completed)
    return updated_support_steps, completed, timed_out
