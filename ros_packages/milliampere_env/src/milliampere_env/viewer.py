#!/usr/bin/env python3
"""Standalone viewer for MilliAmpere1 environment.

Faithfully reproduces the legacy env rendering (heatmap, value bars,
observation text) as a separate ROS subscriber process. The control
loop is never blocked by rendering.

Usage (inside Docker):
    rosrun milliampere_env viewer.py

Or standalone:
    python -m milliampere_env.viewer
"""

from __future__ import annotations

import math
import sys

import numpy as np


def main() -> None:
    import argparse
    import os

    parser = argparse.ArgumentParser(
        description="Standalone viewer for MilliAmpere1 environment"
    )
    parser.add_argument(
        "--gpu",
        action="store_true",
        help="Use GPU rendering (requires working GPU driver in container)",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=4,
        help="Render framerate in Hz (default: 4)",
    )
    args = parser.parse_args()

    if not args.gpu:
        os.environ["LIBGL_ALWAYS_SOFTWARE"] = "1"
        print("Viewer: using software rendering (use --gpu to try GPU)")
    else:
        print("Viewer: using GPU rendering")

    import rospy

    try:
        import pygame
        import pygame.surfarray
    except ImportError:
        print("pygame is required for the viewer. Install with: pip install pygame")
        sys.exit(1)

    try:
        from custom_ros_msgs.msg import EnvState
    except ImportError:
        print("custom_ros_msgs not found. Build the catkin workspace first.")
        sys.exit(1)

    from scipy.ndimage import map_coordinates

    # --- Viewer state ---
    state = {"data": None}

    def callback(msg: EnvState) -> None:
        state["data"] = msg

    rospy.init_node("milliampere_viewer", anonymous=True)
    rospy.Subscriber("/milliampere_env/state", EnvState, callback)

    # --- Display parameters (match legacy env exactly) ---
    width_meters, height_meters = 20, 20
    resolution = 50  # pixels per meter
    width = width_meters * resolution
    height = height_meters * resolution
    bar_width = 30
    bar_height = height - 30
    max_reward = 1.0
    min_reward = -0.1 - 0.1 - 0.1 - 1.0  # -1.3

    # Vessel dimensions in pixels
    rect_width = 2.86 * resolution
    rect_height = 5.06 * resolution

    # --- Precompute Gaussian heatmap grids (match legacy __init__) ---
    sigma_d = 1.0
    sigma_AS_d = 25.0
    inv_sigma_render = np.linalg.inv(np.diag([sigma_d, sigma_d]))
    inv_sigma_AS_render = np.linalg.inv(np.diag([sigma_AS_d, sigma_AS_d]))

    x_grid, y_grid = np.meshgrid(
        np.linspace(0, width_meters, width),
        np.linspace(0, height_meters, height),
    )
    pos = np.dstack((x_grid, y_grid))
    diff = pos - np.array([10, 10])  # centered
    R_gauss_render = np.exp(
        -0.5 * np.einsum("...i,ij,...j", diff, inv_sigma_render, diff)
    )
    R_AS_gauss_render = np.exp(
        -0.5 * np.einsum("...i,ij,...j", diff, inv_sigma_AS_render, diff)
    )

    # --- Pygame setup ---
    pygame.init()
    screen = pygame.display.set_mode((width + bar_width + 50, height))
    pygame.display.set_caption("MilliAmpere1 DP Viewer")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont(None, 24)

    rospy.loginfo("Viewer started. Waiting for /milliampere_env/state ...")

    while not rospy.is_shutdown():
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return

        msg = state["data"]
        if msg is None:
            screen.fill((200, 200, 200))
            text = font.render("Waiting for env state...", True, (0, 0, 0))
            screen.blit(text, (10, 10))
            pygame.display.flip()
            clock.tick(args.fps)
            continue

        # --- Extract reward components from message ---
        rc = msg.reward_components  # [gauss, AS_gauss, vel, thrust, thrust_d, angle_d]
        R_gauss = rc[0]
        R_AS_gauss = rc[1]
        R_vel = rc[2]
        R_thrust = rc[3]
        R_thrust_d = rc[4]
        R_angle_d = rc[5]
        R_rest = R_vel + R_thrust + R_thrust_d + R_angle_d

        # Compute R_gauss_psi and R_AS_gauss_psi for heatmap
        # (factoring out the position-only component from the full reward)
        d = np.sqrt(msg.epsilon[0] ** 2 + msg.epsilon[1] ** 2)
        sigma_d_val = 1.0
        sigma_psi_val = 250.0
        sigma_AS_d_val = 25.0
        sigma_AS_psi_val = 3500.0
        inv_sig = np.linalg.inv(np.diag([sigma_d_val, sigma_psi_val]))
        inv_sig_AS = np.linalg.inv(np.diag([sigma_AS_d_val, sigma_AS_psi_val]))
        denom_gauss = np.exp(-0.5 * d**2 * inv_sig[0][0])
        denom_AS = np.exp(-0.5 * d**2 * inv_sig_AS[0][0])
        R_gauss_psi = R_gauss / denom_gauss if denom_gauss > 1e-30 else 0.0
        R_AS_gauss_psi = R_AS_gauss / denom_AS if denom_AS > 1e-30 else 0.0

        # Build spatial reward heatmap (match legacy _calculate_reward)
        z = (
            np.squeeze(
                R_gauss_render * R_gauss_psi + R_AS_gauss_render * R_AS_gauss_psi
            )
            / 1.4
            + R_rest
        )

        # --- Clear and draw ---
        screen.fill((255, 255, 255))

        # 1. Heatmap
        _draw_heatmap(screen, z, width, height, min_reward, max_reward)

        # 2. Target vessel (gray dashed rectangle)
        target_psi = msg.target_pose[2]
        _draw_target(screen, target_psi, rect_width, rect_height, width, height)

        # 3. Agent vessel (blue filled rectangle) — use NED-frame error for positioning
        agent_pos = np.array(
            [
                int(width // 2)
                - msg.epsilon_ned[1] * resolution,  # East error → screen X
                int(height // 2)
                + msg.epsilon_ned[0] * resolution,  # North error → screen Y (inverted)
            ]
        )
        agent_psi = target_psi - msg.epsilon_ned[2]
        _draw_agent(screen, agent_pos, agent_psi, rect_width, rect_height)

        # 4. Distance circle (10m termination boundary)
        pygame.draw.circle(
            screen, (255, 0, 0), (width // 2, height // 2), int(10.0 * resolution), 2
        )

        # 5. Interpolate reward at agent position for color
        z_value = float(
            np.clip(
                map_coordinates(z, [[agent_pos[1]], [agent_pos[0]]], order=1)[0],
                min_reward,
                max_reward,
            )
        )
        if z_value >= 0:
            intensity = int((z_value / max_reward) * 255)
            center_color = (
                np.clip(255 - intensity, 0, 255),
                255,
                np.clip(255 - intensity, 0, 255),
            )
        else:
            intensity = int((abs(z_value) / abs(max_reward)) * 255)
            center_color = (
                255,
                np.clip(255 - intensity, 0, 255),
                np.clip(255 - intensity, 0, 255),
            )

        # 6. Vertical value bar (scaled by 10x, matching legacy)
        _draw_value_bar(
            screen,
            z_value * 10,
            center_color,
            width,
            bar_width,
            bar_height,
            height,
            min_reward,
            max_reward,
            font,
        )

        # 7. Horizontal penalty bars [R_vel, R_thrust, R_thrust_d, R_angle_d]
        _draw_horizontal_value_bars(
            screen,
            [R_vel, R_thrust, R_thrust_d, R_angle_d],
            4,
            bar_width,
            width,
            height,
        )

        # 8. Observation text overlay (all 22 lines, matching legacy)
        _draw_observations(screen, msg, font)

        pygame.display.flip()
        clock.tick(args.fps)


# ---------------------------------------------------------------------------
# Drawing functions (ported from legacy env render code)
# ---------------------------------------------------------------------------


def _draw_heatmap(screen, z, width, height, min_reward, max_reward) -> None:
    import pygame

    min_value = round(min_reward * 10)
    max_value = round(max_reward * 10)
    p_num_levels = abs(max_value * 2)
    n_num_levels = abs(min_value * 2)

    color_map = np.zeros((height, width, 3), dtype=np.uint8)
    for i in range(max(p_num_levels, n_num_levels) + 1):
        pos_threshold = (i / p_num_levels) * max_value
        next_pos_threshold = ((i + 1) / p_num_levels) * max_value
        neg_threshold = -(i / n_num_levels) * abs(min_value)
        next_neg_threshold = -((i + 1) / n_num_levels) * abs(min_value)
        mask_pos = (10 * z >= pos_threshold) & (10 * z < next_pos_threshold)
        mask_neg = (10 * z < neg_threshold) & (10 * z >= next_neg_threshold)
        p_intensity = min(255, int((i / p_num_levels) * 255))
        n_intensity = min(255, int((i / n_num_levels) * 255))

        color_map[..., 0][mask_neg] = 255
        color_map[..., 1][mask_neg] = max(0, 255 - n_intensity)
        color_map[..., 2][mask_neg] = max(0, 255 - n_intensity)
        color_map[..., 0][mask_pos] = max(0, 255 - p_intensity)
        color_map[..., 1][mask_pos] = 255
        color_map[..., 2][mask_pos] = max(0, 255 - p_intensity)

    surface = pygame.surfarray.make_surface(color_map)
    screen.blit(surface, (0, 0))


def _draw_dashed_line(
    surface, color, start_pos, end_pos, dash_length=10, line_width=1
) -> None:
    import pygame

    x1, y1 = start_pos
    x2, y2 = end_pos
    total_length = math.hypot(x2 - x1, y2 - y1)
    if total_length == 0:
        return
    dash_count = max(int(total_length / dash_length), 1)
    x_inc = (x2 - x1) / dash_count
    y_inc = (y2 - y1) / dash_count

    for i in range(dash_count):
        if i % 2 == 0:
            s = (round(x1 + x_inc * i), round(y1 + y_inc * i))
            e = (round(x1 + x_inc * (i + 1)), round(y1 + y_inc * (i + 1)))
            pygame.draw.line(surface, color, s, e, line_width)


def _draw_dashed_rect(surface, color, rect, dash_length=10, line_width=1) -> None:
    x, y, w, h = rect.x, rect.y, rect.width - 2, rect.height - 2
    _draw_dashed_line(surface, color, (x, y), (x + w, y), dash_length, line_width)
    _draw_dashed_line(
        surface, color, (x + w, y), (x + w, y + h), dash_length, line_width
    )
    _draw_dashed_line(
        surface, color, (x + w, y + h), (x, y + h), dash_length, line_width
    )
    _draw_dashed_line(surface, color, (x, y + h), (x, y), dash_length, line_width)


def _draw_target(screen, orientation, rect_width, rect_height, width, height) -> None:
    import pygame

    rect_center = (width // 2, height // 2)
    target_surface = pygame.Surface(
        (int(rect_width), int(rect_height)), pygame.SRCALPHA
    )

    _draw_dashed_rect(
        target_surface,
        (128, 128, 128),
        target_surface.get_rect(),
        dash_length=10,
        line_width=2,
    )

    # Forward arrow
    arrow_tip = (rect_width / 2, 10)
    arrow_left = (rect_width / 2 - 10, 30)
    arrow_right = (rect_width / 2 + 10, 30)
    pygame.draw.polygon(
        target_surface,
        (128, 128, 128),
        [
            (round(arrow_tip[0]), round(arrow_tip[1])),
            (round(arrow_left[0]), round(arrow_left[1])),
            (round(arrow_right[0]), round(arrow_right[1])),
        ],
    )

    pygame.draw.circle(screen, (128, 128, 128), rect_center, 3)
    rotated = pygame.transform.rotate(target_surface, -orientation)
    new_rect = rotated.get_rect(center=rect_center)
    screen.blit(rotated, new_rect.topleft)


def _draw_agent(screen, position, orientation, rect_width, rect_height) -> None:
    import pygame

    pos = (int(position[0]), int(position[1]))
    target_surface = pygame.Surface((rect_width, rect_height), pygame.SRCALPHA)
    target_surface.fill((0, 0, 255, 128))
    arrow_surface = pygame.Surface((rect_width, rect_height), pygame.SRCALPHA)
    pygame.draw.polygon(
        arrow_surface,
        (0, 0, 0),
        [(rect_width // 2, 0), (rect_width // 2 - 10, 20), (rect_width // 2 + 10, 20)],
    )
    target_surface.blit(arrow_surface, (0, 0))
    rotated = pygame.transform.rotate(target_surface, -orientation)
    new_rect = rotated.get_rect(center=pos)
    screen.blit(rotated, new_rect.topleft)
    pygame.draw.circle(screen, (0, 0, 0), pos, 3)


def _draw_value_bar(
    screen,
    value,
    color,
    disp_width,
    bar_width,
    bar_height,
    disp_height,
    min_reward,
    max_reward,
    font,
) -> None:
    import pygame

    min_value = min_reward * 10
    max_value = max_reward * 10
    border_thickness = 4
    bar_x = disp_width + border_thickness
    bar_y = (disp_height - bar_height) // 2 + border_thickness

    # Border
    pygame.draw.rect(
        screen,
        (0, 0, 0),
        (
            disp_width,
            (disp_height - bar_height) // 2,
            bar_width + border_thickness * 2,
            bar_height + border_thickness * 2,
        ),
    )
    # Background
    pygame.draw.rect(screen, (200, 200, 200), (bar_x, bar_y, bar_width, bar_height))

    def val_to_pos(val):
        return bar_y + (max_value - val) / (max_value - min_value) * bar_height

    zero_pos = val_to_pos(0)
    value_pos = val_to_pos(value)

    if value >= 0:
        draw_top = value_pos
        draw_h = abs(zero_pos - value_pos)
    else:
        draw_top = zero_pos
        draw_h = abs(value_pos - zero_pos)

    pygame.draw.rect(screen, color, (bar_x, draw_top, bar_width, draw_h))

    # Markers and labels
    for val in range(int(min_value), int(max_value) + 1, 1):
        mpos = val_to_pos(val)
        pygame.draw.line(screen, (0, 0, 0), (bar_x, mpos), (bar_x + bar_width, mpos), 2)
        label = f"+{val / 10.0}" if val >= 0 else f"{val / 10.0}"
        text = font.render(label, True, (0, 0, 0))
        text_rect = text.get_rect()
        text_rect.midleft = (bar_x + bar_width + 5, mpos)
        screen.blit(text, text_rect)


def _draw_horizontal_value_bars(
    screen,
    values,
    num_bars,
    bar_w,
    disp_width,
    disp_height,
) -> None:
    import pygame

    min_value = -0.1 * 10
    max_value = 0 * 10
    each_bar_w = (bar_w - 10) // num_bars
    bar_spacing = 10
    bar_x_offset = disp_width - 70
    bar_h = 75
    min_value_alt = -1.0 * 10
    font_small = pygame.font.SysFont(None, 18)

    for i, value in enumerate(values):
        mv = min_value_alt if i == 3 else min_value
        bar_x = bar_x_offset + i * (each_bar_w + bar_spacing)
        bar_y = disp_height - bar_h - 10

        pygame.draw.rect(screen, (255, 0, 0), (bar_x, bar_y, each_bar_w, bar_h))

        fill_height = min(max(0, (value * 10 - mv) / (max_value - mv) * bar_h), bar_h)
        fill_y = bar_y + bar_h - fill_height
        color = (200, 200, 200) if value <= 0 else (0, 255, 0)
        pygame.draw.rect(screen, color, (bar_x, fill_y, each_bar_w, fill_height))

    # Markers on last bar
    last_bar_x = bar_x_offset + (num_bars - 1) * (each_bar_w + bar_spacing)
    last_bar_y = disp_height - bar_h - 10
    for val in range(int(min_value_alt), int(max_value) + 1, 1):
        mpos = last_bar_y + (max_value - val) / (max_value - min_value_alt) * bar_h
        pygame.draw.line(
            screen,
            (0, 0, 0),
            (last_bar_x, mpos),
            (last_bar_x + each_bar_w, mpos),
            1,
        )
        text = font_small.render(f"{val / 10.0}", True, (0, 0, 0))
        text_rect = text.get_rect()
        text_rect.midleft = (last_bar_x + each_bar_w + 5, mpos)
        screen.blit(text, text_rect)


def _draw_observations(screen, msg, font) -> None:
    """Draw all observation text lines matching the legacy 22-line display."""
    text_color = (0, 0, 0)
    eps = msg.epsilon
    vel = msg.est_velocity
    thr = msg.thrusters
    ang = msg.angles
    rc = msg.reward_components

    observations = [
        f"epsilon_x_b = {eps[0]:.3f}",
        f"epsilon_y_b = {eps[1]:.3f}",
        f"epsilon_psi = {eps[2]:.3f}",
        f"est_u = {vel[0]:.3f}",
        f"est_v = {vel[1]:.3f}",
        f"est_r = {vel[2]:.3f}",
        f"target = [{msg.target_pose[0]:.2f}, {msg.target_pose[1]:.2f}, {msg.target_pose[2]:.1f}]",
        f"thrusters = [{thr[0]:.0f}, {thr[1]:.0f}, {thr[2]:.0f}, {thr[3]:.0f}]",
        f"angles = [{ang[0]:.0f}, {ang[1]:.0f}, {ang[2]:.0f}, {ang[3]:.0f}]",
        f"R_gauss = {rc[0]:.4f}",
        f"R_AS_gauss = {rc[1]:.4f}",
        f"R_vel = {rc[2]:.4f}",
        f"R_thrust = {rc[3]:.4f}",
        f"R_thrust_d = {rc[4]:.4f}",
        f"R_angle_d = {rc[5]:.4f}",
        f"R_total = {msg.reward_total:.4f}",
        f"time_step = {msg.time_step}",
        f"episode = {msg.episode}",
    ]

    x, y = 10, 10
    for obs in observations:
        text = font.render(obs, True, text_color)
        screen.blit(text, (x, y))
        y += 30


if __name__ == "__main__":
    main()
