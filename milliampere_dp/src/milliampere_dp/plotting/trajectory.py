"""Ship body drawing, thrust vectors, and angular velocity arcs.

Matplotlib-based plotting primitives extracted from local_env/plot_spline.py.
All functions take an Axes object and draw onto it.
"""

from __future__ import annotations

from math import radians as rad

import matplotlib.transforms as mtransforms
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Arc, Rectangle, RegularPolygon

from milliampere_dp.vessel import (
    THRUSTER_ARM_X,
    THRUSTER_ARM_Y,
    VESSEL_BEAM,
    VESSEL_LENGTH,
)


def draw_vessel_outline(
    ax,
    x: float,
    y: float,
    psi_deg: float,
    *,
    length: float = VESSEL_LENGTH,
    beam: float = VESSEL_BEAM,
    edgecolor: str = "blue",
    facecolor: str = "none",
    alpha: float = 0.6,
) -> Rectangle:
    """Draw a rotated rectangle representing the vessel hull.

    Parameters
    ----------
    ax : matplotlib Axes
        Target axes.
    x, y : float
        Centre position in plot coordinates (y-east, x-north convention).
    psi_deg : float
        Heading in degrees.
    length, beam : float
        Vessel dimensions (metres).
    edgecolor, facecolor : str
        Matplotlib colour strings.
    alpha : float
        Transparency.

    Returns
    -------
    Rectangle
        The patch added to the axes.
    """
    angle_plot = 90.0 - psi_deg
    rect = Rectangle(
        (x - length / 2, y - beam / 2),
        length,
        beam,
        angle=0,
        edgecolor=edgecolor,
        facecolor=facecolor,
        alpha=alpha,
    )
    trans = mtransforms.Affine2D().rotate_deg_around(x, y, angle_plot) + ax.transData
    rect.set_transform(trans)
    ax.add_patch(rect)
    return rect


def draw_thrust_vectors(
    ax,
    x: float,
    y: float,
    psi_deg: float,
    thrusters_rpm: np.ndarray,
    angles_deg: np.ndarray,
    *,
    arm_x: float = THRUSTER_ARM_X,
    arm_y: float = THRUSTER_ARM_Y,
    color: str = "lime",
    alpha: float = 1.0,
    rpm_scale: float = 2 * np.pi / 1200,
):
    """Draw thrust vectors at each thruster position.

    Parameters
    ----------
    ax : matplotlib Axes
        Target axes.
    x, y : float
        Vessel centre in plot coordinates.
    psi_deg : float
        Heading in degrees.
    thrusters_rpm : array-like, shape (4,)
        Thrust magnitude in RPM for each thruster.
    angles_deg : array-like, shape (4,)
        Azimuth angle in degrees for each thruster.
    arm_x, arm_y : float
        Thruster arm lengths from CoG.
    color : str
        Arrow colour.
    alpha : float
        Transparency.
    rpm_scale : float
        Scale factor converting RPM to arrow length.
    """
    thruster_offsets = np.array(
        [[arm_x, arm_y], [arm_x, -arm_y], [-arm_x, arm_y], [-arm_x, -arm_y]]
    )
    angle_rad = np.deg2rad(90.0 - psi_deg)
    rot = np.array(
        [
            [np.cos(angle_rad), -np.sin(angle_rad)],
            [np.sin(angle_rad), np.cos(angle_rad)],
        ]
    )
    global_pos = rot @ thruster_offsets.T
    global_pos[0, :] += x
    global_pos[1, :] += y

    for i in range(4):
        tx = global_pos[0, i]
        ty = global_pos[1, i]
        vec_angle = angle_rad - np.deg2rad(angles_deg[i])
        dx = np.cos(vec_angle) * rpm_scale * thrusters_rpm[i]
        dy = np.sin(vec_angle) * rpm_scale * thrusters_rpm[i]
        ax.quiver(
            tx,
            ty,
            dx,
            dy,
            angles="xy",
            scale_units="xy",
            scale=1,
            color=color,
            alpha=alpha,
        )


def draw_angular_velocity_arc(
    ax,
    x: float,
    y: float,
    psi_deg: float,
    r_deg_per_s: float,
    *,
    radius: float = 1.0,
    color: str = "purple",
) -> Line2D:
    """Draw an arc with arrowhead representing angular velocity.

    Parameters
    ----------
    ax : matplotlib Axes
        Target axes.
    x, y : float
        Centre position.
    psi_deg : float
        Current heading in degrees.
    r_deg_per_s : float
        Yaw rate in degrees/second.
    radius : float
        Arc radius.
    color : str
        Arc and arrowhead colour.

    Returns
    -------
    Line2D
        Legend handle for the arc.
    """
    base_angle = 90.0 - psi_deg
    arc_span = 360.0 * r_deg_per_s / 120.0

    if arc_span > 0:
        theta1, theta2 = -arc_span, 0.0
    else:
        theta1, theta2 = 0.0, 360.0 - arc_span

    arc = Arc(
        (x, y),
        radius,
        radius,
        angle=base_angle,
        theta1=theta1,
        theta2=theta2,
        linestyle="-",
        lw=3,
        color=color,
    )
    ax.add_patch(arc)

    legend_element = Line2D(
        [0],
        [0],
        color=color,
        lw=3,
        linestyle="-",
        label=r"Angular Velocity [$^\circ$/s]",
    )

    # Start dot
    start_x = x + (radius / 2) * np.cos(rad(base_angle))
    start_y = y + (radius / 2) * np.sin(rad(base_angle))
    ax.plot(start_x, start_y, "ko", markersize=3)

    # Arrowhead
    if theta1 == 0:
        end_ang = theta2
        orientation = rad(base_angle + end_ang)
    else:
        end_ang = theta1
        orientation = rad(base_angle + end_ang + 180)

    end_x = x + (radius / 2) * np.cos(rad(end_ang + base_angle))
    end_y = y + (radius / 2) * np.sin(rad(end_ang + base_angle))
    ax.add_patch(
        RegularPolygon(
            (end_x, end_y),
            3,
            radius=radius / 9,
            orientation=orientation,
            color=color,
        )
    )

    return legend_element
