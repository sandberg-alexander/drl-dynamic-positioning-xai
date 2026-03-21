from __future__ import annotations

import argparse
import os

import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Arc, Rectangle, RegularPolygon
from numpy import radians as rad

from milliampere_dp.plotting.episodes import mark_episodes
from milliampere_dp.vessel import (
    MAX_ANGULAR_SPEED,
    MAX_DISTANCE,
    MAX_HEADING_ANGLE,
    MAX_LINEAR_SPEED,
    MAX_THRUSTER_RPM,
)


def load_data(csv_path):
    df = pd.read_csv(csv_path)

    x = df["x"] * MAX_DISTANCE
    y = df["y"] * MAX_DISTANCE
    psi = df["psi"] * MAX_HEADING_ANGLE
    u = df["u"] * MAX_LINEAR_SPEED
    v = df["v"] * MAX_LINEAR_SPEED
    r = df["r"] * MAX_ANGULAR_SPEED
    n1x = df["n1x"] * MAX_THRUSTER_RPM
    n1y = df["n1y"] * MAX_THRUSTER_RPM
    n2x = df["n2x"] * MAX_THRUSTER_RPM
    n2y = df["n2y"] * MAX_THRUSTER_RPM
    n3x = df["n3x"] * MAX_THRUSTER_RPM
    n3y = df["n3y"] * MAX_THRUSTER_RPM
    n4x = df["n4x"] * MAX_THRUSTER_RPM
    n4y = df["n4y"] * MAX_THRUSTER_RPM
    n1 = df["n1d"]
    alpha1 = df["alpha1d"]
    n2 = df["n2d"]
    alpha2 = df["alpha2d"]
    n3 = df["n3d"]
    alpha3 = df["alpha3d"]
    n4 = df["n4d"]
    alpha4 = df["alpha4d"]

    dt = np.arange(len(x))

    return dict(
        df=df,
        x=x,
        y=y,
        psi=psi,
        u=u,
        v=v,
        r=r,
        n1x=n1x,
        n1y=n1y,
        n2x=n2x,
        n2y=n2y,
        n3x=n3x,
        n3y=n3y,
        n4x=n4x,
        n4y=n4y,
        n1=n1,
        alpha1=alpha1,
        n2=n2,
        alpha2=alpha2,
        n3=n3,
        alpha3=alpha3,
        n4=n4,
        alpha4=alpha4,
        dt=dt,
    )


def _rotate_body2ned(psi):
    psi = np.deg2rad(psi)
    c = np.cos(psi)
    s = np.sin(psi)
    return np.array([[c, -s], [s, c]])


def drawCirc(ax, radius, centX, centY, angle_, theta1_, theta2_, color_="black"):
    # ========Line
    arc = Arc(
        [centX, centY],
        radius,
        radius,
        angle=angle_,
        theta1=theta1_,
        theta2=theta2_,
        linestyle="-",
        lw=3,
        color=color_,
    )
    ax.add_patch(arc)
    legend_element = Line2D(
        [0],
        [0],
        color=color_,
        lw=3,
        linestyle="-",
        label=r"Angular Velocity [$^\circ$/s]",
    )

    startX = centX + (radius / 2) * np.cos(rad(angle_))
    startY = centY + (radius / 2) * np.sin(rad(angle_))
    ax.plot(startX, startY, "ko", markersize=3)

    if theta1_ == 0:
        ang = theta2_

        # ========Create the arrow head
        endX = centX + (radius / 2) * np.cos(rad(ang + angle_))
        endY = centY + (radius / 2) * np.sin(rad(ang + angle_))
        ax.add_patch(
            RegularPolygon(
                (endX, endY),
                3,
                radius=radius / 9,
                orientation=rad(angle_ + ang),
                color=color_,
            )
        )
    else:
        ang = theta1_

        # ========Create the arrow head
        endX = centX + (radius / 2) * np.cos(rad(ang + angle_))
        endY = centY + (radius / 2) * np.sin(rad(ang + angle_))
        ax.add_patch(
            RegularPolygon(
                (endX, endY),
                3,
                radius=radius / 9,
                orientation=rad(angle_ + ang + 180),
                color=color_,
            )
        )

    return legend_element


def plot_x(data, scale, render_start, render_end, interval, run_number, output_dir):
    x = data["x"]
    dt = data["dt"]

    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.axhline(
        y=0,
        color="red",
        linestyle="--",
        linewidth=1.5 * scale,
        label=f"Set point (y={0})",
    )
    ax.plot(
        dt[render_start:render_end],
        x[render_start:render_end],
        linewidth=1.5 * scale,
        color="black",
    )
    ax.legend(fontsize=10 * scale)
    ax.set_xlabel(r"time step [dt]", fontsize=14 * scale)
    ax.set_ylabel(r"$\tilde{x}^b_t$ [m]", fontsize=14 * scale)
    ax.tick_params(axis="x", labelsize=10 * scale)
    ax.tick_params(axis="y", labelsize=10 * scale)
    mark_episodes(ax, render_start, render_end, interval)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{run_number}_x.pdf"), format="pdf", dpi=1200)
    plt.close(fig)


def plot_y(data, scale, render_start, render_end, interval, run_number, output_dir):
    y = data["y"]
    dt = data["dt"]

    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.axhline(
        y=0,
        color="red",
        linestyle="--",
        linewidth=1.5 * scale,
        label=f"Set point (y={0})",
    )
    ax.plot(
        dt[render_start:render_end],
        y[render_start:render_end],
        linewidth=1.5 * scale,
        color="black",
    )
    ax.legend(fontsize=10 * scale)
    ax.set_xlabel(r"time step [dt]", fontsize=14 * scale)
    ax.set_ylabel(r"$\tilde{y}^b_t$ [m]", fontsize=14 * scale)
    ax.tick_params(axis="x", labelsize=10 * scale)
    ax.tick_params(axis="y", labelsize=10 * scale)
    mark_episodes(ax, render_start, render_end, interval)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{run_number}_y.pdf"), format="pdf", dpi=1200)
    plt.close(fig)


def plot_psi(data, scale, render_start, render_end, interval, run_number, output_dir):
    psi = data["psi"]
    dt = data["dt"]

    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.axhline(
        y=0,
        color="red",
        linestyle="--",
        linewidth=1.5 * scale,
        label=f"Set point (y={0})",
    )
    ax.plot(
        dt[render_start:render_end],
        psi[render_start:render_end],
        linewidth=1.5 * scale,
        color="black",
    )
    ax.legend(fontsize=10 * scale)
    ax.set_xlabel(r"time step [dt]", fontsize=14 * scale)
    ax.set_ylabel(r"$\tilde{\psi}_t$ [$^\circ$]", fontsize=14 * scale)
    ax.tick_params(axis="x", labelsize=10 * scale)
    ax.tick_params(axis="y", labelsize=10 * scale)
    mark_episodes(ax, render_start, render_end, interval)

    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, f"{run_number}_psi.pdf"), format="pdf", dpi=1200
    )
    plt.close(fig)


def plot_u(data, scale, render_start, render_end, interval, run_number, output_dir):
    u = data["u"]
    dt = data["dt"]

    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.axhline(
        y=0,
        color="red",
        linestyle="--",
        linewidth=1.5 * scale,
        label=f"Set point (y={0})",
    )
    ax.plot(
        dt[render_start:render_end],
        u[render_start:render_end],
        linewidth=1.5 * scale,
        color="black",
    )
    ax.legend(fontsize=10 * scale)
    ax.set_xlabel(r"time step [dt]", fontsize=14 * scale)
    ax.set_ylabel(r"$\hat{u}_t$ [m/s]", fontsize=14 * scale)
    ax.tick_params(axis="x", labelsize=10 * scale)
    ax.tick_params(axis="y", labelsize=10 * scale)
    mark_episodes(ax, render_start, render_end, interval)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{run_number}_u.pdf"), format="pdf", dpi=1200)
    plt.close(fig)


def plot_v(data, scale, render_start, render_end, interval, run_number, output_dir):
    v = data["v"]
    dt = data["dt"]

    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.axhline(
        y=0,
        color="red",
        linestyle="--",
        linewidth=1.5 * scale,
        label=f"Set point (y={0})",
    )
    ax.plot(
        dt[render_start:render_end],
        v[render_start:render_end],
        linewidth=1.5 * scale,
        color="black",
    )
    ax.legend(fontsize=10 * scale)
    ax.set_xlabel(r"time step [dt]", fontsize=14 * scale)
    ax.set_ylabel(r"$\hat{v}_t$ [m/s]", fontsize=14 * scale)
    ax.tick_params(axis="x", labelsize=10 * scale)
    ax.tick_params(axis="y", labelsize=10 * scale)
    mark_episodes(ax, render_start, render_end, interval)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{run_number}_v.pdf"), format="pdf", dpi=1200)
    plt.close(fig)


def plot_r(data, scale, render_start, render_end, interval, run_number, output_dir):
    r = data["r"]
    dt = data["dt"]

    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.axhline(
        y=0,
        color="red",
        linestyle="--",
        linewidth=1.5 * scale,
        label=f"Set point (y={0})",
    )
    ax.plot(
        dt[render_start:render_end],
        r[render_start:render_end],
        linewidth=1.5 * scale,
        color="black",
    )
    ax.legend(fontsize=10 * scale)
    ax.set_xlabel(r"time step [dt]", fontsize=14 * scale)
    ax.set_ylabel(r"$\hat{r}_t$ [$^\circ$/s]", fontsize=14 * scale)
    ax.tick_params(axis="x", labelsize=10 * scale)
    ax.tick_params(axis="y", labelsize=10 * scale)
    mark_episodes(ax, render_start, render_end, interval)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{run_number}_r.pdf"), format="pdf", dpi=1200)
    plt.close(fig)


def plot_n1(data, scale, render_start, render_end, interval, run_number, output_dir):
    n1 = data["n1"]
    dt = data["dt"]

    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.axhline(
        y=0, color="gray", linestyle="--", linewidth=1.5 * scale, label=f"(y={0})"
    )
    ax.plot(
        dt[render_start:render_end],
        n1[render_start:render_end],
        linewidth=1.5 * scale,
        color="black",
    )
    ax.legend(fontsize=10 * scale)
    ax.set_xlabel(r"time step [dt]", fontsize=14 * scale)
    ax.set_ylabel(r"$n_{d_1,t-1}$ [RPM]", fontsize=14 * scale)
    ax.tick_params(axis="x", labelsize=10 * scale)
    ax.tick_params(axis="y", labelsize=10 * scale)
    mark_episodes(ax, render_start, render_end, interval)

    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, f"{run_number}_n1.pdf"), format="pdf", dpi=1200
    )
    plt.close(fig)


def plot_n2(data, scale, render_start, render_end, interval, run_number, output_dir):
    n2 = data["n2"]
    dt = data["dt"]

    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.axhline(y=0, color="gray", linestyle="--", label=f"(y={0})")
    ax.plot(
        dt[render_start:render_end],
        n2[render_start:render_end],
        linewidth=1.5 * scale,
        color="black",
    )
    ax.legend(fontsize=10 * scale)
    ax.set_xlabel(r"time step [dt]", fontsize=14 * scale)
    ax.set_ylabel(r"$n_{d_2,t-1}$ [RPM]", fontsize=14 * scale)
    ax.tick_params(axis="x", labelsize=10 * scale)
    ax.tick_params(axis="y", labelsize=10 * scale)
    mark_episodes(ax, render_start, render_end, interval)

    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, f"{run_number}_n2.pdf"), format="pdf", dpi=1200
    )
    plt.close(fig)


def plot_n3(data, scale, render_start, render_end, interval, run_number, output_dir):
    n3 = data["n3"]
    dt = data["dt"]

    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.axhline(
        y=0, color="gray", linestyle="--", linewidth=1.5 * scale, label=f"(y={0})"
    )
    ax.plot(
        dt[render_start:render_end],
        n3[render_start:render_end],
        linewidth=1.5 * scale,
        color="black",
    )
    ax.legend(fontsize=10 * scale)
    ax.set_xlabel(r"time step [dt]", fontsize=14 * scale)
    ax.set_ylabel(r"$n_{d_3,t-1}$ [RPM]", fontsize=14 * scale)
    ax.tick_params(axis="x", labelsize=10 * scale)
    ax.tick_params(axis="y", labelsize=10 * scale)
    mark_episodes(ax, render_start, render_end, interval)

    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, f"{run_number}_n3.pdf"), format="pdf", dpi=1200
    )
    plt.close(fig)


def plot_n4(data, scale, render_start, render_end, interval, run_number, output_dir):
    n4 = data["n4"]
    dt = data["dt"]

    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.axhline(
        y=0, color="gray", linestyle="--", linewidth=1.5 * scale, label=f"(y={0})"
    )
    ax.plot(
        dt[render_start:render_end],
        n4[render_start:render_end],
        linewidth=1.5 * scale,
        color="black",
    )
    ax.legend(fontsize=10 * scale)
    ax.set_xlabel(r"time step [dt]", fontsize=14 * scale)
    ax.set_ylabel(r"$n_{d_4,t-1}$ [RPM]", fontsize=14 * scale)
    ax.tick_params(axis="x", labelsize=10 * scale)
    ax.tick_params(axis="y", labelsize=10 * scale)
    mark_episodes(ax, render_start, render_end, interval)

    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, f"{run_number}_n4.pdf"), format="pdf", dpi=1200
    )
    plt.close(fig)


def plot_alpha1(
    data, scale, render_start, render_end, interval, run_number, output_dir
):
    alpha1 = data["alpha1"]
    dt = data["dt"]

    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.axhline(
        y=0, color="gray", linestyle="--", linewidth=1.5 * scale, label=f"(y={0})"
    )
    ax.plot(
        dt[render_start:render_end],
        alpha1[render_start:render_end],
        linewidth=1.5 * scale,
        color="black",
    )
    ax.legend(fontsize=10 * scale)
    ax.set_xlabel(r"time step [dt]", fontsize=14 * scale)
    ax.set_ylabel(r"$\alpha_{d_1,t-1}$ [$^\circ$]", fontsize=14 * scale)
    ax.tick_params(axis="x", labelsize=10 * scale)
    ax.tick_params(axis="y", labelsize=10 * scale)
    mark_episodes(ax, render_start, render_end, interval)

    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, f"{run_number}_alpha1.pdf"), format="pdf", dpi=1200
    )
    plt.close(fig)


def plot_alpha2(
    data, scale, render_start, render_end, interval, run_number, output_dir
):
    alpha2 = data["alpha2"]
    dt = data["dt"]

    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.axhline(
        y=0, color="gray", linestyle="--", linewidth=1.5 * scale, label=f"(y={0})"
    )
    ax.plot(
        dt[render_start:render_end],
        alpha2[render_start:render_end],
        linewidth=1.5 * scale,
        color="black",
    )
    ax.legend(fontsize=10 * scale)
    ax.set_xlabel(r"time step [dt]", fontsize=14 * scale)
    ax.set_ylabel(r"$\alpha_{d_2,t-1}$ [$^\circ$]", fontsize=14 * scale)
    ax.tick_params(axis="x", labelsize=10 * scale)
    ax.tick_params(axis="y", labelsize=10 * scale)
    mark_episodes(ax, render_start, render_end, interval)

    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, f"{run_number}_alpha2.pdf"), format="pdf", dpi=1200
    )
    plt.close(fig)


def plot_alpha3(
    data, scale, render_start, render_end, interval, run_number, output_dir
):
    alpha3 = data["alpha3"]
    dt = data["dt"]

    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.axhline(
        y=0, color="gray", linestyle="--", linewidth=1.5 * scale, label=f"(y={0})"
    )
    ax.plot(
        dt[render_start:render_end],
        alpha3[render_start:render_end],
        linewidth=1.5 * scale,
        color="black",
    )
    ax.legend(fontsize=10 * scale)
    ax.set_xlabel(r"time step [dt]", fontsize=14 * scale)
    ax.set_ylabel(r"$\alpha_{d_3,t-1}$ [$^\circ$]", fontsize=14 * scale)
    ax.tick_params(axis="x", labelsize=10 * scale)
    ax.tick_params(axis="y", labelsize=10 * scale)
    mark_episodes(ax, render_start, render_end, interval)

    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, f"{run_number}_alpha3.pdf"), format="pdf", dpi=1200
    )
    plt.close(fig)


def plot_alpha4(
    data, scale, render_start, render_end, interval, run_number, output_dir
):
    alpha4 = data["alpha4"]
    dt = data["dt"]

    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.axhline(
        y=0, color="gray", linestyle="--", linewidth=1.5 * scale, label=f"(y={0})"
    )
    ax.plot(
        dt[render_start:render_end],
        alpha4[render_start:render_end],
        linewidth=1.5 * scale,
        color="black",
    )
    ax.legend(fontsize=10 * scale)
    ax.set_xlabel(r"time step [dt]", fontsize=14 * scale)
    ax.set_ylabel(r"$\alpha_{d_4,t-1}$ [$^\circ$]", fontsize=14 * scale)
    ax.tick_params(axis="x", labelsize=10 * scale)
    ax.tick_params(axis="y", labelsize=10 * scale)
    mark_episodes(ax, render_start, render_end, interval)

    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, f"{run_number}_alpha4.pdf"), format="pdf", dpi=1200
    )
    plt.close(fig)


def plot_combined_states(
    data, scale, render_start, render_end, interval, run_number, output_dir
):
    x = data["x"]
    y = data["y"]
    psi = data["psi"]
    dt = data["dt"]

    x_d = [0] * 200 + [4] * 400 + [0] * 400
    y_d = [0] * 400 + [4] * 200 + [0] * 400
    psi_d = [0] * 800 + [180] * 200

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 4.8 * 2))

    ax1.plot(
        x_d, color="red", linestyle="--", linewidth=1.5 * scale, label="Desired pose"
    )
    ax1.plot(
        dt[render_start:render_end],
        x_d - x[render_start:render_end],
        linewidth=1.5 * scale,
        color="black",
        label="Real pose (simulator)",
    )
    ax1.legend(fontsize=10 * scale)
    ax1.set_xlabel(r"time step [dt]", fontsize=14 * scale)
    ax1.set_ylabel(r"$x^n_t$ [m]", fontsize=14 * scale)
    ax1.tick_params(axis="x", labelsize=10 * scale)
    ax1.tick_params(axis="y", labelsize=10 * scale)
    mark_episodes(ax1, render_start, render_end, interval)

    ax2.plot(
        y_d, color="red", linestyle="--", linewidth=1.5 * scale, label="Desired pose"
    )
    ax2.plot(
        dt[render_start:render_end],
        y_d - y[render_start:render_end],
        linewidth=1.5 * scale,
        color="black",
        label="Real pose (simulator)",
    )
    ax2.legend(fontsize=10 * scale)
    ax2.set_xlabel(r"time step [dt]", fontsize=14 * scale)
    ax2.set_ylabel(r"$y^n_t$ [m]", fontsize=14 * scale)
    ax2.tick_params(axis="x", labelsize=10 * scale)
    ax2.tick_params(axis="y", labelsize=10 * scale)
    mark_episodes(ax2, render_start, render_end, interval)

    ax3.plot(
        psi_d, color="red", linestyle="--", linewidth=1.5 * scale, label="Desired pose"
    )
    ax3.plot(
        dt[render_start:render_end],
        psi_d - psi[render_start:render_end],
        linewidth=1.5 * scale,
        color="black",
        label="Real pose (simulator)",
    )
    ax3.legend(fontsize=10 * scale)
    ax3.set_xlabel(r"time step [dt]", fontsize=14 * scale)
    ax3.set_ylabel(r"$\psi_t$ [$^\circ$]", fontsize=14 * scale)
    ax3.tick_params(axis="x", labelsize=10 * scale)
    ax3.tick_params(axis="y", labelsize=10 * scale)
    mark_episodes(ax3, render_start, render_end, interval)

    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, f"{run_number}_states.pdf"), format="pdf", dpi=1200
    )
    plt.close(fig)


def plot_combined_velocities(
    data, scale, render_start, render_end, interval, run_number, output_dir
):
    u = data["u"]
    v = data["v"]
    r = data["r"]
    dt = data["dt"]

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 4.8 * 2))

    ax1.axhline(
        y=0, color="gray", linestyle="--", linewidth=1.5 * scale, label=f"(y={0})"
    )
    ax1.plot(
        dt[render_start:render_end],
        u[render_start:render_end],
        linewidth=1.5 * scale,
        color="black",
    )
    ax1.legend(fontsize=10 * scale)
    ax1.set_xlabel(r"time step [dt]", fontsize=14 * scale)
    ax1.set_ylabel(r"$\hat{u}_t$ [m/s]", fontsize=14 * scale)
    ax1.tick_params(axis="x", labelsize=10 * scale)
    ax1.tick_params(axis="y", labelsize=10 * scale)
    mark_episodes(ax1, render_start, render_end, interval)

    ax2.axhline(
        y=0, color="gray", linestyle="--", linewidth=1.5 * scale, label=f"(y={0})"
    )
    ax2.plot(
        dt[render_start:render_end],
        v[render_start:render_end],
        linewidth=1.5 * scale,
        color="black",
    )
    ax2.legend(fontsize=10 * scale)
    ax2.set_xlabel(r"time step [dt]", fontsize=14 * scale)
    ax2.set_ylabel(r"$\hat{v}_t$ [m/s]", fontsize=14 * scale)
    ax2.tick_params(axis="x", labelsize=10 * scale)
    ax2.tick_params(axis="y", labelsize=10 * scale)
    mark_episodes(ax2, render_start, render_end, interval)

    ax3.axhline(
        y=0, color="gray", linestyle="--", linewidth=1.5 * scale, label=f"(y={0})"
    )
    ax3.plot(
        dt[render_start:render_end],
        r[render_start:render_end],
        linewidth=1.5 * scale,
        color="black",
    )
    ax3.legend(fontsize=10 * scale)
    ax3.set_xlabel(r"time step [dt]", fontsize=14 * scale)
    ax3.set_ylabel(r"$\hat{r}_t$ [$^\circ$/s]", fontsize=14 * scale)
    ax3.tick_params(axis="x", labelsize=10 * scale)
    ax3.tick_params(axis="y", labelsize=10 * scale)
    mark_episodes(ax3, render_start, render_end, interval)

    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, f"{run_number}_states2.pdf"), format="pdf", dpi=1200
    )
    plt.close(fig)


def plot_xy_trajectory(data, render_start, render_end, run_number, output_dir):
    x = data["x"]
    y = data["y"]
    psi = data["psi"]

    xy = np.array([list(x), list(y)])
    xy = [-_rotate_body2ned(-p) @ xy[:, i] for i, p in enumerate(psi)]
    xy = np.array(xy)
    x_rot = xy[:, 0]
    y_rot = xy[:, 1]

    for a in range(5):
        offset = a * 200 * 0
        k = 10 * 0

        fig, ax = plt.subplots()
        ax.axhline(y=0, color="gray", linestyle="--", label=f"(y={0})")
        ax.axvline(x=0, color="gray", linestyle="--")
        ax.plot(
            y_rot[render_start + k + offset : render_start + offset + 200],
            x_rot[render_start + k + offset : render_start + offset + 200],
            color="red",
        )
        ax.set_xlabel(r"$y^n$", fontsize=14)
        ax.set_ylabel(r"$x^n$", fontsize=14)
        ax.tick_params(axis="x", labelsize=10)
        ax.tick_params(axis="y", labelsize=10)
        ax.set_xlim((-7, 7))
        ax.set_ylim((-7, 7))
        ax.set_aspect("equal", adjustable="datalim")

        my_range = [render_start + k + offset, render_end - 1]

        # Add rectangles at start and end
        for t in my_range:
            rect_x = y_rot[t]
            rect_y = x_rot[t]
            width, height = 5.06, 2.86
            angle = 90 - psi[t]

            rect = Rectangle(
                (rect_x - width / 2, rect_y - height / 2),
                width,
                height,
                angle=0,
                color="blue",
                alpha=0.6,
            )
            trans = (
                transforms.Affine2D().rotate_deg_around(rect_x, rect_y, angle)
                + ax.transData
            )
            rect.set_transform(trans)
            ax.add_patch(rect)

        # Add vectors every 10 time steps
        for t in range(render_start + k + offset, render_start + offset + 200, 10):
            vector_x = y_rot[t]
            vector_y = x_rot[t]
            angle = np.deg2rad(90 - psi[t])

            dx = np.cos(angle)
            dy = np.sin(angle)

            scale_factor = 0.5 * 2
            dx *= scale_factor
            dy *= scale_factor

            ax.quiver(
                vector_x,
                vector_y,
                dx,
                dy,
                angles="xy",
                scale_units="xy",
                scale=1,
                color="black",
                alpha=0.6,
            )

        plt.tight_layout()
        plt.savefig(
            os.path.join(output_dir, f"{run_number}_xy{a}.pdf"), format="pdf", dpi=1200
        )
        plt.close(fig)


def plot_xy_thruster(data, render_start, render_end, run_number, output_dir):
    x = data["x"]
    y = data["y"]
    psi = data["psi"]
    u = data["u"]
    v = data["v"]
    r = data["r"]
    n1 = data["n1"]
    n2 = data["n2"]
    n3 = data["n3"]
    n4 = data["n4"]
    alpha1 = data["alpha1"]
    alpha2 = data["alpha2"]
    alpha3 = data["alpha3"]
    alpha4 = data["alpha4"]

    xy = np.array([list(x), list(y)])
    xy = [-_rotate_body2ned(-p) @ xy[:, i] for i, p in enumerate(psi)]
    xy = np.array(xy)
    x_rot = xy[:, 0]
    y_rot = xy[:, 1]

    for a in range(20):
        offset = a * 201 * 0
        k = 10 * a

        fig, ax = plt.subplots()
        ax.axhline(y=0, color="gray", linestyle="--", label=f"(y={0})")
        ax.axvline(x=0, color="gray", linestyle="--")
        ax.set_xlabel(r"$y^n$", fontsize=14)
        ax.set_ylabel(r"$x^n$", fontsize=14)
        ax.tick_params(axis="x", labelsize=10)
        ax.tick_params(axis="y", labelsize=10)
        ax.set_xlim((-2, 10))
        ax.set_ylim((-6, 4))
        ax.set_aspect("equal", adjustable="datalim")

        legend_handles = []

        t = render_start + k + offset
        rect_x = y_rot[t]
        rect_y = x_rot[t]
        width, height = 5.06, 2.86
        angle = 90 - psi[t]

        rect_x_list = [rect_x, 0]
        rect_y_list = [rect_y, 0]
        angle_list = [angle, 90]

        color = ["blue", "red", "blue", "none"]

        for i in range(2):
            rect = Rectangle(
                (rect_x_list[i] - width / 2, rect_y_list[i] - height / 2),
                width,
                height,
                angle=0,
                edgecolor=color[i],
                facecolor=color[i + 2],
                alpha=0.6,
            )
            trans = (
                transforms.Affine2D().rotate_deg_around(
                    rect_x_list[i], rect_y_list[i], angle_list[i]
                )
                + ax.transData
            )
            rect.set_transform(trans)
            ax.add_patch(rect)

        # Define thruster positions relative to the rectangle's center
        Lx = 1.8
        Ly = 0.8
        thruster_positions = np.array([[Lx, Ly], [Lx, -Ly], [-Lx, Ly], [-Lx, -Ly]])

        rect_x = y_rot[t]
        rect_y = x_rot[t]
        angle = np.deg2rad(90 - psi[t])

        rotation_matrix = np.array(
            [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
        )

        global_thruster_positions = rotation_matrix @ thruster_positions.T
        global_thruster_positions[0, :] += rect_x
        global_thruster_positions[1, :] += rect_y
        ax.plot(rect_x, rect_y, "ko", markersize=3, label="Point Label")

        delta = [n1, n2, n3, n4, alpha1, alpha2, alpha3, alpha4]
        for i in range(4):
            thruster_x = global_thruster_positions[0, i]
            thruster_y = global_thruster_positions[1, i]

            vector_angle = angle - np.deg2rad(delta[i + 4][t])
            dx = np.cos(vector_angle) * 2 * np.pi / 1200 * delta[i][t]
            dy = np.sin(vector_angle) * 2 * np.pi / 1200 * delta[i][t]

            ax.plot(thruster_x, thruster_y, "ko", markersize=3, label="Point Label")

            if i == 0:
                thrust_handle = ax.quiver(
                    thruster_x,
                    thruster_y,
                    dx,
                    dy,
                    angles="xy",
                    scale_units="xy",
                    scale=1,
                    color="lime",
                    alpha=1,
                    label="Thrust [RPM]",
                )
            else:
                ax.quiver(
                    thruster_x,
                    thruster_y,
                    dx,
                    dy,
                    angles="xy",
                    scale_units="xy",
                    scale=1,
                    color="lime",
                    alpha=1,
                )
        legend_handles.append(thrust_handle)

        # Linear Speed Vector
        _linear_speed = np.hypot(u[t], v[t])
        dx = u[t] * np.cos(np.deg2rad(-psi[t])) - v[t] * np.sin(np.deg2rad(-psi[t]))
        dy = u[t] * np.sin(np.deg2rad(-psi[t])) + v[t] * np.cos(np.deg2rad(-psi[t]))
        scale_factor = 2 * np.pi / 3.5
        linear_handle = ax.quiver(
            rect_x,
            rect_y,
            -dx * scale_factor,
            -dy * scale_factor,
            angles="xy",
            scale_units="xy",
            scale=1,
            color="orange",
            alpha=0.8,
            label="Linear Speed [m/s]",
        )
        legend_handles.append(linear_handle)

        angle = 360 * r[t] / 120
        if angle > 0:
            circ_handle = drawCirc(
                ax, 1, rect_x, rect_y, 90 - psi[t], -angle, 0, color_="purple"
            )
        else:
            circ_handle = drawCirc(
                ax, 1, rect_x, rect_y, 90 - psi[t], 0, 360 - angle, color_="purple"
            )

        legend_handles.append(circ_handle)

        ax.legend(handles=legend_handles)
        plt.tight_layout()
        plt.savefig(
            os.path.join(output_dir, f"{run_number}_xy2{a}.pdf"), format="pdf", dpi=1200
        )
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Plot run data from CSV")
    parser.add_argument("--csv", required=True, help="Path to input CSV file")
    parser.add_argument(
        "--output-dir", default="plots/", help="Output directory for PDFs"
    )
    parser.add_argument(
        "--scale", type=float, default=1.0, help="Scale factor for plot elements"
    )
    parser.add_argument(
        "--interval", type=int, default=200, help="Episode interval for shading"
    )
    parser.add_argument("--render-start", type=int, default=0, help="Start time step")
    parser.add_argument("--render-end", type=int, default=1000, help="End time step")
    parser.add_argument(
        "--run-number", type=int, default=1, help="Run number for output filenames"
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    data = load_data(args.csv)

    plot_x(
        data,
        args.scale,
        args.render_start,
        args.render_end,
        args.interval,
        args.run_number,
        args.output_dir,
    )
    plot_y(
        data,
        args.scale,
        args.render_start,
        args.render_end,
        args.interval,
        args.run_number,
        args.output_dir,
    )
    plot_psi(
        data,
        args.scale,
        args.render_start,
        args.render_end,
        args.interval,
        args.run_number,
        args.output_dir,
    )
    plot_u(
        data,
        args.scale,
        args.render_start,
        args.render_end,
        args.interval,
        args.run_number,
        args.output_dir,
    )
    plot_v(
        data,
        args.scale,
        args.render_start,
        args.render_end,
        args.interval,
        args.run_number,
        args.output_dir,
    )
    plot_r(
        data,
        args.scale,
        args.render_start,
        args.render_end,
        args.interval,
        args.run_number,
        args.output_dir,
    )
    plot_n1(
        data,
        args.scale,
        args.render_start,
        args.render_end,
        args.interval,
        args.run_number,
        args.output_dir,
    )
    plot_n2(
        data,
        args.scale,
        args.render_start,
        args.render_end,
        args.interval,
        args.run_number,
        args.output_dir,
    )
    plot_n3(
        data,
        args.scale,
        args.render_start,
        args.render_end,
        args.interval,
        args.run_number,
        args.output_dir,
    )
    plot_n4(
        data,
        args.scale,
        args.render_start,
        args.render_end,
        args.interval,
        args.run_number,
        args.output_dir,
    )
    plot_alpha1(
        data,
        args.scale,
        args.render_start,
        args.render_end,
        args.interval,
        args.run_number,
        args.output_dir,
    )
    plot_alpha2(
        data,
        args.scale,
        args.render_start,
        args.render_end,
        args.interval,
        args.run_number,
        args.output_dir,
    )
    plot_alpha3(
        data,
        args.scale,
        args.render_start,
        args.render_end,
        args.interval,
        args.run_number,
        args.output_dir,
    )
    plot_alpha4(
        data,
        args.scale,
        args.render_start,
        args.render_end,
        args.interval,
        args.run_number,
        args.output_dir,
    )
    plot_combined_states(
        data,
        args.scale,
        args.render_start,
        args.render_end,
        args.interval,
        args.run_number,
        args.output_dir,
    )
    plot_combined_velocities(
        data,
        args.scale,
        args.render_start,
        args.render_end,
        args.interval,
        args.run_number,
        args.output_dir,
    )
    plot_xy_trajectory(
        data, args.render_start, args.render_end, args.run_number, args.output_dir
    )
    plot_xy_thruster(
        data, args.render_start, args.render_end, args.run_number, args.output_dir
    )


if __name__ == "__main__":
    main()
