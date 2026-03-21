"""
combined_plot.py – reference (spline *or* straight line) vs. realised path
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.interpolate import CubicSpline
import math
import argparse

from matplotlib.patches import Rectangle
import matplotlib.transforms as transforms
from matplotlib.patches import Arc, RegularPolygon
from numpy import radians as rad
from matplotlib.lines import Line2D
import time


# ------------------------------------------------------------------ helpers
def build_reference(N, kind="spline"):
    """
    Return (x_d, y_d, psi_d, control_pts)
        • x_d, y_d .......... arrays length N   (desired North/East)
        • psi_d ............. desired heading (rad) at each sample
        • control_pts ....... only for plotting; [] for straight line
    """
    if kind == "line":
        # straight line:  x∈[0,20], y=0  ⇒  ψ_d = 0
        x_d = np.linspace(0.0, 20.0, N)
        y_d = np.zeros(N)
        psi_d = np.zeros(N)
        ctrl = np.array([[0.0, 0.0], [20.0, 0.0]])  # just for markers
        return x_d, y_d, psi_d, ctrl

    # --- cubic spline (default) ------------------------------------
    ctrl = np.array([[0.0, 0.0], [5.0, 2.0], [10.0, 0.0], [15.0, -3.0], [20.0, 0.0]])
    t = np.linspace(0, 1, len(ctrl))
    spl_x = CubicSpline(t, ctrl[:, 0], bc_type="natural")
    spl_y = CubicSpline(t, ctrl[:, 1], bc_type="natural")

    s_vals = np.linspace(0, 1, N)
    x_d = spl_x(s_vals)
    y_d = spl_y(s_vals)
    dx_ds = spl_x(s_vals, 1)
    dy_ds = spl_y(s_vals, 1)
    psi_d = np.arctan2(dy_ds, dx_ds)  # rad
    return x_d, y_d, psi_d, ctrl


def heading_true(psi_d, psi_err):
    """ψ_true = ψ_d – 𝛙̃ ."""
    return psi_d - psi_err


# ------------------------------------------------------------------ main
def main(csv_path, output_dir, kind, run_number):
    df = pd.read_csv(csv_path)

    # log → body-frame errors
    x_b = df["x"].values * 10  # m
    y_b = df["y"].values * 10
    psi_err = df["psi"].values * np.pi  # rad

    N = len(df)
    x_d, y_d, psi_d, ctrl_pts = build_reference(N, kind=kind)

    # rotate error into NED and subtract
    psi_true = heading_true(psi_d, psi_err)
    c, s = np.cos(psi_true), np.sin(psi_true)
    tilde_n = np.column_stack((c * x_b - s * y_b, s * x_b + c * y_b))
    x_n = x_d - tilde_n[:, 0]
    y_n = y_d - tilde_n[:, 1]

    # ---------------------------------------------------------------- plot
    scale = 10 / 6.4
    fig, ax = plt.subplots(figsize=(10, 4.8))

    ax.plot(x_d, y_d, lw=2, label=f"Reference ({kind})")
    # ax.plot(x_n, y_n, lw=2, label="Realised path")
    if len(ctrl_pts):
        ax.scatter(
            ctrl_pts[:, 0], ctrl_pts[:, 1], marker="x", s=80, label="Control points"
        )

    ax.set_xlabel(r"$x^n$ [m]", fontsize=14 * scale)
    ax.set_ylabel(r"$y^n$ [m]", fontsize=14 * scale)
    ax.tick_params(axis="both", labelsize=10 * scale)
    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.legend(fontsize=10 * scale)
    if kind == "line":
        ax.set_ylim((-5, 5))
    plt.tight_layout()

    outdir = Path(output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    fig.savefig(outdir / f"{run_number}_track_{kind}.pdf", dpi=1200)
    plt.close(fig)

    # ------------------------------------------------------------------ CLI

    scale = 1  # 10/6.4
    render_start = 0
    render_end = 200  # *5
    episode_interval = 200

    x = df["x"] * 10
    y = df["y"] * 10
    psi = df["psi"] * 180
    u = df["u"] * 3.24
    v = df["v"] * 3.24
    r = df["r"] * 112.6
    n1x = df["n1x"] * 900
    n1y = df["n1y"] * 900
    n2x = df["n2x"] * 900
    n2y = df["n2y"] * 900
    n3x = df["n3x"] * 900
    n3y = df["n3y"] * 900
    n4x = df["n4x"] * 900
    n4y = df["n4y"] * 900
    n1 = df["n1d"]
    alpha1 = df["alpha1d"]
    n2 = df["n2d"]
    alpha2 = df["alpha2d"]
    n3 = df["n3d"]
    alpha3 = df["alpha3d"]
    n4 = df["n4d"]
    alpha4 = df["alpha4d"]

    dt = np.arange(len(x))

    def mark_episode(render_start, render_end, ax):
        # Assume render_start = 600, render_end = 2600 and you want 200-step intervals.
        interval = 201
        flag = True  # use this to alternate colors

        for start in range(render_start, render_end, interval):
            end = start + interval
            # Check if the ending goes beyond render_end, if so, truncate:
            if end > render_end:
                end = render_end

            if flag:
                # Add a gray vertical span for these 200 time steps
                ax.axvspan(start, end, facecolor="gray", alpha=0.2)
            else:
                # Else leave it white or add another color if you like
                # ax.axvspan(start, end, facecolor='white', alpha=1.0)  # Typically unnecessary if background is white by default
                pass

            # Flip the flag for the next iteration
            flag = not flag

    def mark_episode2(render_start, render_end, episode_interval, ax):
        # Assume render_start = 600, render_end = 2600 and you want 200-step intervals.
        flag = True  # use this to alternate colors

        for start in range(render_start, render_end, episode_interval):
            end = start + episode_interval
            # Check if the ending goes beyond render_end, if so, truncate:
            if end > render_end:
                end = render_end

            if flag:
                # Add a gray vertical span for these 200 time steps
                ax.axvspan(start, end, facecolor="gray", alpha=0.2)
            else:
                # Else leave it white or add another color if you like
                # ax.axvspan(start, end, facecolor='white', alpha=1.0)  # Typically unnecessary if background is white by default
                pass

            # Flip the flag for the next iteration
            flag = not flag

    # x_d = [0]*200 + [4]*400 + [0]*400
    # y_d = [0]*400 + [4]*200 + [0]*400
    # psi_d = [0]*800 + [180]*200

    x_d = x_n
    y_d = y_n
    psi_d = psi_d / np.pi * 180  # convert to degrees

    # States
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 4.8 * 2))

    # ax1.axhline(y=0, color='red', linestyle='--', linewidth=1.5*scale,label=f"(y={0})")
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
    mark_episode2(render_start, render_end, episode_interval, ax1)

    # ax2.axhline(y=0, color='red', linestyle='--', linewidth=1.5*scale,label=f"(y={0})")
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
    mark_episode2(render_start, render_end, episode_interval, ax2)

    # ax3.axhline(y=0, color='red', linestyle='--', linewidth=1.5*scale,label=f"(y={0})")
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
    mark_episode2(render_start, render_end, episode_interval, ax3)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/{run_number}_states.pdf", format="pdf", dpi=1200)
    plt.close(fig)

    # States2
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
    ax1.set_ylim(-0.1, 0.7)
    mark_episode2(render_start, render_end, episode_interval, ax1)

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
    ax2.set_ylim(-0.4, 0.4)
    mark_episode2(render_start, render_end, episode_interval, ax2)

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
    ax3.set_ylim(-30, 30)
    mark_episode2(render_start, render_end, episode_interval, ax3)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/{run_number}_states2.pdf", format="pdf", dpi=1200)
    plt.close(fig)

    # X
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
    mark_episode(render_start, render_end, ax)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/{run_number}_x.pdf", format="pdf", dpi=1200)
    plt.close(fig)

    # Y
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
    mark_episode(render_start, render_end, ax)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/{run_number}_y.pdf", format="pdf", dpi=1200)
    plt.close(fig)

    # psi
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
    mark_episode(render_start, render_end, ax)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/{run_number}_psi.pdf", format="pdf", dpi=1200)
    plt.close(fig)

    # U
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
    mark_episode(render_start, render_end, ax)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/{run_number}_u.pdf", format="pdf", dpi=1200)
    plt.close(fig)

    # V
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
    mark_episode(render_start, render_end, ax)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/{run_number}_v.pdf", format="pdf", dpi=1200)
    plt.close(fig)

    # R
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
    mark_episode(render_start, render_end, ax)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/{run_number}_r.pdf", format="pdf", dpi=1200)
    plt.close(fig)

    # N1
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
    mark_episode(render_start, render_end, ax)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/{run_number}_n1.pdf", format="pdf", dpi=1200)
    plt.close(fig)

    # N2
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
    mark_episode(render_start, render_end, ax)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/{run_number}_n2.pdf", format="pdf", dpi=1200)
    plt.close(fig)

    # N3
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
    mark_episode(render_start, render_end, ax)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/{run_number}_n3.pdf", format="pdf", dpi=1200)
    plt.close(fig)

    # N4
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
    mark_episode(render_start, render_end, ax)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/{run_number}_n4.pdf", format="pdf", dpi=1200)
    plt.close(fig)

    # ALPHA1
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
    mark_episode(render_start, render_end, ax)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/{run_number}_alpha1.pdf", format="pdf", dpi=1200)
    plt.close(fig)

    # ALPHA2
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
    mark_episode(render_start, render_end, ax)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/{run_number}_alpha2.pdf", format="pdf", dpi=1200)
    plt.close(fig)

    # ALPHA3
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
    mark_episode(render_start, render_end, ax)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/{run_number}_alpha3.pdf", format="pdf", dpi=1200)
    plt.close(fig)

    # ALPHA4
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
    mark_episode(render_start, render_end, ax)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/{run_number}_alpha4.pdf", format="pdf", dpi=1200)
    plt.close(fig)

    def _rotate_body2ned(psi):
        psi = np.deg2rad(psi)
        c = np.cos(psi)
        s = np.sin(psi)
        return np.array([[c, -s], [s, c]])

    xy = np.array([list(x), list(y)])
    print(xy.shape)
    print(xy.T.shape)
    xy = [-_rotate_body2ned(-p) @ xy[:, i] for i, p in enumerate(psi)]
    xy = np.array(xy)
    print(xy.shape)
    x = xy[:, 0]
    y = xy[:, 1]

    x = x_d + x
    y = y_d + y
    psi = psi_d + psi

    for a in range(1):
        offset = a * 200
        k = 10 * 0

        fig, ax = plt.subplots(figsize=(4.8, 4.8))
        print(xy)
        ax.axhline(y=0, color="gray", linestyle="--", label=f"(y={0})")
        ax.axvline(x=0, color="gray", linestyle="--")
        ax.plot(
            y[render_start + k + offset : render_start + offset + 200],
            x[render_start + k + offset : render_start + offset + 200],
            color="red",
        )
        ax.set_xlabel(r"$y^n$", fontsize=14)
        ax.set_ylabel(r"$x^n$", fontsize=14)
        ax.tick_params(axis="x", labelsize=10)
        ax.tick_params(axis="y", labelsize=10)
        ax.set_xlim((-5, 5))
        ax.set_ylim((-4, 15))
        ax.set_aspect("equal")  # , adjustable='datalim')

        my_range = [render_start + k + offset, render_start + offset + 199]

        # Add rectangles every 20 time steps
        for t in my_range:
            # Define rectangle properties
            rect_x = y[t]  # Use the `y` coordinate for the rectangle center
            rect_y = x[t]  # Use the `x` coordinate for the rectangle center
            width, height = 5.06, 2.86  # Dimensions of the rectangle
            angle = 90 - psi[t]  # Rotation angle in degrees

            # Create the rectangle
            rect = Rectangle(
                (rect_x - width / 2, rect_y - height / 2),
                width,
                height,
                angle=0,
                color="blue",
                alpha=0.6,
            )

            # Rotate the rectangle around its center
            trans = (
                transforms.Affine2D().rotate_deg_around(rect_x, rect_y, angle)
                + ax.transData
            )
            rect.set_transform(trans)

            # Add the rectangle to the plot
            ax.add_patch(rect)

        # Add vectors every 20 time steps
        for t in range(render_start + k + offset, render_start + offset + 200, 10):
            # Define vector properties
            vector_x = y[t]  # Starting point of the vector (y-coordinate)
            vector_y = x[t]  # Starting point of the vector (x-coordinate)
            angle = np.deg2rad(
                90 - psi[t]
            )  # Convert angle to radians for vector calculation

            # Vector components
            dx = np.cos(angle)  # x-component of the vector
            dy = np.sin(angle)  # y-component of the vector

            # Scale the vector length (optional)
            scale_factor = 0.5 * 2
            dx *= scale_factor
            dy *= scale_factor

            # Plot the vector
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
        plt.savefig(f"{output_dir}/{run_number}_xy{a}.pdf", format="pdf", dpi=1200)
        plt.close(fig)

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
            endX = centX + (radius / 2) * np.cos(
                rad(ang + angle_)
            )  # Do trig to determine end position
            endY = centY + (radius / 2) * np.sin(rad(ang + angle_))
            ax.add_patch(  # Create triangle as arrow head
                RegularPolygon(
                    (endX, endY),  # (x,y)
                    3,  # number of vertices
                    radius=radius / 9,  # radius
                    orientation=rad(angle_ + ang),  # orientation
                    color=color_,
                )
            )
        else:
            ang = theta1_

            # ========Create the arrow head
            endX = centX + (radius / 2) * np.cos(
                rad(ang + angle_)
            )  # Do trig to determine end position
            endY = centY + (radius / 2) * np.sin(rad(ang + angle_))
            ax.add_patch(  # Create triangle as arrow head
                RegularPolygon(
                    (endX, endY),  # (x,y)
                    3,  # number of vertices
                    radius=radius / 9,  # radius
                    orientation=rad(angle_ + ang + 180),  # orientation
                    color=color_,
                )
            )

        return legend_element
        # ax.set_xlim([centX-radius,centY+radius]) and ax.set_ylim([centY-radius,centY+radius])
        # Make sure you keep the axes scaled or else arrow will distort

    for a in range(20):
        offset = a * 201 * 0
        k = 10 * a

        fig, ax = plt.subplots()
        print(xy)
        ax.axhline(y=0, color="gray", linestyle="--", label=f"(y={0})")
        ax.axvline(x=0, color="gray", linestyle="--")
        # ax.plot(y[render_start+k+offset:render_start+offset+201],x[render_start+k+offset:render_start+offset+201], color='red')
        ax.set_xlabel(r"$y^n$", fontsize=14)
        ax.set_ylabel(r"$x^n$", fontsize=14)
        ax.tick_params(axis="x", labelsize=10)
        ax.tick_params(axis="y", labelsize=10)
        ax.set_xlim((-2, 10))
        ax.set_ylim((-6, 4))
        ax.set_aspect("equal", adjustable="datalim")

        legend_handles = []

        my_range = [render_start + k + offset, render_end]

        t = render_start + k + offset
        # Add rectangles every 20 time steps
        # Define rectangle properties
        rect_x = y[t]  # Use the `y` coordinate for the rectangle center
        rect_y = x[t]  # Use the `x` coordinate for the rectangle center
        width, height = 5.06, 2.86  # Dimensions of the rectangle
        angle = 90 - psi[t]  # Rotation angle in degrees

        rect_x_list = [rect_x, 0]
        rect_y_list = [rect_y, 0]
        angle_list = [angle, 90]

        color = ["blue", "red", "blue", "none"]

        for i in range(2):
            # Create the rectangle
            rect = Rectangle(
                (rect_x_list[i] - width / 2, rect_y_list[i] - height / 2),
                width,
                height,
                angle=0,
                edgecolor=color[i],
                facecolor=color[i + 2],
                alpha=0.6,
            )

            # Rotate the rectangle around its center
            trans = (
                transforms.Affine2D().rotate_deg_around(
                    rect_x_list[i], rect_y_list[i], angle_list[i]
                )
                + ax.transData
            )
            rect.set_transform(trans)

            # Add the rectangle to the plot
            ax.add_patch(rect)

        # Define thruster positions relative to the rectangle's center
        Lx = 1.8
        Ly = 0.8
        thruster_positions = np.array([[Lx, Ly], [Lx, -Ly], [-Lx, Ly], [-Lx, -Ly]])

        rect_x = y[t]
        rect_y = x[t]
        angle = np.deg2rad(90 - psi[t])  # Convert angle to radians for rotation

        # Rotation matrix for the rectangle's orientation
        rotation_matrix = np.array(
            [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
        )

        # Compute global coordinates of the thruster positions
        global_thruster_positions = rotation_matrix @ thruster_positions.T
        global_thruster_positions[0, :] += rect_x
        global_thruster_positions[1, :] += rect_y
        ax.plot(rect_x, rect_y, "ko", markersize=3, label="Point Label")

        # Debugging output
        print(f"Time step {t}: Rectangle Center ({rect_x}, {rect_y})")
        print(f"Global Thruster Positions:\n{global_thruster_positions}")

        delta = [n1, n2, n3, n4, alpha1, alpha2, alpha3, alpha4]
        # Plot vectors originating from thruster positions
        for i in range(4):
            thruster_x = global_thruster_positions[0, i]
            thruster_y = global_thruster_positions[1, i]

            # Define vector components (customize these as needed)
            vector_angle = angle - np.deg2rad(
                delta[i + 4][t]
            )  # Assuming the vector points along the rectangle's orientation
            dx = (
                np.cos(vector_angle) * 2 * np.pi / 1200 * delta[i][t]
            )  # Scale vector length as needed
            dy = np.sin(vector_angle) * 2 * np.pi / 1200 * delta[i][t]
            print(f"alpha_{i}: {delta[i + 4][t]}, n_{i}: {delta[i][t]}")

            ax.plot(thruster_x, thruster_y, "ko", markersize=3, label="Point Label")

            # Plot the vector
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
        print(f"u: {u[t]}, v: {v[t]}, r: {r[t]}")
        linear_speed = np.hypot(u[t], v[t])  # Compute linear speed magnitude
        dx = u[t] * np.cos(np.deg2rad(-psi[t])) - v[t] * np.sin(
            np.deg2rad(-psi[t])
        )  # x-component of the linear velocity
        dy = u[t] * np.sin(np.deg2rad(-psi[t])) + v[t] * np.cos(
            np.deg2rad(-psi[t])
        )  # y-component of the linear velocity
        # dx = v[t]
        # dy = u[t]
        scale_factor = 2 * np.pi / 3.5  # Adjust this for arrow length scaling
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
        plt.savefig(f"{output_dir}/{run_number}_xy2{a}.pdf", format="pdf", dpi=1200)
        plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot spline reference vs realised path."
    )
    parser.add_argument(
        "--csv", type=str, required=True, help="path to the run CSV file"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="plots/",
        help="directory to save output plots (default: plots/)",
    )
    parser.add_argument(
        "--kind",
        type=str,
        default="hermite_derivative",
        help="reference path type (default: hermite_derivative)",
    )
    parser.add_argument(
        "--run-number",
        type=int,
        default=1,
        help="run number used in output filenames (default: 1)",
    )
    args = parser.parse_args()
    main(args.csv, args.output_dir, args.kind, args.run_number)
