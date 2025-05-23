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
    ctrl = np.array([[0.0,  0.0],
                     [5.0,  2.0],
                     [10.0, 0.0],
                     [15.0,-3.0],
                     [20.0, 0.0]])
    t      = np.linspace(0, 1, len(ctrl))
    spl_x  = CubicSpline(t, ctrl[:, 0], bc_type="natural")
    spl_y  = CubicSpline(t, ctrl[:, 1], bc_type="natural")

    s_vals = np.linspace(0, 1, N)
    x_d    = spl_x(s_vals)
    y_d    = spl_y(s_vals)
    dx_ds  = spl_x(s_vals, 1)
    dy_ds  = spl_y(s_vals, 1)
    psi_d  = np.arctan2(dy_ds, dx_ds)          # rad
    return x_d, y_d, psi_d, ctrl

def heading_true(psi_d, psi_err):
    """ψ_true = ψ_d – 𝛙̃ ."""
    return psi_d - psi_err

# ------------------------------------------------------------------ main
def main(kind):
    run_number = 5
    df         = pd.read_csv(
        "../data/runs/sim/data_test_spline_20250521_084859.csv"
    )
    # df         = pd.read_csv(
    #     "../data/runs/sim/data_test_north_20250521_085429.csv"
    # )

    # log → body-frame errors
    x_b      = df["x"].values * 10          # m
    y_b      = df["y"].values * 10
    psi_err  = df["psi"].values             # rad

    N        = len(df)
    x_d, y_d, psi_d, ctrl_pts = build_reference(N, kind=kind)

    # rotate error into NED and subtract
    psi_true = heading_true(psi_d, psi_err)
    c, s     = np.cos(psi_true), np.sin(psi_true)
    tilde_n  = np.column_stack((c*x_b - s*y_b, s*x_b + c*y_b))
    x_n      = x_d - tilde_n[:, 0]
    y_n      = y_d - tilde_n[:, 1]

    # ---------------------------------------------------------------- plot
    scale = 10/6.4
    fig, ax = plt.subplots(figsize=(10, 4.8))

    ax.plot(x_d, y_d, lw=2, label=f"Reference ({kind})")
    ax.plot(x_n, y_n, lw=2, label="Realised path")
    if len(ctrl_pts):
        ax.scatter(ctrl_pts[:, 0], ctrl_pts[:, 1],
                   marker="x", s=80, label="Control points")

    ax.set_xlabel(r"$x^n$ [m]", fontsize=14*scale)
    ax.set_ylabel(r"$y^n$ [m]", fontsize=14*scale)
    ax.tick_params(axis="both", labelsize=10*scale)
    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.legend(fontsize=10*scale)
    if kind == 'line':
        ax.set_ylim((-5,5))
    plt.tight_layout()

    outdir = Path("plots/drl/runs")
    outdir.mkdir(parents=True, exist_ok=True)
    fig.savefig(outdir / f"{run_number}_track_{kind}.pdf", dpi=1200)
    plt.close(fig)

# ------------------------------------------------------------------ CLI
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", choices=["spline", "line"],
                        default="spline",
                        help="reference path type")
    args = parser.parse_args()
    main(args.kind)
