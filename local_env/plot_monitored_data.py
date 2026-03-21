##### Plot figures
######

import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description="Plot monitored training data.")
    parser.add_argument(
        "--csv", type=str, required=True, help="Path to the monitor CSV file"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="plots/",
        help="Directory to save output plots (default: plots/)",
    )
    parser.add_argument(
        "--max-reward",
        type=float,
        default=1000,
        help="Value for the horizontal reference line (default: 1000)",
    )
    args = parser.parse_args()

    scale = 1  # Change the scale 0.7/0.4

    # Import
    monitor_data = pd.read_csv(args.csv, skiprows=1)  # Skip the first row (header)

    r = monitor_data["r"]
    e = np.arange(len(r))
    t = monitor_data["t"]
    l = monitor_data["l"]

    colors = ["green" if val >= 0 else "red" for val in r]

    # Ensure output directory exists
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # Plot

    fig, ax = plt.subplots()

    ax.bar(e, r, color=colors)
    ax.axhline(
        y=args.max_reward,
        color="black",
        linestyle="--",
        label=f"Maximum Reward (y={args.max_reward})",
    )
    ax.legend()
    ax.set_xlabel(r"Episode", fontsize=14 * scale)
    ax.set_ylabel(r"Episode Reward", fontsize=14 * scale)
    ax.tick_params(axis="x", labelsize=10 * scale)
    ax.tick_params(axis="y", labelsize=10 * scale)

    plt.tight_layout()
    plt.savefig(
        os.path.join(args.output_dir, "episode_plot_3_r.pdf"), format="pdf", dpi=1200
    )
    plt.close(fig)

    fig, ax = plt.subplots()

    ax.plot(t, r, color="green")
    ax.axhline(
        y=args.max_reward,
        color="black",
        linestyle="--",
        label=f"Maximum Reward (y={args.max_reward})",
    )
    ax.legend()
    ax.set_xlabel(r"Time [s]", fontsize=14 * scale)
    ax.set_ylabel(r"Episode Reward", fontsize=14 * scale)
    ax.tick_params(axis="x", labelsize=10 * scale)
    ax.tick_params(axis="y", labelsize=10 * scale)

    plt.tight_layout()
    plt.savefig(
        os.path.join(args.output_dir, "episode_plot_3_r2.pdf"), format="pdf", dpi=1200
    )
    plt.close(fig)

    # Plot

    fig, ax = plt.subplots()

    ax.bar(e, l, color="blue")
    ax.axhline(
        y=args.max_reward,
        color="black",
        linestyle="--",
        label=f"Maximum Length (y={args.max_reward})",
    )
    ax.legend()
    ax.set_xlabel(r"Episode", fontsize=14 * scale)
    ax.set_ylabel(r"Episode Length $[\Delta t]$", fontsize=14 * scale)
    ax.tick_params(axis="x", labelsize=10 * scale)
    ax.tick_params(axis="y", labelsize=10 * scale)

    plt.tight_layout()
    plt.savefig(
        os.path.join(args.output_dir, "episode_plot_3_l.pdf"), format="pdf", dpi=1200
    )
    plt.close(fig)


if __name__ == "__main__":
    main()
