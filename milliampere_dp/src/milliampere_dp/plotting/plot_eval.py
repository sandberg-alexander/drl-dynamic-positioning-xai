from __future__ import annotations

import argparse
import glob
import os
import re
import traceback

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_evaluation_rewards(
    run_dir, output_file="plots/drl/eval/evaluation_rewards_3_r.pdf"
):
    """
    Reads monitor.csv files from evaluation subdirectories, calculates average
    rewards, and plots them.

    Args:
        run_dir (str): The main directory of the training run
                       (e.g., '../data/models/training_20250328_145357').
        output_file (str): The name of the PDF file to save the plot.
    """

    eval_base_dir = os.path.join(run_dir, "evaluation")
    if not os.path.isdir(eval_base_dir):
        print(f"Error: Evaluation directory not found at '{eval_base_dir}'")
        return

    # Find all model-specific monitor directories
    monitor_dirs = glob.glob(os.path.join(eval_base_dir, "monitor_PPO_*"))

    if not monitor_dirs:
        print(f"Error: No 'monitor_PPO_*' directories found in '{eval_base_dir}'")
        return

    results = []  # Store tuples of (steps, mean_reward, model_name)

    print(f"Found {len(monitor_dirs)} monitor directories. Processing...")

    for monitor_dir in monitor_dirs:
        monitor_csv_path = os.path.join(monitor_dir, "monitor.csv")
        model_name = os.path.basename(monitor_dir).replace(
            "monitor_", ""
        )  # Extract model name

        if not os.path.exists(monitor_csv_path):
            print(f"Warning: monitor.csv not found in {monitor_dir}. Skipping.")
            continue

        try:
            # Read the monitor CSV, skipping the initial JSON comment line
            monitor_data = pd.read_csv(monitor_csv_path)

            if "r" not in monitor_data.columns:
                print(
                    f"Warning: 'r' (reward) column not found "
                    f"in {monitor_csv_path}. Skipping."
                )
                continue

            if monitor_data.empty:
                print(
                    f"Warning: {monitor_csv_path} is empty "
                    f"after skipping header. Skipping."
                )
                continue

            # Calculate the mean reward for this model's evaluation episodes
            mean_reward = monitor_data["r"].mean()

            # Extract training steps from the model name using regex
            match = re.search(r"steps_(\d+)", model_name)
            if match:
                steps = int(match.group(1))
                results.append((steps, mean_reward, model_name))
                print(
                    f"Processed {model_name}: "
                    f"Steps={steps}, "
                    f"Mean Reward={mean_reward:.2f}"
                )
            else:
                print(
                    f"Warning: Could not extract steps "
                    f"from model name '{model_name}'. "
                    f"Skipping."
                )

        except pd.errors.EmptyDataError:
            print(
                f"Warning: {monitor_csv_path} contained "
                f"no data or only headers. Skipping."
            )
        except Exception as e:
            print(f"Error processing {monitor_csv_path}: {e}")
            traceback.print_exc()  # Use traceback for detailed errors

    if not results:
        print("Error: No valid evaluation results found to plot.")
        return

    # Sort results by the number of training steps
    results.sort(key=lambda x: x[0])

    # Unpack sorted results for plotting
    steps_array = np.array([res[0] for res in results])
    mean_rewards_array = np.array([res[1] for res in results])

    # --- Plotting ---
    print("\nGenerating plot...")
    scale = 1.0  # Adjust scaling if needed
    colors = ["green" if val >= 0 else "red" for val in mean_rewards_array]

    # Find the best model based on mean reward
    best_mean_reward = np.max(mean_rewards_array)
    best_model_indices = np.where(mean_rewards_array == best_mean_reward)[0]
    # Handle multiple best models if they have the same mean reward
    best_model_steps = steps_array[best_model_indices]

    fig, ax = plt.subplots()

    # Plot bars for mean rewards
    ax.bar(
        steps_array,
        mean_rewards_array,
        color=colors,
        width=np.min(np.diff(steps_array)) * 0.8 if len(steps_array) > 1 else 10000,
    )

    # Highlight the best model(s) with a star
    ax.plot(
        best_model_steps,
        mean_rewards_array[best_model_indices],
        "*",
        markersize=12,
        color="purple",
        label=f"Best Mean Reward ({best_mean_reward:.2f})",
    )

    # Optional: Add a line for maximum possible reward if known
    max_possible_reward = 1000
    ax.axhline(
        y=max_possible_reward,
        color="black",
        linestyle="--",
        label=f"Target/Max Reward ({max_possible_reward})",
    )

    ax.set_xlabel("Training Steps", fontsize=14 * scale)
    ax.set_ylabel("Mean Evaluation Reward", fontsize=14 * scale)
    ax.tick_params(axis="x", labelsize=10 * scale)
    ax.tick_params(axis="y", labelsize=10 * scale)
    ax.legend(fontsize=10 * scale)

    plt.tight_layout()

    # Ensure output directory exists
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    try:
        plt.savefig(output_file, format="pdf", dpi=1200)
        print(f"Plot saved successfully to {output_file}")
    except Exception as e:
        print(f"Error saving plot to {output_file}: {e}")

    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Plot average evaluation rewards for a training run."
    )
    parser.add_argument(
        "--dir",
        dest="run_dir",
        type=str,
        required=True,
        help="Path to the main training run directory",
    )
    parser.add_argument(
        "--output",
        dest="output_file",
        type=str,
        default="plots/drl/eval/evaluation_rewards.pdf",
        help="Output filename for the plot",
    )

    args = parser.parse_args()
    plot_evaluation_rewards(args.run_dir, args.output_file)


if __name__ == "__main__":
    main()
