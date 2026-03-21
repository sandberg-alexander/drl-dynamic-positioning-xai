#!/usr/bin/env python3

# Same imports as before...
import os
import rospy
import milliampere_env  # noqa: F401 -- registers MilliAmpere1-v1
from milliampere_dp import __version__
from stable_baselines3 import PPO
import gymnasium as gym
from stable_baselines3.common.monitor import Monitor
import csv
import json

# import signal # No custom signal handler needed
import sys
import glob
import re
import argparse
import traceback
import time


# --- EvaluationTracker Class (Keep as is) ---
class EvaluationTracker:
    # ... (no changes needed) ...
    def __init__(self, eval_dir):
        self.tracker_file = os.path.join(eval_dir, "evaluation_progress.json")
        self.progress = self._load_progress()
        self._print_current_progress()

    def _print_current_progress(self):
        print(f"Current tracker state: {self.progress}")

    def _load_progress(self):
        try:
            if os.path.exists(self.tracker_file):
                with open(self.tracker_file, "r") as f:
                    return json.load(f)
        except Exception as e:
            print(
                f"Error loading progress file: {e}\nStarting with fresh progress tracking"
            )
        return {"completed_models": [], "current_model": None, "completed_episodes": 0}

    def save_progress(self):
        time.sleep(0.1)
        try:
            os.makedirs(os.path.dirname(self.tracker_file), exist_ok=True)
            with open(self.tracker_file, "w") as f:
                json.dump(self.progress, f, indent=2)
                f.flush()
                os.fsync(f.fileno())
            print(
                f"Progress saved: model={os.path.basename(self.progress['current_model']) if self.progress['current_model'] else 'None'}, episodes={self.progress['completed_episodes']}"
            )
        except Exception as e:
            print(f"Error saving progress: {e}")
            traceback.print_exc()

    def is_model_completed(self, model_path):
        return model_path in self.progress["completed_models"]

    def get_current_model(self):
        return self.progress["current_model"]

    def get_completed_episodes(self):
        return self.progress["completed_episodes"]

    def start_model_evaluation(self, model_path):
        if self.progress["current_model"] != model_path:
            print(f"Tracker: Starting evaluation for {os.path.basename(model_path)}")
            self.progress["current_model"] = model_path
            self.progress["completed_episodes"] = 0
            self.save_progress()
        else:
            print(f"Tracker: Resuming evaluation for {os.path.basename(model_path)}")

    def complete_episode(self):
        self.progress["completed_episodes"] += 1
        self.save_progress()

    def complete_model_evaluation(self):
        current_model = self.progress["current_model"]
        if current_model:
            print(
                f"Tracker: Marking model {os.path.basename(current_model)} as complete."
            )
            if current_model not in self.progress["completed_models"]:
                self.progress["completed_models"].append(current_model)
            self.progress["current_model"] = None
            self.progress["completed_episodes"] = 0
            self.save_progress()

    def get_all_completed_models(self):
        return self.progress["completed_models"]


# --- append_monitor_data Function (Keep as is) ---
def append_monitor_data(temp_csv_path, main_csv_path):
    # ... (no changes needed) ...
    print(f"Attempting to append data from {temp_csv_path} to {main_csv_path}")
    if not os.path.exists(temp_csv_path):
        print(f"Temporary monitor file {temp_csv_path} not found.")
        return False
    temp_data_rows = []
    header = None
    try:
        with open(temp_csv_path, "r") as temp_f:
            first_line = temp_f.readline()
            if first_line.startswith("#"):
                header_line = temp_f.readline().strip()
            else:
                header_line = first_line.strip()
            if header_line:
                header = header_line.split(",")
                reader = csv.reader(temp_f)
                temp_data_rows = list(reader)
            else:
                print(f"Warning: Temp file {temp_csv_path} empty/header missing.")
    except Exception as e:
        print(f"Error reading temp file {temp_csv_path}: {e}")
        traceback.print_exc()
        return False
    if not temp_data_rows:
        print(f"No data rows found in {temp_csv_path}. Cleaning up.")
        try:
            os.remove(temp_csv_path)
            temp_dir = os.path.dirname(temp_csv_path)
            os.rmdir(temp_dir)
            print(f"Removed empty temp file/dir: {temp_dir}")
        except OSError as e:
            print(f"Error removing temp file/dir {temp_csv_path}: {e}")
        return True
    main_dir = os.path.dirname(main_csv_path)
    os.makedirs(main_dir, exist_ok=True)
    needs_header = True
    if os.path.exists(main_csv_path):
        try:
            if os.path.getsize(main_csv_path) > 0:
                needs_header = False
        except OSError as e:
            print(
                f"Warning: size check failed {main_csv_path}: {e}. Assuming header needed."
            )
    try:
        with open(main_csv_path, "a", newline="") as main_f:
            writer = csv.writer(main_f)
            if needs_header and header:
                print(f"Writing header to {main_csv_path}")
                writer.writerow(header)
            print(f"Appending {len(temp_data_rows)} data rows to {main_csv_path}")
            writer.writerows(temp_data_rows)
    except Exception as e:
        print(f"Error writing main file {main_csv_path}: {e}")
        traceback.print_exc()
        return False
    try:
        print(f"Removing temporary monitor file: {temp_csv_path}")
        os.remove(temp_csv_path)
        temp_dir = os.path.dirname(temp_csv_path)
        if not os.listdir(temp_dir):
            os.rmdir(temp_dir)
            print(f"Removed empty temp dir: {temp_dir}")
    except OSError as e:
        print(f"Error cleaning up temp {temp_csv_path}: {e}")
    return True


# --- evaluate_one_model Function (Modified evaluate_model) ---
def evaluate_one_model(model_path, tracker, eval_dir, n_eval_episodes=10):
    """
    Evaluates a SINGLE specified model, resuming if necessary.
    Uses temporary monitor file and appends at the end.
    """
    model_name = os.path.basename(model_path)
    model_monitor_dir = os.path.join(eval_dir, f"monitor_{model_name}")
    main_monitor_filepath = os.path.join(model_monitor_dir, "monitor.csv")

    # Check if this specific model is already fully completed
    if tracker.is_model_completed(model_path):
        print(
            f"Model {model_name} is already marked as completed in tracker. Skipping."
        )
        return True  # Indicate successful skip

    # Tell tracker we are starting/resuming this specific model
    tracker.start_model_evaluation(model_path)
    start_episode = tracker.get_completed_episodes()

    # Double-check if resuming means it's already done
    if start_episode >= n_eval_episodes:
        print(
            f"Model {model_name} already has {start_episode} completed episodes (>= target {n_eval_episodes}). Marking complete."
        )
        tracker.complete_model_evaluation()
        return True  # Indicate successful completion

    # --- Setup Temp File ---
    env = None
    timestamp = int(time.time() * 1000)
    temp_dir = os.path.join(model_monitor_dir, f"temp_{timestamp}")
    temp_monitor_filepath = os.path.join(temp_dir, "monitor.csv")
    info_keywords_to_log = ("target_pose",)
    should_append_data = False
    evaluation_successful = (
        False  # Track if the evaluation part ran without critical errors
    )

    try:
        # --- Load Model ---
        print(f"Loading model {model_name}...")
        model = PPO.load(model_path, device="auto")
        print("Model loaded.")

        # --- Create Environment ---
        print(f"Creating environment instance for model {model_name}...")
        if not rospy.core.is_initialized():
            rospy.init_node("drl_eval", anonymous=True)
        env = gym.make(
            "MilliAmpere1-v1", config_path="/app/configs/env/legacy/v5_equivalent.yaml"
        )
        print("Environment created.")
        os.makedirs(temp_dir, exist_ok=True)

        # --- Initialize Monitor (to temp file) ---
        print(f"Monitor will write temporary data to: {temp_monitor_filepath}")
        env = Monitor(
            env, filename=temp_monitor_filepath, info_keywords=info_keywords_to_log
        )
        print("Monitor wrapper applied.")

        # --- Start Episode Loop ---
        print(
            f"Starting evaluation loop from episode {start_episode + 1} to {n_eval_episodes}"
        )
        successful_episodes_this_run = 0

        seed = 42

        for i in range(start_episode, n_eval_episodes):
            current_episode_num = i + 1
            print(f"\n--- Running Episode {current_episode_num}/{n_eval_episodes} ---")

            episode_seed = seed + i
            print(f"------EPISODE SEED------\n{episode_seed} {start_episode} {i}")
            obs, info = env.reset(seed=episode_seed)
            done = False
            truncated = False
            episode_reward = 0
            episode_length = 0

            while not (done or truncated):
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated
                episode_reward += reward
                episode_length += 1

            print(
                f"--- Episode {current_episode_num} Finished (Reward: {episode_reward:.2f}, Length: {episode_length}) ---"
            )
            tracker.complete_episode()  # Update progress immediately
            successful_episodes_this_run += 1
            should_append_data = True

        # If loop finishes naturally, evaluation was successful up to this point
        evaluation_successful = True

    except KeyboardInterrupt:
        print(f"\nKeyboardInterrupt caught during evaluation of model {model_name}.")
        print("Stopping evaluation FOR THIS MODEL. Will save partial results.")
        if successful_episodes_this_run > 0:
            should_append_data = True
        # Let finally block handle cleanup. Return False to indicate incomplete run.
        return False

    except Exception as e:
        print(f"\n!!!!! ERROR during evaluation of model {model_name}: {e} !!!!!")
        traceback.print_exc()
        print(
            "Stopping evaluation FOR THIS MODEL due to error. Will save partial results if any."
        )
        if successful_episodes_this_run > 0:
            should_append_data = True
        # Let finally block handle cleanup. Return False to indicate error.
        return False

    finally:
        # --- Cleanup and Data Appending ---
        if env:
            print("Closing environment instance (flushes final monitor data)...")
            try:
                env.close()
                time.sleep(0.5)  # Close and wait
            except Exception as e:
                print(f"Error closing environment: {e}")
        if should_append_data:
            append_monitor_data(temp_monitor_filepath, main_monitor_filepath)
        else:
            # Clean up temp if no data generated
            if os.path.exists(temp_monitor_filepath):
                try:
                    os.remove(temp_monitor_filepath)
                    os.rmdir(os.path.dirname(temp_monitor_filepath))
                except OSError as e:
                    print(f"Error cleaning up unused temp: {e}")

        # --- Finalize tracker state ONLY if evaluation was successful ---
        if evaluation_successful:
            final_completed_count = tracker.get_completed_episodes()
            if final_completed_count >= n_eval_episodes:
                print(
                    f"All {n_eval_episodes} episodes completed for model {model_name}."
                )
                tracker.complete_model_evaluation()  # Mark as fully done
            else:  # Should not happen if loop completes, but safety check
                print(
                    f"Loop finished, but only {final_completed_count}/{n_eval_episodes} episodes recorded. Check logic."
                )
        else:
            # If stopped by error or interrupt, just print state, don't mark complete
            final_completed_count = tracker.get_completed_episodes()
            print(
                f"Evaluation stopped for model {model_name} after {final_completed_count}/{n_eval_episodes} total episodes."
            )

    # Return True if the natural end of evaluation (all episodes done) was reached successfully
    return evaluation_successful and (
        tracker.get_completed_episodes() >= n_eval_episodes
    )


# --- Main Logic (Finds the *next* model and evaluates it) ---
def main():
    parser = argparse.ArgumentParser(description="Evaluate models for a training run")
    parser.add_argument(
        "--dir",
        "--run_dir",
        dest="run_dir",
        type=str,
        required=True,
        help="Training run directory (required)",
    )
    parser.add_argument(
        "--episodes",
        "--n_eval_episodes",
        dest="n_eval_episodes",
        type=int,
        default=10,
        help="Number of episodes to evaluate each model",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Evaluate all unevaluated models in a loop (replaces run_evaluation.sh)",
    )
    args = parser.parse_args()

    print(f"""
    ### ___T_ ################################################
       | n n |                   _      ____  ____  _
       |__E__|      _ __ ___    / \    |  _ \|  _ \| |
    >===]__o[===<  | '_ ` _ \  / _ \   | | | | |_) | |
        [o__]      | | | | | |/ ___ \  | |_| |  _ <| |___
        /7 [|      |_| |_| |_/_/   \_\ |____/|_| \_\_____| v{__version__}
      \/7  [|_     eval.py
    ##########################################################

    Starting evaluation run{" (--all mode)" if args.all else ""}...
    """)

    models_dir = os.path.join(args.run_dir, "models")
    eval_dir = os.path.join(args.run_dir, "evaluation")
    os.makedirs(eval_dir, exist_ok=True)  # Ensure evaluation dir exists

    print(f"Looking for models in: {models_dir}")
    print(f"Evaluation data directory: {eval_dir}")
    print(f"Target episodes per model: {args.n_eval_episodes}")

    tracker = EvaluationTracker(eval_dir)

    def find_next_model(tracker, models_dir, n_eval_episodes):
        """Find the next model to evaluate, checking resume state first."""
        model_to_evaluate = None

        # 1. Check if resuming a specific model
        current_model_path = tracker.get_current_model()
        if current_model_path:
            if os.path.exists(current_model_path):
                if tracker.get_completed_episodes() >= n_eval_episodes:
                    print(
                        f"Model {os.path.basename(current_model_path)} already has enough episodes. Marking complete."
                    )
                    tracker.complete_model_evaluation()
                else:
                    print(
                        f"Resuming evaluation for model: {os.path.basename(current_model_path)}"
                    )
                    model_to_evaluate = current_model_path
            else:
                print(
                    f"Warning: Tracker's current model '{current_model_path}' not found. Resetting."
                )
                tracker.progress["current_model"] = None
                tracker.progress["completed_episodes"] = 0
                tracker.save_progress()

        # 2. If not resuming, find the newest model not yet completed
        if not model_to_evaluate:
            print("Searching for the next unevaluated model...")
            model_files = []
            for ext in ["zip"]:
                model_files.extend(glob.glob(os.path.join(models_dir, f"PPO_*{ext}")))
            model_files = [
                f for f in model_files if "FINAL" not in f and "BEST" not in f
            ]

            def extract_steps(filename):
                match = re.search(r"steps_(\d+)", filename)
                return int(match.group(1)) if match else 0

            model_files.sort(key=extract_steps, reverse=True)

            completed_models = tracker.get_all_completed_models()
            for model_path in model_files:
                if model_path not in completed_models:
                    model_to_evaluate = model_path
                    print(
                        f"Found next model to evaluate: {os.path.basename(model_to_evaluate)}"
                    )
                    break

        return model_to_evaluate

    def evaluate_and_report(model_to_evaluate, tracker, eval_dir, n_eval_episodes):
        """Evaluate a single model, returning True on success."""
        try:
            success = evaluate_one_model(
                model_to_evaluate, tracker, eval_dir, n_eval_episodes
            )
            if success:
                print(
                    f"\nSuccessfully completed evaluation for: {os.path.basename(model_to_evaluate)}"
                )
            else:
                print(
                    f"\nEvaluation for {os.path.basename(model_to_evaluate)} was interrupted or failed."
                )
            return success
        except KeyboardInterrupt:
            print("\nKeyboardInterrupt caught. Exiting.")
            return False
        except Exception as e:
            print(f"\nFATAL ERROR during evaluation: {e}")
            traceback.print_exc()
            if tracker.get_current_model() == model_to_evaluate:
                tracker.progress["current_model"] = model_to_evaluate
                tracker.save_progress()
            return False

    # --- Main evaluation logic ---
    if args.all:
        # Loop through all unevaluated models
        print("Evaluating all unevaluated models...")
        while True:
            model_to_evaluate = find_next_model(
                tracker, models_dir, args.n_eval_episodes
            )
            if model_to_evaluate is None:
                print("\nAll models evaluated.")
                sys.exit(0)
            success = evaluate_and_report(
                model_to_evaluate, tracker, eval_dir, args.n_eval_episodes
            )
            if not success:
                sys.exit(1)
            print("-------------------------------------")
    else:
        # Single model evaluation (original behavior)
        model_to_evaluate = find_next_model(tracker, models_dir, args.n_eval_episodes)
        if model_to_evaluate:
            success = evaluate_and_report(
                model_to_evaluate, tracker, eval_dir, args.n_eval_episodes
            )
            sys.exit(0 if success else 1)
        else:
            print("\nNo more models found to evaluate.")
            sys.exit(0)


if __name__ == "__main__":
    main()
