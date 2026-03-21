"""Model evaluation entry point for milliAmpere1 DRL-DP."""

from __future__ import annotations

import argparse
import glob
import os
import re
import sys
import time
import traceback


def evaluate_one_model(
    model_path: str,
    tracker,
    eval_dir: str,
    env_config: str,
    n_eval_episodes: int = 10,
    seed: int = 42,
) -> bool:
    """Evaluate a single model, resuming if necessary."""
    import gymnasium as gym
    import milliampere_env  # noqa: F401
    import rospy
    from stable_baselines3 import PPO
    from stable_baselines3.common.monitor import Monitor

    from milliampere_drl.tracker import append_monitor_data

    model_name = os.path.basename(model_path)
    model_monitor_dir = os.path.join(eval_dir, f"monitor_{model_name}")
    main_monitor_filepath = os.path.join(model_monitor_dir, "monitor.csv")

    if tracker.is_model_completed(model_path):
        print(f"Model {model_name} already completed. Skipping.")
        return True

    tracker.start_model_evaluation(model_path)
    start_episode = tracker.get_completed_episodes()

    if start_episode >= n_eval_episodes:
        print(f"Model {model_name} already has {start_episode} episodes. Done.")
        tracker.complete_model_evaluation()
        return True

    env = None
    timestamp = int(time.time() * 1000)
    temp_dir = os.path.join(model_monitor_dir, f"temp_{timestamp}")
    temp_monitor_filepath = os.path.join(temp_dir, "monitor.csv")
    info_keywords_to_log = ("target_pose",)
    should_append_data = False
    evaluation_successful = False
    successful_episodes_this_run = 0

    try:
        print(f"Loading model {model_name}...")
        model = PPO.load(model_path, device="cpu")

        print(f"Creating environment for model {model_name}...")
        if not rospy.core.is_initialized():
            rospy.init_node("drl_eval", anonymous=True)
        env = gym.make("MilliAmpere1-v1", config_path=env_config)
        os.makedirs(temp_dir, exist_ok=True)

        print(f"Monitor temp: {temp_monitor_filepath}")
        env = Monitor(
            env, filename=temp_monitor_filepath, info_keywords=info_keywords_to_log
        )

        print(f"Evaluating episodes {start_episode + 1} to {n_eval_episodes}")

        for i in range(start_episode, n_eval_episodes):
            current_episode_num = i + 1
            print(f"\n--- Episode {current_episode_num}/{n_eval_episodes} ---")

            episode_seed = seed + i
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
                f"--- Episode {current_episode_num} done "
                f"(Reward: {episode_reward:.2f}, Length: {episode_length}) ---"
            )
            tracker.complete_episode()
            successful_episodes_this_run += 1
            should_append_data = True

        evaluation_successful = True

    except KeyboardInterrupt:
        print(f"\nInterrupted during {model_name}. Saving partial results.")
        if successful_episodes_this_run > 0:
            should_append_data = True
        return False

    except Exception as e:
        print(f"\nERROR during {model_name}: {e}")
        traceback.print_exc()
        if successful_episodes_this_run > 0:
            should_append_data = True
        return False

    finally:
        if env:
            try:
                env.close()
                time.sleep(0.5)
            except Exception as e:
                print(f"Error closing environment: {e}")
        if should_append_data:
            append_monitor_data(temp_monitor_filepath, main_monitor_filepath)
        else:
            if os.path.exists(temp_monitor_filepath):
                try:
                    os.remove(temp_monitor_filepath)
                    os.rmdir(os.path.dirname(temp_monitor_filepath))
                except OSError as e:
                    print(f"Error cleaning up unused temp: {e}")

        if evaluation_successful:
            final_count = tracker.get_completed_episodes()
            if final_count >= n_eval_episodes:
                print(f"All {n_eval_episodes} episodes completed for {model_name}.")
                tracker.complete_model_evaluation()
        else:
            final_count = tracker.get_completed_episodes()
            print(
                f"Stopped for {model_name} after "
                f"{final_count}/{n_eval_episodes} episodes."
            )

    return evaluation_successful and (
        tracker.get_completed_episodes() >= n_eval_episodes
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate models for a training run")
    parser.add_argument(
        "--dir",
        "--run_dir",
        dest="run_dir",
        type=str,
        required=True,
        help="Training run directory",
    )
    parser.add_argument(
        "--episodes",
        "--n_eval_episodes",
        dest="n_eval_episodes",
        type=int,
        default=10,
        help="Number of episodes per model",
    )
    parser.add_argument(
        "--env-config",
        default=None,
        help="Override env config (default: from run snapshot or v5_equivalent)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override seed (default: from run snapshot or 42)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Evaluate all unevaluated models in a loop",
    )
    args = parser.parse_args()

    from milliampere_drl import __version__
    from milliampere_drl.tracker import EvaluationTracker

    # Try to read env config from training snapshot
    env_config = args.env_config
    seed = args.seed if args.seed is not None else 42
    snapshot_path = os.path.join(args.run_dir, "env_config.yaml")
    if env_config is None:
        if os.path.isfile(snapshot_path):
            env_config = snapshot_path
            print(f"Using snapshotted env config: {snapshot_path}")
        else:
            env_config = "/app/configs/env/legacy/v5_equivalent.yaml"

    print(f"""
    ### ___T_ ################################################
       | n n |                   _      ____  ____  _
       |__E__|      _ __ ___    / \\    |  _ \\|  _ \\| |
    >===]__o[===<  | '_ ` _ \\  / _ \\   | | | | |_) | |
        [o__]      | | | | | |/ ___ \\  | |_| |  _ <| |___
        /7 [|      |_| |_| |_/_/   \\_\\ |____/|_| \\_\\_____| v{__version__}
      \\/7  [|_     drl-evaluate
    ##########################################################

    Starting evaluation{" (--all mode)" if args.all else ""}...
    """)

    models_dir = os.path.join(args.run_dir, "models")
    eval_dir = os.path.join(args.run_dir, "evaluation")
    os.makedirs(eval_dir, exist_ok=True)

    print(f"Models: {models_dir}")
    print(f"Eval dir: {eval_dir}")
    print(f"Episodes per model: {args.n_eval_episodes}")
    print(f"Env config: {env_config}")

    tracker = EvaluationTracker(eval_dir)

    def find_next_model(tracker, models_dir, n_eval_episodes):
        model_to_evaluate = None

        current_model_path = tracker.get_current_model()
        if current_model_path:
            if os.path.exists(current_model_path):
                if tracker.get_completed_episodes() >= n_eval_episodes:
                    tracker.complete_model_evaluation()
                else:
                    print(f"Resuming: {os.path.basename(current_model_path)}")
                    model_to_evaluate = current_model_path
            else:
                print(f"Warning: '{current_model_path}' not found. Resetting.")
                tracker.progress["current_model"] = None
                tracker.progress["completed_episodes"] = 0
                tracker.save_progress()

        if not model_to_evaluate:
            model_files = glob.glob(os.path.join(models_dir, "PPO_*.zip"))
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
                    print(f"Next: {os.path.basename(model_to_evaluate)}")
                    break

        return model_to_evaluate

    def evaluate_and_report(model_to_evaluate, tracker, eval_dir, n_eval_episodes):
        try:
            success = evaluate_one_model(
                model_to_evaluate,
                tracker,
                eval_dir,
                env_config,
                n_eval_episodes,
                seed,
            )
            if success:
                print(f"\nCompleted: {os.path.basename(model_to_evaluate)}")
            else:
                print(f"\nIncomplete: {os.path.basename(model_to_evaluate)}")
            return success
        except KeyboardInterrupt:
            print("\nExiting.")
            return False
        except Exception as e:
            print(f"\nFATAL: {e}")
            traceback.print_exc()
            return False

    if args.all:
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
        model_to_evaluate = find_next_model(tracker, models_dir, args.n_eval_episodes)
        if model_to_evaluate:
            success = evaluate_and_report(
                model_to_evaluate, tracker, eval_dir, args.n_eval_episodes
            )
            sys.exit(0 if success else 1)
        else:
            print("\nNo more models to evaluate.")
            sys.exit(0)


if __name__ == "__main__":
    main()
