"""DRL training entry point for milliAmpere1 dynamic positioning."""

from __future__ import annotations

import argparse
import os
import shutil
import signal
import sys
import traceback
from datetime import datetime

import yaml


def main() -> None:
    parser = argparse.ArgumentParser(description="DRL training for milliAmpere1 DP")
    parser.add_argument("--config", default=None, help="Path to training YAML config")
    parser.add_argument(
        "--device",
        default=None,
        choices=["auto", "cpu", "cuda"],
        help="Override device (default: from config or auto)",
    )
    parser.add_argument(
        "--no-wandb", action="store_true", help="Disable Weights & Biases logging"
    )
    args = parser.parse_args()

    from milliampere_drl import __version__
    from milliampere_drl.config import TrainingConfig

    # Load config
    if args.config:
        config = TrainingConfig.from_yaml(args.config)
    else:
        config = TrainingConfig()

    # CLI overrides
    overrides = {}
    if args.device:
        overrides["device"] = args.device
    if args.no_wandb:
        overrides["wandb_project"] = None
    if overrides:
        config = config.model_copy(update=overrides)

    print(f"""
    ### ___T_ ################################################
       | n n |                   _      ____  ____  _
       |__E__|      _ __ ___    / \\    |  _ \\|  _ \\| |
    >===]__o[===<  | '_ ` _ \\  / _ \\   | | | | |_) | |
        [o__]      | | | | | |/ ___ \\  | |_| |  _ <| |___
        /7 [|      |_| |_| |_/_/   \\_\\ |____/|_| \\_\\_____| v{__version__}
      \\/7  [|_     drl-train
    ##########################################################

    Starting DRL training ...
    Config: {args.config or "defaults"}
    Device: {config.device}
    Seed:   {config.seed}
    """)

    # Seed management
    from stable_baselines3.common.utils import set_random_seed

    set_random_seed(config.seed)

    # Create run directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = f"/app/models/training_{timestamp}"
    models_dir = f"{run_dir}/models"
    logs_dir = f"{run_dir}/logs"
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)
    print(f"Training run directory: {run_dir}")

    # Snapshot configs into run directory
    with open(f"{run_dir}/training_config.yaml", "w") as f:
        yaml.dump(config.model_dump(), f, default_flow_style=False)
    if os.path.isfile(config.env_config):
        shutil.copy(config.env_config, f"{run_dir}/env_config.yaml")

    # Lazy ROS imports
    import gymnasium as gym
    import milliampere_env  # noqa: F401 -- registers MilliAmpere1-v1
    import rospy
    from stable_baselines3 import PPO
    from stable_baselines3.common.monitor import Monitor

    from milliampere_drl.callbacks import (
        DPMetricsCallback,
        HParamCallback,
        SaveModelCallback,
    )

    rospy.init_node("drl_train", anonymous=True)

    # Create environment
    env = gym.make("MilliAmpere1-v1", config_path=config.env_config)
    env.reset(seed=config.seed)
    env = Monitor(env, filename=logs_dir)

    # Create model
    model = PPO(
        config.policy,
        env,
        n_steps=config.n_steps,
        learning_rate=config.learning_rate,
        batch_size=config.batch_size,
        n_epochs=config.n_epochs,
        gamma=config.gamma,
        seed=config.seed,
        verbose=1,
        device=config.device,
        tensorboard_log=logs_dir,
    )

    # Setup callbacks
    save_callback = SaveModelCallback(
        save_interval=config.save_interval, save_path=models_dir
    )

    # Signal handler for graceful stop
    def _handle_sigint(signum, frame):
        if save_callback._interrupted:
            print("\nForced exit.")
            sys.exit(1)
        save_callback.request_stop()
        print("\nInterrupted — stopping after current step...")

    signal.signal(signal.SIGINT, _handle_sigint)

    callbacks = [save_callback, HParamCallback(), DPMetricsCallback()]

    # W&B integration (optional)
    if config.wandb_project and not args.no_wandb:
        try:
            import wandb
            from wandb.integration.sb3 import WandbCallback

            wandb.init(
                project=config.wandb_project,
                config=config.model_dump(),
                sync_tensorboard=True,
                mode="offline" if config.wandb_offline else "online",
            )
            callbacks.append(
                WandbCallback(
                    model_save_path=models_dir,
                    verbose=1,
                )
            )
        except ImportError:
            print(
                "Warning: wandb not installed. "
                "Install with: pip install milliampere-drl[wandb]"
            )

    try:
        model.learn(
            total_timesteps=config.total_timesteps,
            callback=callbacks,
            progress_bar=True,
        )
    except KeyboardInterrupt:
        print("Training interrupted by user. Saving final model...")
    except Exception as e:
        print(f"An error occurred: {e}")
        traceback.print_exc()
    finally:
        final_path = f"{models_dir}/FINAL_PPO_{timestamp}"
        model.save(final_path)
        print(f"Final model saved at {final_path}")
        env.close()

        # Finish W&B run if active
        try:
            import wandb

            if wandb.run is not None:
                wandb.finish()
        except ImportError:
            pass


if __name__ == "__main__":
    main()
