"""Headless DRL deployment entry point for milliAmpere1 DP."""

from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="DRL deployment for milliAmpere1 DP")
    parser.add_argument("--config", default=None, help="Path to deploy YAML config")
    parser.add_argument("--model", default=None, help="Override model path")
    parser.add_argument("--env-config", default=None, help="Override env config path")
    parser.add_argument(
        "--device",
        default=None,
        choices=["cpu", "cuda"],
        help="Device for model inference (default: cpu)",
    )
    args = parser.parse_args()

    from milliampere_drl import __version__
    from milliampere_drl.config import DeployConfig
    from milliampere_drl.deployer import DRLDeployer

    # Load config
    if args.config:
        config = DeployConfig.from_yaml(args.config)
    else:
        config = DeployConfig()

    # CLI overrides
    overrides = {}
    if args.model:
        overrides["model_path"] = args.model
    if args.env_config:
        overrides["env_config"] = args.env_config
    if overrides:
        config = config.model_copy(update=overrides)

    print(f"""
    ### ___T_ ################################################
       | n n |                   _      ____  ____  _
       |__E__|      _ __ ___    / \\    |  _ \\|  _ \\| |
    >===]__o[===<  | '_ ` _ \\  / _ \\   | | | | |_) | |
        [o__]      | | | | | |/ ___ \\  | |_| |  _ <| |___
        /7 [|      |_| |_| |_/_/   \\_\\ |____/|_| \\_\\_____| v{__version__}
      \\/7  [|_     drl-deploy (headless)
    ##########################################################

    Starting DRL deployment in DP mode ...
    Model:  {config.model_path}
    Config: {config.env_config}
    """)

    # Lazy ROS import
    import rospy

    rospy.init_node("drl_deployer", anonymous=True)

    node = DRLDeployer(
        model_path=config.model_path,
        config_path=config.env_config,
        device=args.device or "cpu",
    )
    node.spin()


if __name__ == "__main__":
    main()
