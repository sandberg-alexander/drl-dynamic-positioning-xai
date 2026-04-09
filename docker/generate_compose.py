import argparse
import sys
import socket
import re
from pathlib import Path

import yaml

# --- Configuration ---
_SCRIPT_DIR = Path(__file__).resolve().parent
TEMPLATE_FILE = str(_SCRIPT_DIR / "docker-compose-template.yml")
OUTPUT_FILE = str(_SCRIPT_DIR / "docker-compose.yml")
PARALLEL_OUTPUT_FILE = str(_SCRIPT_DIR / "docker-compose-parallel.yml")
LOCAL_SIM_SERVICE_NAME = "simulator_local"
# --- Placeholders ---
MASTER_URI_PLACEHOLDER = "ROS_MASTER_URI_PLACEHOLDER"
# Use a single placeholder for IP, as it's the same logic for drl/xai
# The script will replace this with the Laptop's IP for remote mode,
# or remove the line entirely for local mode.
ROS_IP_PLACEHOLDER = "ROS_IP_PLACEHOLDER"
DEPENDENCY_PLACEHOLDER = "# DEPENDENCY_PLACEHOLDER"
NETWORK_PLACEHOLDER = "# NETWORK_PLACEHOLDER"
NETWORK_DEFINITION_PLACEHOLDER = "# NETWORK_DEFINITION_PLACEHOLDER"
# --- ROS Network for Local Mode ---
LOCAL_ROS_NETWORK_NAME = "ros_local_net"
LOCAL_NETWORK_DEFINITION = f"""
networks:
  {LOCAL_ROS_NETWORK_NAME}:
    driver: bridge
"""
LOCAL_NETWORK_BLOCK = f"""networks:
      - {LOCAL_ROS_NETWORK_NAME}"""
# ---------------------


def get_local_ip_suggestion():
    """Suggests a local IP address usable on the LAN."""
    try:
        # Use a known public IP that's likely reachable
        target_ip, target_port = "1.1.1.1", 80
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect((target_ip, target_port))
        ip = s.getsockname()[0]
        s.close()
        # Basic check if it's likely a private/LAN IP
        if (
            ip.startswith(("192.168.", "10.", "172.")) or ":" in ip
        ):  # Include IPv6 check loosely
            return ip
        else:
            print(
                f"Warning: Auto-detected IP {ip} might not be a typical LAN IP.",
                file=sys.stderr,
            )
            return ip
    except Exception as e:
        print(f"Warning: Could not auto-detect local IP: {e}", file=sys.stderr)
        return "<your_laptop_ip>"


def generate_laptop_compose(mode, remote_ip=None, local_ip=None):
    """
    Generates the docker-compose.yml file for the laptop.

    Args:
        mode (str): 'local' or 'remote'.
        remote_ip (str, optional): IP of the remote PC (required for remote mode).
        local_ip (str, optional): IP of the laptop (for remote mode, can be auto-detected).
    """
    print("--- Generating Laptop Configuration ---")
    print(f"Mode: {mode}")

    # --- Determine IPs and Master URI ---
    laptop_ip_to_set = None  # IP to set for ROS_IP in remote mode

    if mode == "local":
        if remote_ip:
            print("Warning: --remote-ip is ignored for local mode.", file=sys.stderr)
        # Use service names for local communication on the Docker bridge network
        ros_master_uri = f"http://{LOCAL_SIM_SERVICE_NAME}:11311"
        include_local_simulator = True
        use_docker_network = True
        dependency_target = f"- {LOCAL_SIM_SERVICE_NAME}"
        print("Local mode selected.")
        print(
            "ROS_IP will be unset (containers use Docker bridge network with ROS_HOSTNAME)."
        )
        print(f"ROS Master URI: {ros_master_uri}")

    elif mode == "remote":
        if not remote_ip:
            print("Error: --remote-ip is required for remote mode.", file=sys.stderr)
            sys.exit(1)
        if not local_ip:
            local_ip = get_local_ip_suggestion()
            print(
                f"Auto-detected Laptop IP: {local_ip} (Use --local-ip to override if incorrect)"
            )
        if local_ip == "<your_laptop_ip>":
            print(
                "Error: Could not auto-detect local IP. Please provide it using --local-ip.",
                file=sys.stderr,
            )
            sys.exit(1)

        ros_master_uri = f"http://{remote_ip}:11311"
        laptop_ip_to_set = local_ip  # Laptop's IP needed for ROS_IP
        include_local_simulator = False
        use_docker_network = (
            False  # Communication via host network (network_mode: host)
        )
        dependency_target = ""
        print("Remote mode selected.")
        print(f"Remote PC (Master) IP: {remote_ip}")
        print(f"Laptop IP (for ROS_IP): {laptop_ip_to_set}")
        print(f"ROS Master URI: {ros_master_uri}")
    else:
        print(
            f"Error: Invalid mode '{mode}'. Use 'local' or 'remote'.", file=sys.stderr
        )
        sys.exit(1)

    # --- Read the template file ---
    try:
        with open(TEMPLATE_FILE, "r") as f:
            template_content = f.read()
    except FileNotFoundError:
        print(f"Error: Template file '{TEMPLATE_FILE}' not found.", file=sys.stderr)
        sys.exit(1)
    except IOError as e:
        print(f"Error reading template file '{TEMPLATE_FILE}': {e}", file=sys.stderr)
        sys.exit(1)

    # --- Perform Replacements on the template string ---
    modified_content = template_content.replace(MASTER_URI_PLACEHOLDER, ros_master_uri)
    modified_content = modified_content.replace(
        DEPENDENCY_PLACEHOLDER, dependency_target
    )

    # Replace or remove ROS_IP line based on mode
    if laptop_ip_to_set:  # Remote mode: Replace placeholder with Laptop's IP
        modified_content = modified_content.replace(
            ROS_IP_PLACEHOLDER, laptop_ip_to_set
        )
        print(f"Setting ROS_IP to: {laptop_ip_to_set}")
    else:  # Local mode: Remove the entire line containing the placeholder
        # Regex explanation:
        # ^\s*      : Match start of line and any leading whitespace
        # - ROS_IP= : Match the literal string "- ROS_IP="
        # .*        : Match the rest of the placeholder and any trailing characters
        # $\n?      : Match the end of the line and an optional newline
        pattern = rf"^\s*- ROS_IP={re.escape(ROS_IP_PLACEHOLDER)}.*$\n?"
        modified_content = re.sub(pattern, "", modified_content, flags=re.MULTILINE)
        print("Removing ROS_IP lines for local mode.")

    # Add/Remove Network Blocks
    if use_docker_network:
        modified_content = modified_content.replace(
            NETWORK_PLACEHOLDER, LOCAL_NETWORK_BLOCK
        )
        modified_content = modified_content.replace(
            NETWORK_DEFINITION_PLACEHOLDER, LOCAL_NETWORK_DEFINITION
        )
    else:
        modified_content = re.sub(
            rf"^\s*{NETWORK_PLACEHOLDER}.*$\n?",
            "",
            modified_content,
            flags=re.MULTILINE,
        )
        modified_content = re.sub(
            rf"^\s*{NETWORK_DEFINITION_PLACEHOLDER}.*$\n?",
            "",
            modified_content,
            flags=re.MULTILINE,
        )

    # --- Load YAML to manipulate structure ---
    try:
        data = yaml.safe_load(modified_content)
        if not isinstance(data, dict) or "services" not in data:
            print("Error: Invalid YAML structure after replacements.", file=sys.stderr)
            sys.exit(1)
    except yaml.YAMLError as e:
        print(f"Error parsing YAML from modified content: {e}", file=sys.stderr)
        sys.exit(1)

    # --- Post-Load Modifications (network_mode, remove services) ---
    services_dict = data.get("services", {})

    # Remove network_mode: host only if in local mode
    if mode == "local":
        print(
            "Local mode detected. Removing 'network_mode: host' from drl and xai services."
        )
        for service_name in ["drl", "xai"]:
            if (
                service_name in services_dict
                and "network_mode" in services_dict[service_name]
            ):
                del services_dict[service_name]["network_mode"]
                print(f" - Removed network_mode from {service_name}")
        # Local simulator should never use host network in this setup
        if (
            LOCAL_SIM_SERVICE_NAME in services_dict
            and "network_mode" in services_dict[LOCAL_SIM_SERVICE_NAME]
        ):
            print(
                f"Warning: Found unexpected network_mode in {LOCAL_SIM_SERVICE_NAME} for local mode. Removing."
            )
            del services_dict[LOCAL_SIM_SERVICE_NAME]["network_mode"]

    # Remove networks definitions from services if in remote mode (host mode makes them irrelevant)
    elif mode == "remote":
        print(
            "Remote mode detected. Removing 'networks:' definition from services if present."
        )
        for service_name in ["drl", "xai"]:
            if (
                service_name in services_dict
                and "networks" in services_dict[service_name]
            ):
                del services_dict[service_name]["networks"]
                print(f" - Removed networks definition from {service_name}")
        # Remove the top-level networks definition if it exists (should have been removed by placeholder logic)
        if "networks" in data:
            print("Removing residual top-level 'networks' definition.")
            del data["networks"]

    # Remove local simulator service if in remote mode
    if not include_local_simulator:
        if LOCAL_SIM_SERVICE_NAME in services_dict:
            del services_dict[LOCAL_SIM_SERVICE_NAME]
            print(f"Removed '{LOCAL_SIM_SERVICE_NAME}' service for remote mode.")
        else:
            print(
                f"Warning: Local service '{LOCAL_SIM_SERVICE_NAME}' not found in parsed data.",
                file=sys.stderr,
            )

    # --- Write the final Laptop docker-compose.yml file ---
    try:
        with open(OUTPUT_FILE, "w") as f:
            yaml.dump(data, f, sort_keys=False, default_flow_style=False, width=1000)
        print(f"Successfully generated '{OUTPUT_FILE}'.")
    except Exception as e:  # Broader exception for write/dump
        print(f"Error writing output file '{OUTPUT_FILE}': {e}", file=sys.stderr)
        sys.exit(1)


def generate_parallel_compose(n_envs):
    """Generate a docker-compose file for parallel training with N simulator instances.

    Each simulator runs in its own container with an independent roscore.
    The drl container connects to all simulators via a shared bridge network.
    """
    print("--- Generating Parallel Training Configuration ---")
    print(f"Number of environments: {n_envs}")

    network_name = "ros_parallel_net"

    # --- Simulator service template ---
    # Use ';' (not '&&') between source and backgrounded commands.
    # With '&&', bash groups 'source ... && roscore &' as one background job,
    # leaving 'roslaunch' in a subshell without the sourced environment.
    # With ';', sources run in the main shell before anything is backgrounded.
    sim_command = (
        "bash -c '"
        "source /opt/ros/noetic/setup.bash; "
        "source /workspace/devel/setup.bash; "
        "roscore & "
        "sleep 3; "
        "roslaunch /workspace/src/simulator.launch & "
        "tail -f /dev/null"
        "'"
    )

    # Health check uses 'bash -c' because Docker runs CMD-SHELL with /bin/sh,
    # which doesn't support 'source' (a bashism).
    sim_healthcheck = {
        "test": [
            "CMD-SHELL",
            "bash -c 'source /opt/ros/noetic/setup.bash && rostopic list > /dev/null 2>&1'",
        ],
        "interval": "5s",
        "timeout": "5s",
        "retries": 10,
        "start_period": "30s",
    }

    sim_env_common = {
        "LD_LIBRARY_PATH": "/workspace/src/ma27",
        "ROS_LANG_DISABLE": "genlisp;gennodejs;geneus",
        "ROS_LOG_DIR": "/logging/node_logs",
        "CATKIN_ENABLE_TESTING": "0",
    }

    # --- Build services dict ---
    services = {}

    # Base image (build-only, required by drl/xai)
    services["base"] = {
        "build": {"context": "..", "dockerfile": "docker/Dockerfile.base"},
        "image": "my-ros-base:latest",
    }

    # N simulator services
    sim_depends = {}
    for i in range(n_envs):
        name = f"sim_{i}"
        sim_depends[name] = {"condition": "service_healthy"}
        services[name] = {
            "image": "milliampere-sim-built:latest",
            "container_name": name,
            "hostname": name,
            "environment": {
                **sim_env_common,
                "ROS_MASTER_URI": f"http://{name}:11311",
                "ROS_HOSTNAME": name,
            },
            "command": sim_command,
            "healthcheck": sim_healthcheck,
            "networks": [network_name],
            "init": True,
        }

    # DRL training service
    drl_depends = {"base": {"condition": "service_started"}}
    drl_depends.update(sim_depends)
    services["drl"] = {
        "build": {"context": "..", "dockerfile": "docker/Dockerfile.drl"},
        "container_name": "drl",
        "hostname": "drl_container",
        "depends_on": drl_depends,
        "environment": {
            "N_ENVS": str(n_envs),
            "ROS_MASTER_URI": "http://sim_0:11311",
        },
        "volumes": [
            "../data/models:/app/models:rw",
            "../data/xai_samples:/app/xai_samples:rw",
            "../data/runs:/app/runs:rw",
            "../drl/src/milliampere_drl:"
            "/usr/local/lib/python3.8/dist-packages/milliampere_drl:ro",
            "../configs:/app/configs:ro",
            "../milliampere_dp/src/milliampere_dp:"
            "/usr/local/lib/python3.8/dist-packages/milliampere_dp:ro",
            "../ros_packages/milliampere_env/src/milliampere_env:"
            "/root/catkin_ws/src/milliampere_env/src/milliampere_env:ro",
        ],
        "command": (
            "bash -c '"
            "source /root/catkin_ws/devel/setup.bash && "
            "source /opt/ros/noetic/setup.bash && "
            "tail -f /dev/null"
            "'"
        ),
        "networks": [network_name],
        "init": True,
    }

    # XAI service (optional, connects to sim_0 by default)
    xai_depends = {"base": {"condition": "service_started"}}
    xai_depends["sim_0"] = {"condition": "service_healthy"}
    services["xai"] = {
        "build": {"context": "..", "dockerfile": "docker/Dockerfile.xai"},
        "container_name": "xai",
        "hostname": "xai_container",
        "depends_on": xai_depends,
        "environment": {
            "ROS_MASTER_URI": "http://sim_0:11311",
        },
        "volumes": [
            "../data/models:/app/models:rw",
            "../data/xai_samples:/app/xai_samples:rw",
            "../data/runs:/app/runs:rw",
            "../xai/src/milliampere_xai:"
            "/usr/local/lib/python3.8/dist-packages/milliampere_xai:ro",
            "../configs:/app/configs:ro",
            "../milliampere_dp/src/milliampere_dp:"
            "/usr/local/lib/python3.8/dist-packages/milliampere_dp:ro",
            "../ros_packages/milliampere_env/src/milliampere_env:"
            "/root/catkin_ws/src/milliampere_env/src/milliampere_env:ro",
        ],
        "ports": ["8080:8080"],
        "command": (
            "bash -c '"
            "source /root/catkin_ws/devel/setup.bash && "
            "source /opt/ros/noetic/setup.bash && "
            "tail -f /dev/null"
            "'"
        ),
        "networks": [network_name],
        "init": True,
    }

    # --- Assemble full compose structure ---
    compose = {
        "services": services,
        "networks": {network_name: {"driver": "bridge"}},
    }

    # --- Write output ---
    try:
        with open(PARALLEL_OUTPUT_FILE, "w") as f:
            f.write(
                f"# Auto-generated parallel training config ({n_envs} simulators).\n"
                f"# Generated by: python3 generate_compose.py parallel --n-envs {n_envs}\n"
                f"# Do not edit manually — re-run the generator instead.\n\n"
            )
            yaml.dump(compose, f, sort_keys=False, default_flow_style=False, width=1000)
        print(f"Successfully generated '{PARALLEL_OUTPUT_FILE}'.")
    except Exception as e:
        print(f"Error writing output file: {e}", file=sys.stderr)
        sys.exit(1)

    # Print summary
    print("\nServices:")
    print(f"  Simulators: {', '.join(f'sim_{i}' for i in range(n_envs))}")
    print(f"  Training:   drl (N_ENVS={n_envs})")
    print("  XAI:        xai (connected to sim_0)")
    print(f"  Network:    {network_name}")
    print("\nUsage:")
    print("  just build-parallel         # Build all images")
    print("  just up-parallel            # Start infrastructure")
    print("  just down-parallel          # Stop infrastructure")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate docker-compose files for local, remote, or parallel ROS configurations."
    )
    parser.add_argument(
        "mode",
        choices=["local", "remote", "parallel"],
        help=(
            "Specify the mode: 'local' (run simulator locally), "
            "'remote' (connect to remote ROS master), "
            "'parallel' (N simulator instances for parallel training)."
        ),
    )
    parser.add_argument(
        "--remote-ip", help="IP address of the remote PC (required for 'remote' mode)."
    )
    parser.add_argument(
        "--local-ip",
        help="IP address of this Laptop (required for 'remote' mode, attempts auto-detection if not provided).",
    )
    parser.add_argument(
        "--n-envs",
        type=int,
        default=4,
        help="Number of parallel simulator instances (default: 4, for 'parallel' mode).",
    )

    args = parser.parse_args()

    if args.mode == "remote" and not args.remote_ip:
        parser.error("--remote-ip is required when mode is 'remote'")

    if args.mode == "parallel":
        if args.n_envs < 1:
            parser.error("--n-envs must be at least 1")
        generate_parallel_compose(args.n_envs)
    else:
        generate_laptop_compose(args.mode, args.remote_ip, args.local_ip)
