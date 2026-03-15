import argparse
import yaml
import sys
import socket
import re
from pathlib import Path

# --- Configuration ---
_SCRIPT_DIR = Path(__file__).resolve().parent
TEMPLATE_FILE = str(_SCRIPT_DIR / "docker-compose-template.yml")
OUTPUT_FILE = str(_SCRIPT_DIR / "docker-compose.yml")
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
        print("ROS_IP will be unset (using Docker bridge network DNS).")
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=f"Generate {OUTPUT_FILE} from {TEMPLATE_FILE} for local or remote ROS connection."
    )
    parser.add_argument(
        "mode",
        choices=["local", "remote"],  # Simplified modes
        help="Specify the mode: 'local' (run simulator locally), 'remote' (connect to remote ROS master).",
    )
    parser.add_argument(
        "--remote-ip", help="IP address of the remote PC (required for 'remote' mode)."
    )
    parser.add_argument(
        "--local-ip",
        help="IP address of this Laptop (required for 'remote' mode, attempts auto-detection if not provided).",
    )

    args = parser.parse_args()

    # Validate remote_ip requirement
    if args.mode == "remote" and not args.remote_ip:
        parser.error("--remote-ip is required when mode is 'remote'")

    generate_laptop_compose(args.mode, args.remote_ip, args.local_ip)
