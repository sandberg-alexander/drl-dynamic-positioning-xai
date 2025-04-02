import argparse
import yaml
import sys
import os

# --- Configuration ---
TEMPLATE_FILE = 'docker-compose-template.yml'
OUTPUT_FILE = 'docker-compose.yml'
PLACEHOLDER = 'SERVICE_PLACEHOLDER'
SERVICE_SIM = 'simulator'
SERVICE_REAL = 'real_vessel'
# ---------------------

def generate_compose_file(mode):
    """
    Generates the docker-compose.yml file based on the selected mode.

    Args:
        mode (str): 'sim' or 'real'.
    """
    if mode == 'sim':
        selected_service = SERVICE_SIM
        service_to_remove = SERVICE_REAL
        print(f"Selected mode: '{mode}'. Including service '{selected_service}' and removing '{service_to_remove}'.")
    elif mode == 'real':
        selected_service = SERVICE_REAL
        service_to_remove = SERVICE_SIM
        print(f"Selected mode: '{mode}'. Including service '{selected_service}' and removing '{service_to_remove}'.")
    else:
        # This should ideally be caught by argparse, but added as a safeguard
        print(f"Error: Invalid mode '{mode}'. Use 'sim' or 'real'.", file=sys.stderr)
        sys.exit(1)

    # --- Read the template file ---
    try:
        with open(TEMPLATE_FILE, 'r') as f:
            # Read the raw content first for placeholder replacement
            template_content = f.read()
    except FileNotFoundError:
        print(f"Error: Template file '{TEMPLATE_FILE}' not found in the current directory '{os.getcwd()}'.", file=sys.stderr)
        sys.exit(1)
    except IOError as e:
        print(f"Error reading template file '{TEMPLATE_FILE}': {e}", file=sys.stderr)
        sys.exit(1)

    # --- Replace the placeholder ---
    # Doing string replacement before parsing is often easier for simple cases
    modified_content = template_content.replace(PLACEHOLDER, selected_service)

    # --- Load the modified YAML content ---
    try:
        # Use safe_load to avoid potential security risks with arbitrary code execution
        data = yaml.safe_load(modified_content)
        if not isinstance(data, dict) or 'services' not in data:
             print(f"Error: Invalid YAML structure in '{TEMPLATE_FILE}' after modification. Missing 'services' key?", file=sys.stderr)
             sys.exit(1)

    except yaml.YAMLError as e:
        print(f"Error parsing YAML from modified template content: {e}", file=sys.stderr)
        sys.exit(1)


    # --- Remove the unwanted service ---
    if service_to_remove in data.get('services', {}):
        del data['services'][service_to_remove]
        print(f"Successfully removed '{service_to_remove}' service definition.")
    else:
         # This might happen if the template was already modified
        print(f"Warning: Service '{service_to_remove}' not found in the template data. Skipping removal.", file=sys.stderr)


    # --- Write the final docker-compose.yml file ---
    try:
        with open(OUTPUT_FILE, 'w') as f:
            # dump preserves order better if using standard dicts since Python 3.7+
            # sort_keys=False is important to maintain the original order as much as possible
            # default_flow_style=False ensures block style (indented) output
            yaml.dump(data, f, sort_keys=False, default_flow_style=False)
        print(f"Successfully generated '{OUTPUT_FILE}' for '{selected_service}' mode.")
    except IOError as e:
        print(f"Error writing output file '{OUTPUT_FILE}': {e}", file=sys.stderr)
        sys.exit(1)
    except yaml.YAMLError as e:
         print(f"Error dumping YAML to '{OUTPUT_FILE}': {e}", file=sys.stderr)
         sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=f"Generate {OUTPUT_FILE} from {TEMPLATE_FILE}, selecting either '{SERVICE_SIM}' or '{SERVICE_REAL}' service."
    )
    parser.add_argument(
        'mode',
        choices=['sim', 'real'],
        help=f"Specify the mode: 'sim' to include '{SERVICE_SIM}', 'real' to include '{SERVICE_REAL}'."
    )

    args = parser.parse_args()
    generate_compose_file(args.mode)