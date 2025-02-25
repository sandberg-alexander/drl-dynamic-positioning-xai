#!/usr/bin/env python3
import sys

def main():

    if len(sys.argv) < 3:
        print("Usage: python generate_compose.py NUM_ENVS NUM_EVAL_ENVS")
        sys.exit(1)

    n_envs = int(sys.argv[1])
    n_eval_envs = int(sys.argv[2])
 
    # Read the template
    with open("docker-compose-template.yml", "r") as f:
        template = f.read()

    # Locate the environment block markers
    start_marker = "\n  ##########################################################################\n  #BEGIN_ENV_BLOCK"
    end_marker = "#END_ENV_BLOCK\n  ##########################################################################"

    start_index = template.find(start_marker)
    end_index = template.find(end_marker)

    if start_index == -1 or end_index == -1:
        print("Could not find #BEGIN_ENV_BLOCK or #END_ENV_BLOCK markers in the template!")
        sys.exit(1)

    # Extract the block (excluding the markers themselvs)
    block_start = start_index + len(start_marker)
    block_end = end_index
    env_block = template[block_start:block_end]

    before_block = template[:start_index]
    after_block = template[end_index + len(end_marker):]

    cleaned_templete = before_block + after_block
    
    env_block = env_block.strip("\n")

    env_services_str = []
    env_dependencies_str = []
    env_uris_str = []
    env_hostnames_str = []

    for i in range(1, n_envs + 1):
        service_name = f"env_{i}"
        # Replace placeholders in the block
        block_instance = env_block.replace("ENV_NAME_PLACEHOLDER", service_name)
        block_instance = block_instance.replace("ENV_HOSTNAME_PLACEHOLDER", service_name + "_container")
        block_instance = block_instance.replace("ENV_URI_PLACEHOLDER", "http://" + service_name + ":11311")

        # Add the environment services
        env_services_str.append("\n" + block_instance)
        # Add a depends_on entry
        env_dependencies_str.append(f"      - {service_name}")
        # Add the ROS uri
        env_uris_str.append("http://" + service_name + ":11311")
        env_hostnames_str.append(service_name)
    
    for i in range(1, n_eval_envs + 1):
        service_name = f"eval_env_{i}"
        # Replace placeholders in the block
        block_instance = env_block.replace("ENV_NAME_PLACEHOLDER", service_name)
        #block_instance = block_instance.replace("ENV_HOSTNAME_PLACEHOLDER", service_name + "_container")
        block_instance = block_instance.replace("ENV_URI_PLACEHOLDER", "http://" + service_name + ":11311")

        # Add the environment services
        env_services_str.append("\n" + block_instance)
        # Add a depends_on entry
        env_dependencies_str.append(f"      - {service_name}")
        # Add the ROS uri
        env_uris_str.append("http://" + service_name + ":11311")
        env_hostnames_str.append(service_name)
    
    # Join them
    env_services_str_final = "".join(env_services_str)
    env_dependencies_str_final = "\n".join(env_dependencies_str)
    env_uris_str_final = ",".join(env_uris_str)
    env_hostnames_str_final = ",".join(env_hostnames_str)
    
    final_compose = cleaned_templete.replace(
        "PLACEHOLDER_DEPENDENCIES",
        env_dependencies_str_final if env_dependencies_str_final else ""
    )
    final_compose = final_compose.replace(
        "PLACEHOLDER_URIS",
        env_uris_str_final if env_uris_str_final else ""
    )
    
    final_compose = final_compose.replace(
        "PLACEHOLDER_HOSTNAMES",
        env_hostnames_str_final if env_hostnames_str_final else ""
    )

    insertion_point = final_compose.find("networks:\n  ros_net:")
    #insertion_point = block_start
    if insertion_point == -1:
        final_compose = final_compose + env_services_str_final
    else:
        final_compose = (final_compose[:insertion_point-2]
                         + env_services_str_final
                         + "\n"
                         + final_compose[insertion_point:])
    
    # Write the resulting YAML
    output_file = "docker-compose.yml"
    with open(output_file, "w") as f_out:
        f_out.write(final_compose)
    
    print(f"Generated {output_file} with {n_envs} env(s) and {n_eval_envs} eval_env(s).")


if __name__ == "__main__":
    main()