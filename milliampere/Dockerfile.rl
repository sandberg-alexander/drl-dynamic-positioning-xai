# Use the official ROS Noetic image
FROM ros:noetic

# Install necessary packages
RUN apt-get update && apt-get install -y \
    bash \
    python3-pip \
    python3-venv \
    python3-rospy \
    python3-rosdep \
    python3-catkin-pkg \
    ros-noetic-std-msgs \
    ros-noetic-message-generation \
    ros-noetic-message-runtime \
    # Any other dependencies you need
    && rm -rf /var/lib/apt/lists/*

# Set up the ROS workspace
WORKDIR /root/catkin_ws/src

# Copy the custom_msgs package into the workspace
COPY milliampere_code/workspace/src/custom_msgs ./custom_msgs
COPY milliampere_code/workspace/src/sim_integrator ./sim_integrator
COPY milliampere_code/workspace/src/sim_milliampere ./sim_milliampere

# Build the ROS workspace
WORKDIR /root/catkin_ws
RUN /bin/bash -c "source /opt/ros/noetic/setup.bash && catkin_make"

# Source the workspace environment
RUN echo "source /root/catkin_ws/devel/setup.bash" >> ~/.bashrc

# Create a Python virtual environment with access to system site packages
WORKDIR /app/rl_code

# Copy and install Python requirements
COPY ./rl_code/requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy RL code into the container
COPY ./rl_code/ .

# Set environment variables
ENV ROS_MASTER_URI=http://ros_master:11311
ENV ROS_HOSTNAME=rl_agent

# Expose ROS port
EXPOSE 11311

RUN apt-get update && apt-get install -y x11-apps
RUN apt-get update && apt-get install -y fontconfig

# Start the training script when the container runs
#CMD ["/bin/bash", "-c", "source /root/catkin_ws/devel/setup.bash && cd /app/rl_code && exec bash","tail", "-f", "/dev/null"]
CMD ["tail", "-f", "/dev/null"]

