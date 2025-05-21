cmake_minimum_required(VERSION 3.0.2)
project(custom_ros_msgs)

find_package(catkin REQUIRED COMPONENTS
  std_msgs
  message_generation
)

add_message_files(
  FILES
  Action.msg
  Observation.msg
  ObservationActuatorRefPair.msg
  Mode.msg
)

generate_messages(
  DEPENDENCIES
  std_msgs
)

catkin_package(
  CATKIN_DEPENDS message_runtime std_msgs
)
