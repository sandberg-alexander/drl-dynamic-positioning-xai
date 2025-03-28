#!/usr/bin/env python3
import rospy
import os
import time
import sys
from custom_msgs.msg import ActuatorSetpoints
from geometry_msgs.msg import PoseStamped

def pose_callback(data):
    """Callback for pose data."""
    print(f"Received pose - Position: [{data.pose.position.x:.2f}, {data.pose.position.y:.2f}, {data.pose.position.z:.2f}]")

def test_publish():
    """Test publishing to actuator topics."""
    # Initialize the ROS node
    rospy.init_node('ros_debug_node', anonymous=True)
    
    # Create publishers for actuator references
    pub1 = rospy.Publisher('/actuator_ref_1', ActuatorSetpoints, queue_size=1)
    pub2 = rospy.Publisher('/actuator_ref_2', ActuatorSetpoints, queue_size=1)
    pub3 = rospy.Publisher('/actuator_ref_3', ActuatorSetpoints, queue_size=1)
    pub4 = rospy.Publisher('/actuator_ref_4', ActuatorSetpoints, queue_size=1)
    
    # Create a subscriber to check if we receive messages
    pose_sub = rospy.Subscriber('/navigation/pose', PoseStamped, pose_callback)
    
    # Wait for publishers to connect
    print("Waiting for publishers to connect...")
    time.sleep(2)
    
    # Create a test message
    msg = ActuatorSetpoints()
    msg.throttle_reference = 100.0  # Set to a non-zero value for visibility
    msg.angle_reference = 45.0
    
    # Print environment info
    print("ROS Environment:")
    print(f"ROS_MASTER_URI: {os.environ.get('ROS_MASTER_URI', 'Not set')}")
    print(f"ROS_HOSTNAME: {os.environ.get('ROS_HOSTNAME', 'Not set')}")
    print(f"ROS_IP: {os.environ.get('ROS_IP', 'Not set')}")
    print(f"Node name: {rospy.get_name()}")
    
    # Publish test messages
    rate = rospy.Rate(1)  # 1 Hz
    print("\nStarting to publish test messages...")
    
    for i in range(10):
        # Update message to show changing values
        msg.throttle_reference = 100.0 + i * 10
        msg.angle_reference = 45.0 + i * 5
        
        pub1.publish(msg)
        pub2.publish(msg)
        pub3.publish(msg)
        pub4.publish(msg)
        
        print(f"Published test message {i+1}/10 - Throttle: {msg.throttle_reference}, Angle: {msg.angle_reference}")
        rate.sleep()
    
    print("Test publishing complete")
    print("Checking if we're receiving pose data...")
    time.sleep(3)  # Wait a bit to see if we get pose data
    
    if not pose_sub.get_num_connections():
        print("WARNING: No publishers for /navigation/pose detected!")
    
    print("\nRun these commands in another terminal to check communication:")
    print("rostopic echo /actuator_ref_1")
    print("rostopic hz /actuator_ref_1")
    
    # Keep node running for a while to receive messages
    print("\nKeeping node alive for 30 seconds to receive messages...")
    time.sleep(30)

def check_rostopic():
    """Try to run rostopic commands to list topics."""
    import subprocess
    try:
        print("\nRunning rostopic list...")
        result = subprocess.run(["rostopic", "list"], capture_output=True, text=True)
        if result.returncode == 0:
            print("Available topics:")
            print(result.stdout)
        else:
            print("Error running rostopic list:")
            print(result.stderr)
            
        # Check specifically for our critical topics
        for topic in ['/actuator_ref_1', '/navigation/pose']:
            print(f"\nChecking info for {topic}...")
            result = subprocess.run(["rostopic", "info", topic], capture_output=True, text=True)
            if result.returncode == 0:
                print(result.stdout)
            else:
                print(f"Error getting info for {topic}:")
                print(result.stderr)
    except Exception as e:
        print(f"Error running subprocess commands: {e}")

if __name__ == '__main__':
    try:
        check_rostopic()
        test_publish()
    except rospy.ROSInterruptException:
        pass
    except KeyboardInterrupt:
        print("Script interrupted by user")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()