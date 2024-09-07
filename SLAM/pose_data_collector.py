#!/usr/bin/env python3

import rospy
import tf2_ros
import geometry_msgs.msg
import numpy as np
from tf.transformations import quaternion_matrix

class HectorSLAMDataCollector:
    def __init__(self):
        rospy.init_node('hector_slam_data_collector', anonymous=True)
        
        # Create a tf2 buffer and listener
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        
        # The frame names you're interested in
        self.base_frame = rospy.get_param('~base_frame', 'base_link')
        self.map_frame = rospy.get_param('~map_frame', 'map')
        
        # File to save the data
        self.output_file = rospy.get_param('~output_file', '/home/sashitsharma/Desktop/thesis_github/Thesis/SLAM/Turtlebot3/gmapping/gmapping_pose.txt')
        
        # Data collection rate
        self.rate = rospy.Rate(2)  # 10 Hz, adjust as needed
        
        # Open the file for writing
        self.file = open(self.output_file, 'w')

    def collect_data(self):
        while not rospy.is_shutdown():
            try:
                # Look up the transform
                transform = self.tf_buffer.lookup_transform(self.map_frame, self.base_frame, rospy.Time())
                
                # Extract position
                x = transform.transform.translation.x
                y = transform.transform.translation.y
                z = transform.transform.translation.z
                
                # Extract orientation
                qx = transform.transform.rotation.x
                qy = transform.transform.rotation.y
                qz = transform.transform.rotation.z
                qw = transform.transform.rotation.w
                
                # Convert quaternion to rotation matrix
                matrix = quaternion_matrix([qx, qy, qz, qw])
                
                # Write data to file
                data_line = f"{x} {y} {z} " + " ".join(map(str, matrix.flatten()[:12])) + "\n"
                self.file.write(data_line)
                self.file.flush()  # Ensure data is written immediately
                
                rospy.loginfo(f"Saved pose: x={x:.2f}, y={y:.2f}, z={z:.2f}")
                
            except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
                rospy.logwarn(f"TF2 exception: {e}")
            
            self.rate.sleep()

    def __del__(self):
        if hasattr(self, 'file'):
            self.file.close()

if __name__ == '__main__':
    try:
        collector = HectorSLAMDataCollector()
        collector.collect_data()
    except rospy.ROSInterruptException:
        pass

