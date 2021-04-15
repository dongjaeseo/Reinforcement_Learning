#! /usr/bin/env python

import rospy
import numpy as np
from nav_msgs.msg import Odometry

def callback(data):
    current_x = data.pose.pose.position.x
    current_y = data.pose.pose.position.y
    print('x_data:', current_x, 'y_data:', current_y)

rospy.init_node('odom')
sub = rospy.Subscriber('/slamware_ros_sdk_server_node/odom', Odometry, callback)
rospy.spin()