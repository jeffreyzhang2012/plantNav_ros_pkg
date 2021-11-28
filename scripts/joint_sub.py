#!/usr/bin/env python3

import cv2 as cv
import threading
import random
from time import sleep
import Arm_Lib
import rospy
from sensor_msgs.msg import JointState
from math import pi


RA2DE = 180 / pi
Arm = Arm_Lib.Arm_Device()
rospy.init_node('listener', anonymous=True)
def callback(msg):
    global sbus
    if not isinstance(msg, JointState): return
    joints = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    for i in range(5): joints[i] = (msg.position[i] * RA2DE) + 90
    sbus.Arm_serial_servo_write6_array(joints, 100)
sub = rospy.Subscriber("/joint_states", JointState, callback)
sbus = Arm_Lib.Arm_Device()

def subscribe():
    while not rospy.is_shutdown():
        try:
            rospy.spin()
        except KeyboardInterrupt:
            break

if __name__ == '__main__':
    try:
        subscribe()
    except rospy.ROSInterruptException:
        pass
