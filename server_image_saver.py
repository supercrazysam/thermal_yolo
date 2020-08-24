#! /usr/bin/python

# rospy for the subscriber
import rospy
# ROS Image message
from sensor_msgs.msg import Image
# ROS Image message -> OpenCV2 image converter
from cv_bridge import CvBridge, CvBridgeError
# OpenCV2 for saving an image
import cv2

from time import  strftime,localtime
import time

# Instantiate CvBridge
bridge = CvBridge()

def image_callback(msg):
    print("Received an image!")
    try:
        # Convert your ROS Image message to OpenCV2
        cv2_img = bridge.imgmsg_to_cv2(msg, "bgr8")
    except CvBridgeError, e:
        print(e)
    else:
        # Save your OpenCV2 image as a jpeg 
        time = msg.header.stamp
        cv2.imwrite('/home/security/'+str(strftime("%Y-%m-%d_%Hh%Mm%S", localtime()) )+'.jpeg', cv2_img)
        rospy.sleep(1)

def main():
    rospy.init_node('image_listener')
    # Define your image topic
    image_topic = "/thermal_yolo/tracked"
    # Set up your subscriber and define its callback
    rospy.Subscriber(image_topic, Image, image_callback)
    # Spin until ctrl + c
    rospy.spin()

if __name__ == '__main__':
    main()
