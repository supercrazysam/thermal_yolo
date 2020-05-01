#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import rospy
import cv2
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

import glob

class image_converter:

    def __init__(self):
        rospy.init_node('image_converter', anonymous=True)
        self.image_pub = rospy.Publisher("/thermal_yolo/image",Image,queue_size=1)

        self.bridge = CvBridge()

        #self.camera = cv2.VideoCapture(0)
        #self.image_list = glob.glob("/home/sam/Downloads/FLIR_ADAS_1_3/train/Annotated_thermal_8_bit/*.jpeg")
        self.image_list = glob.glob("/home/sam/Annotated_thermal_8_bit/*.jpeg")
        self.rate = rospy.Rate(10)
    



    def mainloop(self):
        try:
            for image_file in self.image_list:
                image = cv2.imread(image_file)
                #print(image)
                try:
                    self.image_pub.publish(self.bridge.cv2_to_imgmsg(image, "bgr8"))
                    self.rate.sleep()
                except CvBridgeError as e:
                    print(e)
        except KeyboardInterrupt:
            print("Shutting down")
            cv2.destroyAllWindows()


  
  

x = image_converter()
x.mainloop()

