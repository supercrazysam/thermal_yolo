#!/usr/bin/env python3
# -*- coding: utf-8 -*-



#######ROS
import rospy

from std_msgs.msg import String, Float64, Int16MultiArray, Header, Int32
from sensor_msgs.msg import Image, RegionOfInterest, CameraInfo
from cv_bridge import CvBridge, CvBridgeError

from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped

import sys
sys.path.append("/home/big/Music/sam_test/src/yolo_tracker/src")
######

import time
import warnings

import cv2
import numpy as np

import external_yolov3.darknet.darknet as darknet
from tools import generate_detections as gdet

warnings.filterwarnings('ignore')

width = 1280
height= 720

#YOLO
# Definition of the parameters
max_cosine_distance = 0.3 #0.3
nn_budget = None
nms_max_overlap = 1.0


    
metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
tracker = Tracker(metric, 0.1, 100,3)

######## FAST YOLO wrapper
import os
cwd = os.path.dirname(__file__)
print(cwd)
configPath = cwd+"/external_yolov3/darknet/cfg/yolov3-spp.cfg"
#weightPath = cwd+"/external_yolov3/darknet/yolov3-spp.weights"
weightPath = cwd+"/external_yolov3/darknet/yolov3-spp-thermal.weights"
metaPath   = cwd+"/external_yolov3/darknet/cfg/coco_accel.data"
    
netMain = darknet.load_net_custom(configPath.encode(
            "ascii"), weightPath.encode("ascii"), 0, 1)  # batch size = 1

metaMain = darknet.load_meta(metaPath.encode("ascii"))


darknet_image = darknet.make_image(darknet.network_width(netMain),
                                    darknet.network_height(netMain),3)

###########
def convertBack(x, y, w, h):
    xmin = int(round(x - (w / 2)))
    xmax = int(round(x + (w / 2)))
    ymin = int(round(y - (h / 2)))
    ymax = int(round(y + (h / 2)))
    return xmin, ymin, xmax, ymax


def cvDrawBoxes(detections, img):
    for detection in detections:
        x, y, w, h = detection[2][0],\
            detection[2][1],\
            detection[2][2],\
            detection[2][3]
        xmin, ymin, xmax, ymax = convertBack(
            float(x), float(y), float(w), float(h))
        pt1 = (xmin, ymin)
        pt2 = (xmax, ymax)
        cv2.rectangle(img, pt1, pt2, (0, 255, 0), 1)
        cv2.putText(img,
                    detection[0].decode() +
                    " [" + str(round(detection[1] * 100, 2)) + "]",
                    (pt1[0], pt1[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    [0, 255, 0], 2)
    return img

##########

def xyxy_to_xywh(xyxy):
    """Convert [x1 y1 x2 y2] box format to [x1 y1 w h] format."""
    if isinstance(xyxy, (list, tuple)):
        # Single box given as a list of coordinates
        assert len(xyxy) == 4
        x1, y1 = xyxy[2], xyxy[3]
        w = xyxy[0] - x1 + 1
        h = xyxy[1] - y1 + 1
        return (x1, y1, w, h)
    elif isinstance(xyxy, np.ndarray):
        # Multiple boxes given as a 2D ndarray
        return np.hstack((xyxy[:, 2:4], xyxy[:, 0:2] - xyxy[:, 2:4] + 1))
    else:
        raise TypeError('Argument xyxy must be a list, tuple, or numpy array.')
############


yolo_filter_size = 416 #416 #608

class yolo_tracker(object):
    def __init__(self):
                node_name = "yolo_tracker"
                self.node_name = node_name
                rospy.init_node(node_name)
                rospy.loginfo("Starting node " + str(node_name))
                rospy.on_shutdown(self.cleanup)
             
                #self.image_sub = rospy.Subscriber("/blended/image", Image, self.image_callback, queue_size=1)
                self.track_pub = rospy.Publisher("/person_tracking/tracked", Image, queue_size=1)
                self.posid_pub = rospy.Publisher("/person_tracking/person_num", Int32, queue_size=1)
                self.bridge = CvBridge()

                self.show = None

                self.posid_array = Path()

                


                     
    def image_callback(self,data):
            self.show = self.convert_image(data)
            self.time_header = data.header

            if self.show is None: return
  
            # image = Image.fromarray(frame)
            step1 = time.time()
            frame_rgb = cv2.cvtColor(self.show, cv2.COLOR_BGR2RGB)
            frame_resized = cv2.resize(frame_rgb,
                                   (darknet.network_width(netMain),
                                    darknet.network_height(netMain)),
                                   interpolation=cv2.INTER_LINEAR)

            darknet.copy_image_from_bytes(darknet_image,frame_resized.tobytes())

            detections = darknet.detect_image(netMain, metaMain, darknet_image, thresh=0.25)
            #detections is in x y w h in real pixel size already

            #boxs should receive in format similar to  [[584, 268, 160, 316]]

            if detections is None: return
##            image = cvDrawBoxes(detections, frame_resized)
##            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
##            cv2.imshow('Demo', image)
##            cv2.waitKey(1)

            
            detections = np.array(detections)
            detections = detections[ np.where( detections[:,0]==b'person' ) ]
            if detections is None: return
            #print("detections",detections)

            if len(detections)==0: return
            c = np.delete(detections,[0,1],1).squeeze() #remove id and conf, only bbox
            #print("c",c)
            #r =  #resize to coords on original image to compensate for the frame_resized
            if len(c.shape)==0:
                boxs = np.array( [ c.tolist() ] )
            else:   
                boxs = np.array([list(item) for item in c]) #formating
            #print("boxs",boxs)
            

##            boxs = xyxy_to_xywh(transformed)#.astype(np.uint8)
##
            
            boxs[:,2] = (boxs[:,2] /yolo_filter_size) * width  #w
            boxs[:,3] = (boxs[:,3] /yolo_filter_size) * height #h

            boxs[:,0] = (boxs[:,0] /yolo_filter_size) * width   - boxs[:,2]/2#x
            boxs[:,1] = (boxs[:,1] /yolo_filter_size) * height  - boxs[:,3]/2#y
            
            print("time for inference =>"+str(time.time()-step1))
            #print(darknet.network_width(netMain),darknet.network_height(netMain)) #608 #608
            # print("box_num",len(boxs))
            features = encoder(self.show,boxs)
            
            # score to 1.0 here).
            detections = [Detection(bbox, 1.0, feature) for bbox, feature in zip(boxs, features)]
            
            # Run non-maxima suppression.
            boxes = np.array([d.tlwh for d in detections])
            scores = np.array([d.confidence for d in detections])
            indices = preprocessing.non_max_suppression(boxes, nms_max_overlap, scores)
            detections = [detections[i] for i in indices]

            for box in boxes:

                
                bbox = track.to_tlbr()
                try:
                    cv2.rectangle(self.show, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),(255,255,255), 2)
                except ValueError:
                    break
                
            self.posid_pub.publish(Int32(data=len(boxes)))
    
            self.track_pub.publish(self.bridge.cv2_to_imgmsg(self.show, "bgr8"))
 


    def convert_image(self,ros_image):
        # Use cv_bridge() to convert the ROS image to OpenCV format
        try:
                cv_image = self.bridge.imgmsg_to_cv2(ros_image, "bgr8")       
                return np.array(cv_image, dtype=np.uint8)
        except CvBridgeError as e:
                return None
           
    def cleanup(self):
                    print("Shutting down openpose.")
                    rospy.signal_shutdown("ROSPy Shutdown")

    def mainloop(self):
        while not rospy.is_shutdown():
            msg=rospy.wait_for_message("/blended/image", Image)
            self.image_callback(msg)

try:
    x=yolo_tracker()
    x.mainloop()
    #rospy.spin()
except KeyboardInterrupt:
    print("Shutting down yolo.")
