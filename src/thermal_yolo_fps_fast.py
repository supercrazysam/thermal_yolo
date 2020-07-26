#!/usr/bin/env python3
# -*- coding: utf-8 -*-



#######ROS
import rospy

from std_msgs.msg import String, Float64, Int16MultiArray, Header, Int32
from sensor_msgs.msg import Image, RegionOfInterest, CameraInfo
from cv_bridge import CvBridge, CvBridgeError

from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped

##import sys
##sys.path.append("/home/big/Music/sam_test/src/yolo_tracker/src")
######

import time
import warnings

import cv2
import numpy as np

import external_yolov3.darknet.darknet as darknet 

warnings.filterwarnings('ignore')

width = 640#1280
height= 512#720

#YOLO
# Definition of the parameters
max_cosine_distance = 0.3
nn_budget = None
nms_max_overlap = 1.0

######## FAST YOLO wrapper
import os
cwd = os.path.dirname(__file__)
print(cwd)
configPath = cwd+"/external_yolov4/darknet/yolov4_thermal.cfg"
weightPath = cwd+"/external_yolov4/darknet/yolov4_thermal.weights"
metaPath   = cwd+"/external_yolov4/darknet/cfg/coco_yolov4_thermal.data"
    
netMain = darknet.load_net_custom(configPath.encode(
            "ascii"), weightPath.encode("ascii"), 0, 1)  # batch size = 1

metaMain = darknet.load_meta(metaPath.encode("ascii"))


darknet_image = darknet.make_image(darknet.network_width(netMain),
                                    darknet.network_height(netMain),3)

###########
def convertBack(x, y, w, h):
    xmin = int(np.round(x - (w / 2)))
    xmax = int(np.round(x + (w / 2)))
    ymin = int(np.round(y - (h / 2)))
    ymax = int(np.round(y + (h / 2)))
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
def xywh_to_xyxy(xywh):
    """Convert [x1 y1 w h] box format to [x1 y1 x2 y2] format."""
    if isinstance(xywh, (list, tuple)):
        # Single box given as a list of coordinates
        assert len(xywh) == 4
        x1, y1 = xywh[0], xywh[1]
        x2 = x1 + np.maximum(0., xywh[2] - 1.)
        y2 = y1 + np.maximum(0., xywh[3] - 1.)
        return (x1, y1, x2, y2)
    elif isinstance(xywh, np.ndarray):
        # Multiple boxes given as a 2D ndarray
        return np.hstack(
            (xywh[:, 0:2], xywh[:, 0:2] + np.maximum(0, xywh[:, 2:4] - 1))
        )
    else:
        raise TypeError('Argument xywh must be a list, tuple, or numpy array.')


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
             
                #self.image_sub = rospy.Subscriber("/thermal_yolo/image", Image, self.image_callback, queue_size=1)
                self.track_pub = rospy.Publisher("/thermal_yolo/tracked", Image, queue_size=1)
                self.posid_pub = rospy.Publisher("/thermal_yolo/person_num", Int32, queue_size=1)
                self.bridge = CvBridge()

                self.show = None

                self.posid_array = Path()

                # 参数初始化
                self.realtime_fps = '0.0000'
                self.start_time = time.time()
                self.fps_interval = 1
                self.fps_count = 0
                self.run_timer = 0
                self.frame_count = 0
                
                self.camera = cv2.VideoCapture(0) #(512,640,3)

                


                     
    def camera_loop(self):
            
            get, self.show = self.camera.read()

            if not get: return
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

            if detections is None: 
                print("No human in sight!")
                return
##            image = cvDrawBoxes(detections, frame_resized)
##            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
##            cv2.imshow('Demo', image)
##            cv2.waitKey(1)

            
            detections = np.array(detections)
            try:
                detections = detections[ np.where( detections[:,0]==b'person' ) ]
            except:
                print("No human in sight!")
                return
            if detections is None: 
                print("No human in sight!")
                return
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

            self.fps_count += 1
            self.frame_count += 1
#boxs = xyxy_to_xywh(boxs)#.astype(np.uint8)
            boxs = xywh_to_xyxy(boxs)
            
            print(boxs)         

            boxs[:,2] = (boxs[:,2] /yolo_filter_size) * width  #w
            boxs[:,3] = (boxs[:,3] /yolo_filter_size) * height #h

            boxs[:,0] = (boxs[:,0] /yolo_filter_size) * width   #- boxs[:,2]/2#x
            boxs[:,1] = (boxs[:,1] /yolo_filter_size) * height  #- boxs[:,3]/2#y
            
            print("time for inference =>"+str(time.time()-step1))
            #print(darknet.network_width(netMain),darknet.network_height(netMain)) #608 #608
            # print("box_num",len(boxs))
            
            for bbox in boxs:

                box_width  = bbox[2] - bbox[0]
                box_height = bbox[3] - bbox[1]

                bbox[0] = bbox[0] - int(box_width/2)
                bbox[1] = bbox[1] - int(box_height/2)
                
                self.posid_array = Path()
            
                try:
                    cv2.rectangle(self.show, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),(255,255,255), 2)
                except ValueError:
                    break
                #cv2.putText(self.show, str(track.track_id),(int(bbox[0]), int(bbox[1])),0, 5e-3 * 200, (0,255,0),2)


            # 显示实时FPS值
            if (time.time() - self.start_time) > self.fps_interval:
                # 计算这个interval过程中的帧数，若interval为1秒，则为FPS
                self.realtime_fps = self.fps_count / (time.time() - self.start_time)
                self.fps_count = 0  # 帧数清零
                self.start_time = time.time()
            fps_label = "FPS:"+str(self.realtime_fps)
            cv2.putText(self.show, fps_label, (width-160, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)


            # 显示目前的运行时长及总帧数
            if self.frame_count == 1:
                self.run_timer = time.time()
            run_time = time.time() - self.run_timer
            time_frame_label = "Frame:"+str(self.frame_count)
            self.posid_pub.publish(Int32(data=len(boxs)))
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
            self.camera_loop()

try:
    x=yolo_tracker()
    x.mainloop()
    #rospy.spin()
except KeyboardInterrupt:
    print("Shutting down yolo.")
