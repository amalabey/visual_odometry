from __future__ import print_function

import os, sys
os.chdir(os.path.expanduser('/home/amal/.virtualenvs/OpenCV-master-py2/local/lib'))
sys.path.append(os.path.expanduser('/home/amal/.virtualenvs/OpenCV-master-py2/local/lib/python2.7/dist-packages'))


import argparse
from os import path
from time import sleep
from utm import to_latlon
import glob
import cv2
import roslib
import rospy

import CameraParams_Parser as Cam_Parser
import GPS_VO
import Ground_Truth as GT
import Trajectory_Tools as TT
from Common_Modules import *
from py_MVO import VisualOdometry
from sensor_msgs.msg import CompressedImage

VERBOSE=False

class RosVisualOdometry:
    def __init__(self, cam_instrint_params, feature_detector_name, image_topic):
        self.img_id = 0
        self.subscriber = rospy.Subscriber(image_topic, CompressedImage, self.callback, queue_size = 1)
        self.cam_instrint_params = cam_instrint_params
        self.traj = np.zeros((600, 600, 3), dtype=np.uint8)
        self.T_v_dict = OrderedDict()
        self.vo = VisualOdometry(cam_instrint_params, feature_detector_name, None)

    def callback(self, ros_data):
        if VERBOSE:
            print("received image of type: %s" % ros_data.format)

        img_path = "image: "+self.img_id

        np_arr = np.fromstring(ros_data.data, np.uint8)
        imgKLT = cv2.imdecode(np_arr, cv2.CV_LOAD_IMAGE_COLOR) # Read the image for real-time trajectory
        img = cv2.imdecode(np_arr, cv2.CV_LOAD_IMAGE_MONO) # Read the image for Visual Odometry

        # Create a CLAHE object (contrast limiting adaptive histogram equalization)
        clahe = cv2.createCLAHE(clipLimit=5.0)
        img = clahe.apply(img)
        
        if self.vo.update(img, self.img_id):  # Updating the vectors in VisualOdometry class
            if self.img_id == 0:
                self.T_v_dict[img_path] = ([[0], [0], [0]])
            else:
                self.T_v_dict[img_path] = self.vo.cur_t   # Retrieve the translation vectors for dictionary
            cur_t = self.vo.cur_t  # Retrieve the translation vectors

            # ------- Windowed Displays ---------
            if self.img_id > 0:  # Set the points for the real-time trajectory window
                x, y, z = cur_t[0], cur_t[1], cur_t[2]
                TT.drawOpticalFlowField(imgKLT, self.vo.OFF_prev, self.vo.OFF_cur)  # Draw the features that were matched
            else:
                x, y, z = 0., 0., 0.

            self.traj = TT.RT_trajectory_window(self.traj, x, y, z, self.img_id)  # Draw the trajectory window
            # -------------------------------------

        self.img_id += 1  # Increasing the image id

def main(args):
    '''Initializes and cleanup ros node'''
    image_topic = "/usb_cam/image_raw"
    cam_instrint_params = np.array([[718.856, 0, 607.1928],
                                    [0, 718.856, 185.2157],
                                    [0, 0, 1]])
    feature_detector_name = "SIFT"

    ros_vo = RosVisualOdometry(cam_instrint_params, feature_detector_name, image_topic)
    rospy.init_node('image_vo', anonymous=True)
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down ROS Image feature detector module")

    cv2.destroyAllWindows()

if __name__ == '__main__':
    main(sys.argv)