import os, sys
os.chdir(os.path.expanduser('/home/amal/.virtualenvs/OpenCV-master-py2/local/lib'))
sys.path.append(os.path.expanduser('/home/amal/.virtualenvs/OpenCV-master-py2/local/lib/python2.7/dist-packages'))


import argparse
from os import path
from time import sleep
from utm import to_latlon
import glob
import CameraParams_Parser as Cam_Parser
import GPS_VO
import Ground_Truth as GT
import Trajectory_Tools as TT
from Common_Modules import *
from py_MVO import VisualOdometry
import rospy

def main():

    # Load required inputs
    images = sorted(glob.glob("/home/amal/work/visual_odometry/src/KITTI_sample/images/*.png"), reverse=False)
    cam_instrint_params = np.array([[718.856, 0, 607.1928],
                                    [0, 718.856, 185.2157],
                                    [0, 0, 1]])
    feature_detector_name = "SIFT"

    vo = VisualOdometry(cam_instrint_params, feature_detector_name, None)

    # Square for the real-time trajectory window
    traj = np.zeros((600, 600, 3), dtype=np.uint8)

    # ------------------ Image Sequence Iteration and Processing ---------------------
    # Gives each image an id number based position in images list
    img_id = 0
    T_v_dict = OrderedDict()  # dictionary with image and translation vector as value
    # Initial call to print 0% progress bar
    #TT.printProgress(img_id, len(images)-1, prefix='Progress:', suffix='Complete', barLength=50)
    print "Start processing images.."
    k = 0
    while k != 27:  # Iterating through all images

        imgKLT = cv2.imread(img_path)  # Read the image for real-time trajectory
        img = cv2.imread(img_path, 0)  # Read the image for Visual Odometry

        # Create a CLAHE object (contrast limiting adaptive histogram equalization)
        clahe = cv2.createCLAHE(clipLimit=5.0)
        img = clahe.apply(img)
        
        if vo.update(img, img_id):  # Updating the vectors in VisualOdometry class
            if img_id == 0:
                T_v_dict[img_path] = ([[0], [0], [0]])
            else:
                T_v_dict[img_path] = vo.cur_t   # Retrieve the translation vectors for dictionary
            cur_t = vo.cur_t  # Retrieve the translation vectors

            # ------- Windowed Displays ---------
            if img_id > 0:  # Set the points for the real-time trajectory window
                x, y, z = cur_t[0], cur_t[1], cur_t[2]
                TT.drawOpticalFlowField(imgKLT, vo.OFF_prev, vo.OFF_cur)  # Draw the features that were matched
            else:
                x, y, z = 0., 0., 0.

            traj = TT.RT_trajectory_window(traj, x, y, z, img_id)  # Draw the trajectory window
            # -------------------------------------

        #sleep(0.1) # Sleep for progress bar update
        img_id += 1  # Increasing the image id
        #TT.printProgress(i, len(images)-1, prefix='Progress:', suffix='Complete', barLength=50)  # update progress bar

    # -------- Plotting Trajectories ----------
    T_v = []
    for key, value in T_v_dict.items():
        T_v_dict[key] = np.array((value[0][0], value[2][0]))
        T_v.append((value[0][0], value[2][0]))

    TT.VO_plot(T_v)
    # -------------------------------------------
    k = cv2.waitKey(0)

    return

if __name__ == '__main__':
    main()