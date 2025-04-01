import os
import pickle
import shutil
import argparse
import numpy as np
from copy import deepcopy
import rosbag
import cv2
from cv_bridge import CvBridge
from tqdm import tqdm
from scipy.spatial.distance import cdist
from nav_msgs.msg import Odometry
import tf.transformations as tf_trans

LOGGING_FRAME_RATE = 30.0 # Frequency of logging in hz
image_topic = '/rs_mid/color/image_raw'
odom_topic = '/spot/odometry'

import tf
import numpy as np
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Pose, Point, Quaternion


def transform_pose(curr_pose, initial_pose):
    """
    Transforms the current odometry pose to a local frame defined by the initial_pose.
    
    :param odom_msg: nav_msgs/Odometry message
    :param initial_pose: geometry_msgs/Pose defining the origin of the local frame
    :return: Transformed position (x, y, z) and orientation (quaternion)
    """
    # Extract initial position and orientation
    init_pos = initial_pose.position
    init_ori = initial_pose.orientation
    init_translation = np.array([init_pos.x, init_pos.y, init_pos.z])
    init_quat = [init_ori.x, init_ori.y, init_ori.z, init_ori.w]

    # Get current position and orientation from odometry
    curr_pos = curr_pose.position
    curr_ori = curr_pose.orientation
    curr_translation = np.array([curr_pos.x, curr_pos.y, curr_pos.z])
    curr_quat = [curr_ori.x, curr_ori.y, curr_ori.z, curr_ori.w]

    # Compute relative translation
    rel_translation = curr_translation - init_translation

    # Convert initial orientation to rotation matrix
    init_rot_matrix = tf.transformations.quaternion_matrix(init_quat)

    # Invert rotation (transpose of rotation matrix)
    rot_matrix_inv = init_rot_matrix[:3, :3].T

    # Rotate the relative translation into the initial frame
    transformed_position = rot_matrix_inv.dot(rel_translation)

    # Compute relative orientation
    init_quat_inv = tf.transformations.quaternion_inverse(init_quat)
    rel_quat = tf.transformations.quaternion_multiply(init_quat_inv, curr_quat)
    roll, pitch, yaw = tf.transformations.euler_from_quaternion(rel_quat)

    return transformed_position, yaw

def main(args: argparse.Namespace):
    bags_path = args.input_dir
    all_bags = os.listdir(bags_path)
    
    for bag_i, bag in enumerate(all_bags):
        output_folder = os.path.join(args.output_dir, 't{0:05d}'.format(bag_i))
        os.makedirs(output_folder, exist_ok=True)

        bag = rosbag.Bag(os.path.join(bags_path, bag), "r")
        bridge = CvBridge()
        front_imgs = bag.read_messages(image_topic)
        odoms = bag.read_messages(odom_topic)

        count_img = 0
        past_t = 0
        all_img_timestamp = []
        for topic, msg, t in tqdm(front_imgs):
            delta_t = msg.header.stamp.to_sec() - past_t
            if delta_t > 1.0 / args.sample_rate:
                front_img = bridge.imgmsg_to_cv2(msg, "bgr8")
                cv2.imwrite(os.path.join(output_folder, f"{count_img}.jpg"), front_img)
                
                past_t = msg.header.stamp.to_sec()
                all_img_timestamp.append(past_t)
                count_img += 1   
        
        all_odom_timestamp = []
        all_odom = []
        for topic, msg, t in tqdm(odoms):
            all_odom.append(msg.pose.pose)
            all_odom_timestamp.append(msg.header.stamp.to_sec())
        
        all_img_timestamp = np.array(all_img_timestamp)
        all_odom_timestamp = np.array(all_odom_timestamp)

        dists = cdist(all_odom_timestamp[:, None], all_img_timestamp[:, None])
        closest_odom_idx = np.argmin(dists, axis=0)

        traj_data = {
            "position": [], 
            "yaw": [], 
            "frame_ids": [], 
            "intentions": []
        }

        initial_pose = all_odom[closest_odom_idx[0]]

        for i, idx in enumerate(closest_odom_idx):
            curr_pose = all_odom[idx]
            local_pos, yaw = transform_pose(curr_pose, initial_pose)

            print(local_pos[:2], yaw)

            traj_data["intentions"].append('forward')
            traj_data["frame_ids"].append(i)
            traj_data["position"].append(deepcopy(local_pos[:2]))
            traj_data["yaw"].append(deepcopy(yaw))

        with open(os.path.join(output_folder, 'traj_data.pkl'), 'wb') as f:
            pickle.dump(traj_data, f) 

        print("Wrote",  't{0:05d}'.format(bag_i), "of length:", len(closest_odom_idx))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # get arguments for the recon input dir and the output dir
    # add dataset name
    parser.add_argument(
        "--dataset-name",
        "-d",
        type=str,
        help="name of the dataset (must be in process_config.yaml)",
        default="inet",
    )
    parser.add_argument(
        "--input-dir",
        "-i",
        type=str,
        help="path of the datasets with rosbags",
        default="/home/zishuo/i4_overfit_2/",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        default="/home/zishuo/i4_overfit_gnm_data_2",
        type=str,
        help="path for processed dataset (default: ../datasets/tartan_drive/)",
    )
    # sampling rate
    parser.add_argument(
        "--sample-rate",
        "-s",
        default=4.0,
        type=float,
        help="sampling rate (default: 4.0 hz)",
    )

    args = parser.parse_args()
    main(args)