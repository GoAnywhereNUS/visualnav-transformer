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

LOGGING_FRAME_RATE = 30.0 # Frequency of logging in hz
image_topic = '/rs_mid/color/image_raw'
vel_topic = '/cmd_vel'

def main(args: argparse.Namespace):
    bags_path = args.input_dir
    all_bags = os.listdir(bags_path)
    
    for bag_i, bag in enumerate(all_bags):
        output_folder = os.path.join(args.output_dir, 't{0:05d}'.format(bag_i))
        os.makedirs(output_folder, exist_ok=True)

        bag = rosbag.Bag(os.path.join(bags_path, bag), "r")
        bridge = CvBridge()
        front_imgs = bag.read_messages(image_topic)
        vels = bag.read_messages(vel_topic)

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
        
        all_vel_timestamp = []
        all_vel = []
        for topic, msg, t in tqdm(vels):
            all_vel.append([msg.linear.x, msg.angular.z])
            all_vel_timestamp.append(t.to_sec())
        
        all_img_timestamp = np.array(all_img_timestamp)
        all_vel_timestamp = np.array(all_vel_timestamp)

        dists = cdist(all_vel_timestamp[:, None], all_img_timestamp[:, None])
        closest_vel_idx = np.argmin(dists, axis=0)


        traj_data = {
            "position": [], 
            "yaw": [], 
            "frame_ids": [], 
            "intentions": []
        }

        pos = np.zeros(3) # x, y, yaw (used for integrating only, discarded later)
        last_frame_id = None
        # TODO:consider to use high resolution logging rate before downsample
        logging_frame_dt = 1.0 / args.sample_rate

        for i, idx in enumerate(closest_vel_idx):
            v, w = all_vel[idx]
            if last_frame_id is not None:
                pos += np.array([
                    v * np.cos(pos[2]) * logging_frame_dt,
                    v * np.sin(pos[2]) * logging_frame_dt,
                    w * logging_frame_dt
                ])

                # Angle bounds on yaw
                pos[2] = pos[2] % (2 * np.pi)

            traj_data["intentions"].append('forward')
            traj_data["frame_ids"].append(i)
            traj_data["position"].append(deepcopy(pos[:2]))
            traj_data["yaw"].append(deepcopy(pos[2]))
            last_frame_id = i

        with open(os.path.join(output_folder, 'traj_data.pkl'), 'wb') as f:
            pickle.dump(traj_data, f) 

        print("Wrote",  't{0:05d}'.format(bag_i), "of length:", len(closest_vel_idx))


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
        default="/home/zishuo/i4_overfit/",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        default="/home/zishuo/i4_overfit_gnm_data",
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