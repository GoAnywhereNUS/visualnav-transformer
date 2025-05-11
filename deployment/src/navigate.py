import matplotlib.pyplot as plt
import os
from typing import Tuple, Sequence, Dict, Union, Optional, Callable
import numpy as np
import torch
import torch.nn as nn
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from cv_bridge import CvBridge
import matplotlib.pyplot as plt
import yaml
from cam_utils import *

# ROS
import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import Bool, Float32MultiArray
from geometry_msgs.msg import Twist
from utils import msg_to_pil, to_numpy, transform_images, load_model

from vint_train.training.train_utils import get_action
import torch
from PIL import Image as PILImage
import numpy as np
import argparse
import yaml
import time
import cv2
from sensor_msgs.msg import Joy

# UTILS
from topic_names import (IMAGE_TOPIC,
                        WAYPOINT_TOPIC,
                        SAMPLED_ACTIONS_TOPIC)
from geometry_msgs.msg import Twist
from vint_train.models.gnm.gnm_vae import GNM_VAE_Inference
import sys


class GNM:
    # CONSTANTS
    TOPOMAP_IMAGES_DIR = "../topomaps/images"
    MODEL_WEIGHTS_PATH = "../model_weights"
    #ROBOT_CONFIG_PATH ="../config/robot.yaml"
    ROBOT_CONFIG_PATH = "../config/spot.yaml"
    MODEL_CONFIG_PATH = "../config/models.yaml"
    with open(ROBOT_CONFIG_PATH, "r") as f:
        robot_config = yaml.safe_load(f)
    MAX_V = robot_config["max_v"]
    MAX_W = robot_config["max_w"]
    RATE = robot_config["frame_rate"] 
    IMAGE_TOPIC = "/rs_mid/color/image_raw"
    WAYPOINT_TOPIC = "/gnm/waypoint"
    VEL_TOPIC = robot_config["vel_navi_topic"]
    print("Publishing to:", IMAGE_TOPIC, WAYPOINT_TOPIC, SAMPLED_ACTIONS_TOPIC)

    def __init__(self, args):
        self.context_queue = []
        self.context_size = None
        self.subgoal = []
        self.joy_enable = False
        # Load the model 
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Using device:", self.device)

        # load model parameters
        with open(self.MODEL_CONFIG_PATH, "r") as f:
            self.model_paths = yaml.safe_load(f)

        self.model_config_path = self.model_paths[args.model]["config_path"]
        with open(self.model_config_path, "r") as f:
            self.model_params = yaml.safe_load(f)

        self.context_size = self.model_params["context_size"]
        self.model_type = self.model_params["model_type"]

        # load model weights
        self.ckpth_path = self.model_paths[args.model]["ckpt_path"]
        if os.path.exists(self.ckpth_path):
            print(f"Loading model from {self.ckpth_path}")
        else:
            raise FileNotFoundError(f"Model weights not found at {self.ckpth_path}")
        self.model = load_model(
            self.ckpth_path,
            self.model_params,
            self.device,
        )
        self.model = self.model.to(self.device)
        self.model.eval()

        self.inference = GNM_VAE_Inference(self.model)

        # load topomap
        self.topomap_filenames = sorted(os.listdir(os.path.join(
            self.TOPOMAP_IMAGES_DIR, args.dir)), key=lambda x: int(x.split(".")[0]))
        self.topomap_dir = f"{self.TOPOMAP_IMAGES_DIR}/{args.dir}"
        self.num_nodes = len(os.listdir(self.topomap_dir))
        self.topomap = []
        for i in range(self.num_nodes):
            image_path = os.path.join(self.topomap_dir, self.topomap_filenames[i])
            self.topomap.append(PILImage.open(image_path))

        self.closest_node = 0
        assert -1 <= args.goal_node < len(self.topomap), "Invalid goal index"
        if args.goal_node == -1:
            self.goal_node = len(self.topomap) - 1
        else:
            self.goal_node = args.goal_node
        self.reached_goal = False
        cv2.namedWindow("Live/Subgoal")

        # ROS
        rospy.init_node("EXPLORATION", anonymous=False)
        self.rate = rospy.Rate(self.RATE)
        self.cv_bridge = CvBridge()

        self.waypoint_pub = rospy.Publisher(
            self.WAYPOINT_TOPIC, Float32MultiArray, queue_size=1)  
        self.goal_pub = rospy.Publisher("/gnm/reached_goal", Bool, queue_size=1)
        self.vel_pub = rospy.Publisher(self.VEL_TOPIC, Twist, queue_size=1)
        self.gradcam_pub = rospy.Publisher("/gradcam", Image, queue_size=5)

        self.image_curr_msg = rospy.Subscriber(
            self.IMAGE_TOPIC, Image, self.callback_obs, queue_size=1)
        if self.model_params["model_type"] == "gnm_vae":
            self.vel_sub = rospy.Subscriber(
                self.VEL_TOPIC, 
                Twist, 
                lambda msg: self.inference.action_cache.append((msg.linear.x, msg.angular.z)),
                queue_size=1
            )
        self.joy_sub = rospy.Subscriber('/bluetooth_teleop/joy', Joy, self.callback_joy, queue_size=1)

        print("Registered with master node. Waiting for image observations...")

        self.recovery_mode = False
        self.goal_img_id = 0
        self.args = args

    def run(self):
        while not rospy.is_shutdown():
            if self.joy_enable:
                print(self.inference.action_cache)
                chosen_waypoint = np.zeros(4)
                chosen_distance = 0
                if len(self.context_queue) > self.model_params["context_size"]:
                    start = max(self.closest_node - self.args.radius, 0)
                    end = min(self.closest_node + self.args.radius + 1, self.goal_node)
                    distances = []
                    waypoints = []
                    batch_obs_imgs = []
                    batch_goal_data = []
                    for i, sg_img in enumerate(self.topomap[start: end + 1]):
                        transf_obs_img = transform_images(self.context_queue, self.model_params["image_size"])
                        goal_data = transform_images(sg_img, self.model_params["image_size"])
                        batch_obs_imgs.append(transf_obs_img)
                        batch_goal_data.append(goal_data)
                        
                    # predict distances and waypoints
                    batch_obs_imgs = torch.cat(batch_obs_imgs, dim=0).to(self.device)
                    ############ simulate sensor failure ########
                    #batch_obs_imgs = torch.zeros_like(batch_obs_imgs).to(self.device)
                    #############################################
                    batch_goal_data = torch.cat(batch_goal_data, dim=0).to(self.device)
                    yaw = 0.0
                    if self.model_type == 'gnm_vae':
                        output = self.inference(batch_obs_imgs, batch_goal_data, yaw, self.context_queue[-1])
                        if len(output) == 5:
                            # Output from recovery strategy
                            (v, w), _, _, visualisation, help = output
                            print("Recovery action: ", v, w)
                            distances, waypoints = None, None
                            if help:
                                print("Need human intervention. STOPPING!")
                                self.goal_pub.publish(True)
                                #sys.exit(0)

                        elif len(output) == 6:
                            # Output from learned policy
                            distances, waypoints, _, _, visualisation, _ = output
                            v, w = None, None
                        else:
                            raise Exception("Invalid output from GNM VAE model")

                        if visualisation is not None:
                            # TODO
                            self.gradcam_pub.publish(self.cv_bridge.cv2_to_imgmsg(cv2.cvtColor(visualisation, cv2.COLOR_RGB2BGR)))
                    else:
                        distances, waypoints = self.model(batch_obs_imgs, batch_goal_data)

                    # If actions issued are for recovery, we pause the PD controller and
                    # directly send actions to /cmd_vel
                    if distances is None and waypoints is None:
                        self.recovery_mode = True
                        self.goal_pub.publish(True) # Pauses the PD controller so that we can take over

                        assert v is not None and w is not None
                        vel_msg = Twist()
                        vel_msg.linear.x = v
                        vel_msg.angular.z = w                    
                        # vel_pub.publish(vel_msg)
                        self.rate.sleep()
                        continue

                    else:
                        if self.recovery_mode:
                            # First step taken in normal mode after exiting recovery
                            self.recovery_mode = False
                            self.goal_pub.publish(False) # Unpause the PD controller and relinquish control of cmd_vel

                    distances = to_numpy(distances)
                    waypoints = to_numpy(waypoints)

                    if self.args.manual_goal:
                        key = cv2.waitKey(10)
                        if key & 0xFF == ord('a'):
                            self.goal_img_id = max(0, self.goal_img_id - 1)
                        elif key & 0xFF == ord('d'):
                            self.goal_img_id = min(len(self.topomap) - 1, self.goal_img_id + 1)
                        self.closest_node = self.goal_img_id

                        select_index = self.args.radius
                        if self.goal_img_id - self.args.radius < 0:
                            select_index = self.goal_img_id

                        chosen_waypoint = waypoints[select_index][self.args.waypoint]
                        chosen_distance = distances[select_index]
                        sg_img = self.topomap[self.goal_img_id]
                        closest_node_image = np.array(self.topomap[self.goal_img_id].resize((320, 240)))

                        live_image = self.context_queue[-1].resize((320, 240))
                        combined_image = np.concatenate((np.array(live_image), closest_node_image), axis=1)
                        cv2.putText(combined_image, str(self.goal_img_id) + "/" + str(len(self.topomap)-1), (340,20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0))
                        cv2.imshow('Live/Subgoal', combined_image)
                        if key == ord('q'):
                            print("Shutting down...")
                            sys.exit(0)
                    
                    else:
                        # look for closest node
                        closest_node_in_radius = np.argmin(distances)
                        print(distances[closest_node_in_radius])

                        # chose subgoal and output waypoints
                        if distances[closest_node_in_radius] > self.args.close_threshold:
                            chosen_waypoint = waypoints[closest_node_in_radius][self.args.waypoint]
                            sg_img = self.topomap[start + closest_node_in_radius]
                        else:
                            chosen_waypoint = waypoints[min(
                                closest_node_in_radius + 1, len(waypoints) - 1)][self.args.waypoint]
                            sg_img = self.topomap[start + min(closest_node_in_radius + 1, len(waypoints) - 1)]

                        self.closest_node = start + closest_node_in_radius
                        closest_node_image = np.array(self.topomap[self.closest_node].resize((320, 240)))

                        live_image = self.context_queue[-1].resize((320, 240))
                        combined_image = np.concatenate((np.array(live_image), closest_node_image), axis=1)
                        cv2.putText(combined_image, str(self.closest_node) + "/" + str(len(self.topomap)-1), (340,20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0))
                        cv2.imshow('Live/Subgoal', combined_image)
                        if cv2.waitKey(10) == ord('q'):
                            print("Shutting down...")
                            sys.exit(0)


                #chosen_waypoint[0] *= 4
                #chosen_waypoint[1] *= 4
                #print(chosen_waypoint, chosen_distance)
                # RECOVERY MODE
                if self.model_params["normalize"]:
                    chosen_waypoint[:2] *= (self.MAX_V / self.RATE)
                waypoint_msg = Float32MultiArray()
                waypoint_msg.data = chosen_waypoint
                self.waypoint_pub.publish(waypoint_msg)
                
                # comment to only issue one image goal
                #reached_goal = closest_node == goal_node
                #goal_pub.publish(reached_goal)
                #if reached_goal:
                #    print("Reached goal! Stopping...")
                
            self.rate.sleep()


    def callback_obs(self, msg):
        obs_img = msg_to_pil(msg)
        if self.context_size is not None:
            if len(self.context_queue) < self.context_size + 1:
                self.context_queue.append(obs_img)
            else:
                self.context_queue.pop(0)
                self.context_queue.append(obs_img)

    def callback_joy(self, joy_msg):
        self.joy_enable = (joy_msg.buttons[5] == 1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Code to run GNM DIFFUSION EXPLORATION on the locobot")
    parser.add_argument(
        "--model",
        "-m",
        default="vint",
        type=str,
        help="model name (only nomad is supported) (hint: check ../config/models.yaml) (default: nomad)",
    )
    parser.add_argument(
        "--waypoint",
        "-w",
        default=2, # close waypoints exihibit straight line motion (the middle waypoint is a good default)
        type=int,
        help=f"""index of the waypoint used for navigation (between 0 and 4 or 
        how many waypoints your model predicts) (default: 2)""",
    )
    parser.add_argument(
        "--dir",
        "-d",
        default="topomap",
        type=str,
        help="path to topomap images",
    )
    parser.add_argument(
        "--goal-node",
        "-g",
        default=-1,
        type=int,
        help="""goal node index in the topomap (if -1, then the goal node is 
        the last node in the topomap) (default: -1)""",
    )
    parser.add_argument(
        "--close-threshold",
        "-t",
        default=3,
        type=int,
        help="""temporal distance within the next node in the topomap before 
        localizing to it (default: 3)""",
    )
    parser.add_argument(
        "--radius",
        "-r",
        default=4,
        type=int,
        help="""temporal number of locobal nodes to look at in the topopmap for
        localization (default: 2)""",
        )
    parser.add_argument(
        "--num-samples",
        "-n",
        default=8,
        type=int,
        help=f"Number of actions sampled from the exploration model (default: 8)",
    )
    parser.add_argument(
        '--manual_goal',
        action='store_true'
    )
    args = parser.parse_args()

    GNM_node = GNM(args)
    GNM_node.run()


