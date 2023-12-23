import rospy
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image
from std_msgs.msg import Bool, Float32MultiArray, Float32

import torch
import cv2
from PIL import Image as PILImage
import numpy as np
import os
import argparse
import yaml
import time
import bisect
import itertools
from collections import deque

from utils import msg_to_pil, to_numpy, transform_images, load_model
from vint_train.training.train_utils import get_action
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

MODEL_WEIGHTS_PATH = "../model_weights"
ROBOT_CONFIG_PATH ="../config/spot.yaml"
MODEL_CONFIG_PATH = "../config/models.yaml"
with open(ROBOT_CONFIG_PATH, "r") as f:
    robot_config = yaml.safe_load(f)
MAX_V = robot_config["max_v"]
MAX_W = robot_config["max_w"]
RATE = robot_config["frame_rate"]
IMAGE_TOPIC = "/rs_mid/color/image_raw"
ODOM_TOPIC = "/spot/odometry"

# GLOBALS
odom_queue = deque(maxlen=30) # Buffer of past odometry messages (assuming 30Hz sim)
curr_sensor_msg = None

window_context_queue = None # Sensor images used directly as context if only using latest time window
zupt_queue = None # Flags indicating if robot has zero velocity, for each context queue element
moving_context_queue = None # Last sequence of non-zero velocity sensor images

tlx, tly = -1, -1
brx, bry = -1, -1
movex, movey = -1, -1
image_crop = None
drawing = False

dist_queue = deque(maxlen=RATE)
min_dist = None

# Load the model (locobot uses a NUC, so we can't use a GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

def find_closest_odom(query_timestamp):
    if len(odom_queue) == 0:
        return None
    
    timestamps = [ts for ts, _ in odom_queue]

    # Use bisect_left to find the index where the query_timestamp would be inserted
    index = bisect.bisect_left(timestamps, query_timestamp)

    if index == 0:
        return odom_queue[0][1] # query_timestamp is before the first datum in the buffer
    elif index == len(odom_queue):
        return odom_queue[-1][1] # query_timestamp is after the last datum in the buffer
    else:
        # query_timestamp falls between two timestamps in the buffer
        # Determine which datum is closer to the query_timestamp
        left_timestamp = odom_queue[index - 1][0]
        right_timestamp = odom_queue[index][0]

        if abs(query_timestamp - left_timestamp) <= abs(query_timestamp - right_timestamp):
            return odom_queue[index - 1][1] # Return left datum
        else:
            return odom_queue[index][1] # Return right datum

def image_callback(msg):
    global curr_sensor_msg
    curr_sensor_msg = msg

def odom_callback(msg):
    odom_queue.append((msg.header.stamp, msg))

# Mouse callback function to select rectangular region
def draw_rectangle(event, x, y, flags, param):
    global tlx, tly, brx, bry, movex, movey, curr_sensor_msg, image_crop, drawing
    if event == cv2.EVENT_MOUSEMOVE:
        movex, movey = x, y
    elif event == cv2.EVENT_LBUTTONDOWN:
        tlx, tly = x, y
        movex, movey = x, y
        drawing = True
    elif event == cv2.EVENT_LBUTTONUP:
        if tlx == x or tly == y:
            return

        if tlx < x: tlx, brx = tlx, x
        else: tlx, brx = x, tlx
        if tly < y: tly, bry = tly, y
        else: tly, bry = y, tly

        if curr_sensor_msg is not None:
            img = np.asarray(msg_to_pil(curr_sensor_msg))
            cv2.namedWindow("Selected")
            cv2.imshow("Selected", img[tly:bry, tlx:brx])
            
            cv2.namedWindow("Crop")
            cropped = bbox2crop(img, tlx, tly, brx, bry)
            cv2.imshow("Crop", cropped)

            image_crop = cropped

        tlx, tly = -1,-1
        brx, bry = -1,-1
        movex, movey = -1,-1
        drawing = False

# Fit a given bounding box inside an image crop with
# same aspect ratio as original image
def bbox2crop(img, tlx, tly, brx, bry):
    cx = 0.5 * (tlx + brx)
    cy = 0.5 * (tly + bry)
    ih, iw, _ = img.shape
    bboxw, bboxh = brx - tlx, bry - tly
    img_ar = float(iw) / float(ih)
    bbox_ar = float(bboxw) / float(bboxh)

    if img_ar > bbox_ar:
        # Fit bbox's height within the crop
        cropw, croph = img_ar * bboxh, bboxh
    else:
        # Fit bbox's width within the crop
        cropw, croph = bboxw, bboxw / img_ar

    tlx = int(max(0, cx - (0.5 * cropw)))
    brx = int(min(iw-1, cx + (0.5 * cropw)))
    tly = int(max(0, cy - (0.5 * croph)))
    bry = int(min(ih-1, cy + (0.5 * croph)))

    return img[tly:bry, tlx:brx]


def main(args: argparse.Namespace):
    # Load model params
    with open(MODEL_CONFIG_PATH, "r") as f:
        model_paths = yaml.safe_load(f)
    
    model_config_path = model_paths[args.model]["config_path"]
    with open(model_config_path, "r") as f:
        model_params = yaml.safe_load(f)
    
    model_params["min_linear_vel"] = 0.05
    model_params["min_angular_vel"] = 0.03
    print(model_params)

    # Initialise globals that require model_params
    global window_context_queue, moving_context_queue, zupt_queue
    window_context_queue = deque(maxlen=model_params["context_size"] + 1) # Sensor images used directly as context if only using latest time window
    zupt_queue = deque(maxlen=model_params["context_size"] + 1) # Flags indicating if robot has zero velocity, for each context queue element
    moving_context_queue = deque(maxlen=model_params["context_size"] + 1) # Last sequence of non-zero velocity sensor images

    # Load model weights
    ckpth_path = model_paths[args.model]["ckpt_path"]
    if os.path.exists(ckpth_path):
        print(f"Loading model from {ckpth_path}")
    else:
        raise FileNotFoundError(f"Model weights not found at {ckpth_path}")
    model = load_model(
            ckpth_path, 
            model_params, 
            device,
    )
    model.eval()

    print("Loaded model of size: ", sum(p.numel() for p in model.parameters()))

    if model_params["model_type"] == "nomad":
        num_diffusion_iters = model_params["num_diffusion_iters"]
        noise_scheduler = DDPMScheduler(
            num_train_timesteps=model_params["num_diffusion_iters"],
            beta_schedule='squaredcos_cap_v2',
            clip_sample=True,
            prediction_type='epsilon'
        )

    global IMAGE_TOPIC, ODOM_TOPIC
    if args.sim:
        IMAGE_TOPIC = "/gibson_ros/camera/rgb/image"
        ODOM_TOPIC = "/ground_truth_odom"

    # Set up ROS node
    rospy.init_node("gnm", anonymous=False)
    rate = rospy.Rate(RATE)
    image_sub = rospy.Subscriber(IMAGE_TOPIC, Image, image_callback, queue_size=1)
    odom_sub = rospy.Subscriber(ODOM_TOPIC, Odometry, odom_callback, queue_size=5)
    waypoint_pub = rospy.Publisher("/gnm/waypoint", Float32MultiArray, queue_size=5)
    goal_pub = rospy.Publisher("/gnm/reached_goal", Bool, queue_size=5)

    # Debug
    debug_dist_pub = rospy.Publisher("/dbg/dist", Float32, queue_size=1)
    debug_avg_dist_pub = rospy.Publisher("/dbg/avg", Float32, queue_size=1)
    debug_min_dist_pub = rospy.Publisher("/dbg/min", Float32, queue_size=1)

    global image_crop, dist_queue, min_dist
    cv2.namedWindow("Viz")
    cv2.setMouseCallback("Viz", draw_rectangle)

    while not rospy.is_shutdown():
        if curr_sensor_msg is not None:
            print("Curr sensor message")
            curr_im_pil = msg_to_pil(curr_sensor_msg)
            curr_im_array = np.asarray(curr_im_pil)
            curr_im_odom = find_closest_odom(curr_sensor_msg.header.stamp)
            if curr_im_odom is None:
                rospy.loginfo("Waiting for odom...")
                continue
            curr_im_zupt = (abs(curr_im_odom.twist.twist.linear.x) < model_params["min_linear_vel"]
                            and abs(curr_im_odom.twist.twist.angular.z) < model_params["min_angular_vel"])

            # Selection and visualisation of image cropping
            if drawing and (movex, movey) != (-1, -1):
                cv2.rectangle(curr_im_array, (tlx, tly), (movex, movey), (0, 255, 0), 2)
            cv2.imshow("Viz", curr_im_array)
            if cv2.waitKey(1) == ord('q'):
                break
            if cv2.waitKey(1) == ord('s'):
                image_crop = None
            if cv2.waitKey(1) == ord('p'):
                pass

            # Update context_queue
            print("Updating context queue")
            window_context_queue.append(curr_im_pil)
            zupt_queue.append(curr_im_zupt)

            if args.only_moving_contexts:
                if not any(zupt_queue):
                    print("Updating context queue")
                    moving_context_queue = window_context_queue
                context_queue = moving_context_queue
            else:
                context_queue = window_context_queue
            context_queue = list(context_queue)
            
            # If there is an image crop and valid embodiment context, run GNM
            if image_crop is not None and len(context_queue) == model_params["context_size"] + 1:
                pil_crop = PILImage.fromarray(np.uint8(image_crop))

                if model_params["model_type"] == "nomad": # NoMaD
                    obs_images = transform_images(context_queue, model_params["image_size"], center_crop=False)
                    obs_images = torch.cat(torch.split(obs_images, 3, dim=1), dim=1).to(device)
                    mask = torch.zeros(1).long().to(device)
                    goal_image = transform_images(pil_crop, model_params["image_size"], center_crop=False).to(device)

                    start_time = time.time()
                    obsgoal_cond = model('vision_encoder', obs_img=obs_images.repeat(len(goal_image), 1, 1, 1), goal_img=goal_image, input_goal_mask=mask.repeat(len(goal_image)))
                    dists = model("dist_pred_net", obsgoal_cond=obsgoal_cond)
                    dists = to_numpy(dists.flatten())

                    with torch.no_grad():
                        # encoder vision features
                        if len(obsgoal_cond.shape) == 2:
                            obs_cond = obsgoal_cond.repeat(args.num_samples, 1)
                        else:
                            obs_cond = obsgoal_cond.repeat(args.num_samples, 1, 1)

                        # initialize action from Gaussian noise
                        noisy_action = torch.randn(
                            (args.num_samples, model_params["len_traj_pred"], 2), device=device)
                        naction = noisy_action

                        # init scheduler
                        noise_scheduler.set_timesteps(num_diffusion_iters)

                        for k in noise_scheduler.timesteps[:]:
                            # predict noise
                            noise_pred = model(
                                'noise_pred_net',
                                sample=naction,
                                timestep=k,
                                global_cond=obs_cond
                            )
                            # inverse diffusion step (remove noise)
                            naction = noise_scheduler.step(
                                model_output=noise_pred,
                                timestep=k,
                                sample=naction
                            ).prev_sample

                    print("time elapsed:", time.time() - start_time)
                    naction = to_numpy(get_action(naction))
                    #sampled_actions_msg = Float32MultiArray()
                    #sampled_actions_msg.data = np.concatenate((np.array([0]), naction.flatten()))
                    #print("published sampled actions")
                    #sampled_actions_pub.publish(sampled_actions_msg)
                    naction = naction[0] 
                    chosen_waypoint = naction[args.waypoint]

                # ViNT
                else: 
                    #rospy.loginfo("Received crop!")
                    transf_goal_img = transform_images(pil_crop, model_params["image_size"]).to(device)
                    transf_obs_img = transform_images(context_queue, model_params["image_size"]).to(device)
                    start_time = time.time()
                    dist, waypoint = model(transf_obs_img, transf_goal_img) 
                    dist = to_numpy(dist[0])
                    dist_queue.append(dist[0])
                    waypoint = to_numpy(waypoint[0])
                    chosen_waypoint = waypoint[args.waypoint]
                    print("time elapsed:", time.time() - start_time)

                if model_params["normalize"]:
                    chosen_waypoint[:2] *= (MAX_V / RATE)
                waypoint_msg = Float32MultiArray()
                print(chosen_waypoint, dist)
                # TODO: Should we separately scale chosen waypoint with MAX_V and MAX_W?

                waypoint_msg.data = chosen_waypoint
                waypoint_pub.publish(waypoint_msg)

        # Check if we can stop
        if args.auto_stop and len(dist_queue) > 1:
            min_dist = dist_queue[-1] if min_dist is None else min(min_dist, dist_queue[-1])
            deltas = min_dist - np.array(list(dist_queue))
            debug_dist_pub.publish(dist_queue[-1])
            debug_avg_dist_pub.publish(np.mean(np.array(list(dist_queue))))
            debug_min_dist_pub.publish(min_dist)

        rate.sleep()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        "-m",
        default="vint",
        type=str,
        help="model name (hint: check ../config/models.yaml) (default: large_gnm)",
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
        "--num-samples",
        "-n",
        default=1,
        type=int,
        help=f"Number of actions sampled from the exploration model (default: 8)",
    )
    parser.add_argument(
        "--only_moving_contexts",
        default=False,
        type=bool,
        help=f"""If true, the context will be the last contiguous sequence of
        images where the robot's velocity is non-zero. Otherwise it is the most
        recent sequence of images."""
    )
    parser.add_argument(
        "--sim",
        default=False,
        type=bool,
        help="Running in iGibson simulator"
    )
    parser.add_argument(
        "--auto_stop",
        default=True,
        type=bool,
        help="If true, terminate when distance is small enough, otherwise keep going."
    )
    args = parser.parse_args()
    main(args)
