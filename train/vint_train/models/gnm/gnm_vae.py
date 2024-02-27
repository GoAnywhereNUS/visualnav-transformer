import torch
import torch.nn as nn
import torch.nn.functional as F

import os
import sys
import tty
import select
import termios
import itertools
import numpy as np

from filterpy.kalman import KalmanFilter
from collections import deque
from cam_utils import ActivationsAndGradients, GradCAM, show_cam_on_image
from torchvision.transforms import Compose, Normalize, ToTensor, ToPILImage, Resize

from typing import List, Dict, Optional, Tuple
from vint_train.models.gnm.modified_mobilenetv2 import MobileNetEncoder
from vint_train.models.base_model import BaseModel


class GNM_VAE(BaseModel):
    def __init__(
        self,
        context_size: int = 5,
        len_traj_pred: Optional[int] = 5,
        learn_angle: Optional[bool] = True,
        obs_encoding_size: Optional[int] = 1024,
        goal_encoding_size: Optional[int] = 1024,
    ) -> None:
        """
        GNM main class
        Args:
            context_size (int): how many previous observations to used for context
            len_traj_pred (int): how many waypoints to predict in the future
            learn_angle (bool): whether to predict the yaw of the robot
            obs_encoding_size (int): size of the encoding of the observation images
            goal_encoding_size (int): size of the encoding of the goal images
        """
        super(GNM_VAE, self).__init__(context_size, len_traj_pred, learn_angle)
        mobilenet = MobileNetEncoder(num_images=1 + self.context_size)
        self.obs_mobilenet = mobilenet.features
        self.obs_encoding_size = obs_encoding_size
        self.compress_observation = nn.Sequential(
            nn.Linear(mobilenet.last_channel, self.obs_encoding_size),
            nn.ReLU(),
        )
        stacked_mobilenet = MobileNetEncoder(
            num_images=2 + self.context_size
        )  # stack the goal and the current observation
        self.goal_mobilenet = stacked_mobilenet.features
        self.goal_encoding_size = goal_encoding_size
        self.compress_goal = nn.Sequential(
            nn.Linear(stacked_mobilenet.last_channel, 1024),
            nn.ReLU(),
            nn.Linear(1024, self.goal_encoding_size),
            nn.ReLU(),
        )
        self.mu = nn.Linear(self.goal_encoding_size + self.obs_encoding_size, 1024)
        self.logvar = nn.Linear(self.goal_encoding_size + self.obs_encoding_size, 1024)

        self.linear_layers = nn.Sequential(
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Linear(256, 32),
            nn.ReLU(),
        )
        self.dist_predictor = nn.Sequential(
            nn.Linear(32, 1),
        )
        self.action_predictor = nn.Sequential(
            nn.Linear(32, self.len_trajectory_pred * self.num_action_params),
        )

    def forward(
        self, obs_img: torch.tensor, goal_img: torch.tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        obs_encoding = self.obs_mobilenet(obs_img)
        obs_encoding = self.flatten(obs_encoding)
        obs_encoding = self.compress_observation(obs_encoding)

        obs_goal_input = torch.cat([obs_img, goal_img], dim=1)
        goal_encoding = self.goal_mobilenet(obs_goal_input)
        goal_encoding = self.flatten(goal_encoding)
        goal_encoding = self.compress_goal(goal_encoding)

        z = torch.cat([obs_encoding, goal_encoding], dim=1)

        mu = self.mu(z)
        logvar = self.logvar(z)
        z = self._latent_sample(mu, logvar)

        z = self.linear_layers(z)
        dist_pred = self.dist_predictor(z)
        action_pred = self.action_predictor(z)

        # augment outputs to match labels size-wise
        action_pred = action_pred.reshape(
            (action_pred.shape[0], self.len_trajectory_pred, self.num_action_params)
        )
        action_pred[:, :, :2] = torch.cumsum(
            action_pred[:, :, :2], dim=1
        )  # convert position deltas into waypoints
        if self.learn_angle:
            action_pred[:, :, 2:] = F.normalize(
                action_pred[:, :, 2:].clone(), dim=-1
            )  # normalize the angle prediction
        return dist_pred, action_pred, mu, logvar

    def _latent_sample(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = torch.empty_like(std).normal_()
            return eps.mul(std).add_(mu)
        else:
            return mu
        
class kl_divergence(torch.nn.Module):
    def __init__(self, beta=1):
        super(kl_divergence, self).__init__()
        pass

    def forward(self, mu, logvar):
        # recon_x, x: bs, t, 2
        # mu, logvar: bs*t, latent_size
        kldivergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
        return kldivergence
    

class PD_controller:
    def __init__(self, kp, kd):
        self.kp = kp
        self.kd = kd
        self.previous_error = 0
    def __call__(self, error):
        derivative = error - self.previous_error
        output = self.kp * error + self.kd * derivative
        self.previous_error = error
        # constrain output
        if output > 0.5:
           output = 0.5
        if output < -0.5:
           output = -0.5
        return output
    

class GNM_VAE_Inference(GNM_VAE):
    def __init__(
        self,
        context_size: int = 5,
        len_traj_pred: Optional[int] = 5,
        learn_angle: Optional[bool] = True,
        obs_encoding_size: Optional[int] = 1024,
        goal_encoding_size: Optional[int] = 1024,
    ) -> None:
        
        super().__init__(
            context_size, 
            len_traj_pred, 
            learn_angle, 
            obs_encoding_size, 
            goal_encoding_size
        )

        #######################################################################################
        ####### Initialise FaRe anomaly detector and localiser, and recovery strategies #######
        #######################################################################################
        self.DEFAULT_OUTPUT = (0.0, 0.0)
        self.DEFAULT_KL = 0
        self.W_RECOVER = 0.3

        self.kl_divergence = kl_divergence()
        self.last_pred = self.DEFAULT_OUTPUT
        self.last_kl = self.DEFAULT_KL
        self.ticks_count = 0
        self.last_intent = None
        self.gap_count = -1

        self.my_filter = KalmanFilter(dim_x=1, dim_z=1)
        self.init_filter = True
        self.window_length = 40
        self.threshold = 10

        self.detect_flag = False
        self.filtered_kl_cache = deque(maxlen=self.window_length)  #save filtered kl
        self.kl_cache = deque(maxlen=self.window_length)

        self.kl_base = 0
        self.action_length = 20
        self.action_cache = deque(maxlen=self.action_length)
        self.backtrack = True
        self.backtrack_step = -1
        self.rotate = True
        self.rotate_step = -1
        self.pd_controller = PD_controller(kp=0.002, kd=0.002)
        self.kl_rotate_last = 0
        self.kl_rotate_cache = []
        # arbitrary left
        self.w_recover = self.W_RECOVER
        
        self.recovery_step = -1
        self.yaw_base = 0

        self.target_layers_left = [self.model.view_models[0].conv_layers[2],
                          #self.model.view_models[0].conv_layers[1],
                          #self.model.view_models[0].conv_layers[0],
                        ]
        self.target_layers_mid = [self.model.view_models[1].conv_layers[2],
                          #self.model.view_models[1].conv_layers[1],
                          #self.model.view_models[1].conv_layers[0],
                        ]
        self.target_layers_right = [self.model.view_models[2].conv_layers[2],
                          #self.model.view_models[2].conv_layers[1],
                          #self.model.view_models[2].conv_layers[0],
                        ]
        self.activations_and_grads = ActivationsAndGradients(self.model, self.target_layers_left, self.target_layers_mid, self.target_layers_right, None)
        self.cam = GradCAM(None, target_size=(112, 112))
        self.left_raw, self.mid_raw, self.right_raw = 0, 0, 0
        self.resize_raw = Compose([ToPILImage(), Resize((85, 64))])
        
        self.recovery_action = self.DEFAULT_OUTPUT
        self.backtrack_keep = False
        self.switch_threshold = 1500
        self.max_rotate_step = 60
        self.first_pd = True
        self.ask_for_help = False

    def forward(
        self, obs_img: torch.tensor, goal_img: torch.tensor, yaw: float
    ):
        # TODO: Check if this should be done here?
        self.gap_count += 1

        key = self._getKey()
        if key == ' ':
            self._init_detection()
            print("reset detection")

        # recovery mode      
        if self.detect_flag:
            self.recovery_step += 1
            if self.recovery_step == 0:
                d_ = itertools.islice(self.kl_cache, 0, 10)
                self.kl_base = sum(d_) / 10
                self.yaw_base = yaw
            left, mid, right = self._split_image(obs_img)
            self.left_raw, self.mid_raw, self.right_raw = self.resize_raw(left), self.resize_raw(mid), self.resize_raw(right)
            (v, w), kl, filtered_kl, visualization, ask_for_help = self._recovery_policy(left, mid, right, yaw)
            return (v, w), kl, filtered_kl, visualization, ask_for_help
        
        # normal mode
        # inference
        with torch.no_grad():
            dist_pred, action_pred, mu, logvar = super().forward(obs_img, goal_img)
            kl = self.kl_divergence(mu, logvar)

        # check gap
        if self.gap_count % self.interval == 0:
            self.last_pred = (dist_pred, action_pred)
            self.last_kl = kl

        filtered_kl = self._filter_kl(kl)
        return dist_pred, action_pred, self.last_kl.item(), filtered_kl, None, False

    def _split_image(self, obs_img):
        return torch.chunk(obs_img, 3, dim=2)

    def _recovery_policy(self, left, mid, right, yaw):
        print("Anomaly Detected!")
        print("Base value" + str(self.kl_base))
        self._init_states()
        left, mid, right = left.unsqueeze(1).cuda(), mid.unsqueeze(1).cuda(), right.unsqueeze(1).cuda()

        out, mu, logvar = self.activations_and_grads(left, mid, right) # TODO: Check if this is the correct way to call
        self.model.zero_grad()
        kl = self.kl_divergence(mu, logvar)
        gradcam_loss = torch.mean(mu)
        #gradcam_loss = kl
        gradcam_loss.backward(retain_graph=True)
        grayscale_cam_left, grayscale_cam_mid, grayscale_cam_right = self.cam(self.activations_and_grads)
        grayscale_cam_left, grayscale_cam_mid, grayscale_cam_right = grayscale_cam_left[0, :], grayscale_cam_mid[0, :], grayscale_cam_right[0, :]

        vis_left, count_left = show_cam_on_image(np.asarray(self.left_raw) / 255, grayscale_cam_left, use_rgb=True)
        vis_mid, count_mid = show_cam_on_image(np.asarray(self.mid_raw) / 255, grayscale_cam_mid, use_rgb=True)
        vis_right, count_right = show_cam_on_image(np.asarray(self.right_raw) / 255, grayscale_cam_right, use_rgb=True)
        visualization = np.concatenate([vis_left, vis_mid, vis_right], axis=1)

        if count_left > self.switch_threshold and count_mid > self.switch_threshold and count_right > self.switch_threshold:
            action_mode = 'backtrack'
            self.rotate = False
            if not self.backtrack_keep:
                self.backtrack_step += 1
            # Maximum step, fail
            if self.backtrack_step >= self.action_length:
                self.backtrack = False
            else:
                self.backtrack = True
        else:
            self.backtrack = False
            self.rotate_step += 1
            # Maximum step, fail
            if self.rotate_step > self.max_rotate_step:
                self.rotate = False
            else:
                self.rotate = True

            if count_left <= count_mid and count_left <= count_right:
                action_mode = 'left'
            elif count_mid <= count_left and count_mid <= count_right:
                action_mode = 'mid'
            elif count_right <= count_left and count_right <= count_mid:
                action_mode = 'right'

        print(action_mode)
        print(count_left, count_mid, count_right)
        
        if self.backtrack_keep:
            action_mode = 'backtrack'
            self.rotate = False
            self.backtrack = True

        if action_mode != 'mid':
            self.first_pd = True
       
        kl = kl.item()
        # backtrack
        if self.backtrack:
            # Success
            if kl < self.kl_base:
                self._init_detection()
                print("Recover by backtracking")
                self.recovery_action = (0, 0)
            else:
                # back to base_yaw then backtrack
                dist_yaw = yaw - self.yaw_base
                if dist_yaw > 180:
                    dist_yaw -= 360
                if dist_yaw < -180:
                    dist_yaw += 360

                if abs(dist_yaw) > 10:
                    self.backtrack_keep = True
                    if dist_yaw > 0:
                        self.recovery_action = (0, -1 * self.W_RECOVER)
                    else:
                        self.recovery_action = (0, self.W_RECOVER)
                else:
                    # Backtrack policy
                    self.recovery_action = (-1 * self.action_cache[-1-self.backtrack_step][0], -1 * self.action_cache[-1-self.backtrack_step][1])
                    self.yaw_base = yaw
                    self.backtrack_keep = False
        # rotate
        elif self.rotate:
            # Success
            if kl < self.kl_base:
                self._init_detection()
                print("Recover by rotation")
                self.recovery_action = (0, 0)
            else:
                # constrain yaw deviation
                dist_yaw = yaw - self.yaw_base
                if dist_yaw > 180:
                    dist_yaw -= 360
                if dist_yaw < -180:
                    dist_yaw += 360

                if abs(dist_yaw) > 30:
                    if dist_yaw > 0:
                        self.recovery_action = (0, -1 * self.W_RECOVER)
                    else:
                        self.recovery_action = (0, self.W_RECOVER)
                else:
                    if action_mode == 'left':
                        self.recovery_action = (0, self.W_RECOVER)
                    elif action_mode == 'mid':
                        # pd control
                        if self.first_pd:
                            self.kl_rotate_last = kl
                            self.w_recover = self.W_RECOVER
                            self.recovery_action = (0, self.w_recover)
                            self.first_pd = False
                        else:
                            error = kl - self.kl_rotate_last
                            output = self.pd_controller(error)
                            #print("PD control, error = " + str(error) + ", output = " + str(output))
                            self.kl_rotate_last = kl
                            if self.w_recover < 0:
                                self.w_recover = output
                            else:
                                self.w_recover = -1 * output
                            self.recovery_action = (0, self.w_recover)
                    elif action_mode == 'right':
                        self.recovery_action = (0, -1 * self.W_RECOVER)
        else:
            print("Asking for Help!!!!!!!")
            #self._init_detection()
            self.recovery_action = (0, 0)
            self.ask_for_help = True

        return self.recovery_action, kl, 0, visualization, self.ask_for_help
            
    def _init_detection(self):
        self.ask_for_help = False
        self.recovery_action = self.DEFAULT_OUTPUT
        self.detect_flag = False
        self.recovery_step = -1
        self.kl_base = 0
        self.yaw_base = 0
        self.backtrack = True
        self.backtrack_step = -1
        self.rotate = True
        self.rotate_step = -1
        self.w_recover = self.W_RECOVER
        self.kl_rotate_last = 0
        self.kl_rotate_cache = []
        self.kl_cache = deque(maxlen=self.window_length)
        self.filtered_kl_cache = deque(maxlen=self.window_length)
        self.action_cache = deque(maxlen=self.action_length)

    def _init_states(self):
        self.model.reset_states()
        self.ticks_count = 0
        print(f'states initialized and ticks count reset')

    def _init_kalman(self, x):
        self.my_filter = KalmanFilter(dim_x=1, dim_z=1)
        self.my_filter.x = x
        self.my_filter.F = np.eye(1)
        self.my_filter.H = np.eye(1)
        self.my_filter.P = 1 * np.eye(1)
        self.my_filter.R = 100 * np.eye(1)     # observation noise
        self.my_filter.Q = 0.01 * np.eye(1)    # process noise

    def _filter_kl(self, kl):
        kl = kl.cpu().numpy()   
        if self.init_filter:
            self._init_kalman(kl)
            self.init_filter = False
            self.filtered_kl_cache = deque(maxlen=self.window_length)
            self.filtered_kl_cache.append(kl)
            return kl.item()

        self.my_filter.predict()
        self.my_filter.update(kl)
        x = self.my_filter.x
        self.filtered_kl_cache.append(x)
        self.kl_cache.append(kl.item())

        accum_gradient = 0
        if not self.detect_flag:
            if len(self.filtered_kl_cache) == self.window_length:
                for i in range(self.window_length-1):
                    gradient = self.filtered_kl_cache[-1-i] - self.filtered_kl_cache[-1-i-1]
                    accum_gradient += gradient
            if accum_gradient > self.threshold:
                #if self.last_intent == 0:
                self.detect_flag = True
        return x.item()

    def _getKey(self):
        settings = termios.tcgetattr(sys.stdin)
        tty.setraw(sys.stdin.fileno())
        rlist, _, _ = select.select([sys.stdin], [], [], 0.1)
        if rlist:
            key = sys.stdin.read(1)
        else:
            key = ''

        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
        return key