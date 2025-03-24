import os

import torch
from torchvision.transforms import Compose, Normalize, ToTensor, ToPILImage, Resize
from .cam_utils import ActivationsAndGradients, GradCAM, show_cam_on_image

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

norm = lambda in_c: nn.GroupNorm(num_groups=32, num_channels=in_c)

import torch
import torch.nn as nn
import math
from filterpy.kalman import KalmanFilter
from collections import deque
import itertools
import sys, select, os

if os.name == 'nt':
    import msvcrt, time
else:
    import tty, termios
import numpy as np
import cv2


class kl_divergence(torch.nn.Module):
    def __init__(self, beta=1):
        super(kl_divergence, self).__init__()
        pass

    def forward(self, mu, logvar):
        # recon_x, x: bs, t, 2
        # mu, logvar: bs*t, latent_size
        kldivergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
        return kldivergence


class PD_controller():
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


class GNM_VAE(object):
    BATCH_SIZE = 1
    DEFAULT_OUTPUT = (0.0, 0.0)
    DEFAULT_KL = 0
    W_RECOVER = 0.3

    def __init__(self, model):
        self.model = model
        self.kl_divergence = kl_divergence()
        self.target_layers = [model.obs_mobilenet[-1], model.obs_mobilenet[-2], model.obs_mobilenet[-3], ]
        self.activations_and_grads = ActivationsAndGradients(model, self.target_layers, None)
        self.cam = GradCAM(None, target_size=(85, 64))

        self.my_filter = KalmanFilter(dim_x=1, dim_z=1)
        self.init_filter = True
        # 40, 15
        self.window_length = 40
        self.threshold = 10

        self.detect_flag = False
        self.filtered_kl_cache = deque(maxlen=self.window_length)  # save filtered kl
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

        # test sensor failure only #
        self.total_step = 0
        ############################

        self.recovery_action = self.DEFAULT_OUTPUT
        self.backtrack_keep = False
        self.switch_threshold = 1500
        self.max_rotate_step = 60
        self.first_pd = True
        self.ask_for_help = False

    def __call__(self, batch_obs_imgs, batch_goal_data, yaw):
        ## simulate sensor failure
        # self.total_step += 1
        # #if self.ticks_count % self.cycle >= self.cycle * (1 - self.failure_rate):
        # if self.total_step > 50:
        #     #left = torch.zeros_like(left)
        #     mid = torch.zeros_like(mid)
        #     right = torch.zeros_like(right)
        #     print(f'===> sensor failure. (simulation) ')
        #
        # inference
        return self.infer_from_full_view(batch_obs_imgs, batch_goal_data, yaw)



    def infer_from_full_view(self, batch_obs_imgs, batch_goal_data, yaw, image_vis):
        key = self._getKey()
        if key == ' ':
            self._init_detection()
            print("reset detection")
            # self.detect_flag = True
            # print("Trigger Anomaly!!")

        # recovery mode
        if self.detect_flag:
            self.recovery_step += 1
            if self.recovery_step == 0:
                d_ = itertools.islice(self.kl_cache, 0, 10)
                self.kl_base = sum(d_) / 10

                # !!!!!!!!!!!!!!!!!!! remove !!!!!!!!!!!!!!!!!!!!
                # self.kl_base = 0
                # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                self.yaw_base = yaw

            (v, w), kl, filtered_kl, visualization, ask_for_help = self._recovery_policy(batch_obs_imgs, batch_goal_data, yaw, image_vis)
            return (v, w), kl, filtered_kl, visualization, ask_for_help

        # inference
        with torch.no_grad():
            distances, waypoints, mu, logvar = self.model(batch_obs_imgs, batch_goal_data)
            kl = self.kl_divergence(mu, logvar)

        filtered_kl = self._filter_kl(kl)
        self.action_cache.append((1, 0))
        return distances, waypoints, kl.item(), filtered_kl

    def _recovery_policy(self, batch_obs_imgs, batch_goal_data, yaw, image_vis):
        print("Anomaly Detected!")
        print("Base value" + str(self.kl_base))
        distances, waypoints, mu, logvar = self.activations_and_grads(batch_obs_imgs, batch_goal_data)
        self.model.zero_grad()
        kl = self.kl_divergence(mu, logvar)
        # gradcam_loss = torch.mean(mu)
        gradcam_loss = kl
        gradcam_loss.backward(retain_graph=True)
        grayscale_cam = self.cam(self.activations_and_grads)
        grayscale_cam = grayscale_cam[0, :]

        raw_img = image_vis
        raw_img = raw_img.resize((85, 64))
        vis, count_left, count_mid, count_right = show_cam_on_image(np.asarray(raw_img) / 255, grayscale_cam,
                                                                    use_rgb=True)
        cv2.putText(vis, str(count_left) + "   " + str(count_mid) + "    " + str(count_right), (5, 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255))
        cv2.imshow('localisation', vis)


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

            # if count_left <= count_right:
            #    action_mode = 'left'
            # else:
            #    action_mode = 'right'

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
                    self.recovery_action = (-1 * self.action_cache[-1 - self.backtrack_step][0],
                                            -1 * self.action_cache[-1 - self.backtrack_step][1])
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
                            # print("PD control, error = " + str(error) + ", output = " + str(output))
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
            # self._init_detection()
            self.recovery_action = (0, 0)
            self.ask_for_help = True

        return self.recovery_action, kl, 0, None, self.ask_for_help

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

    def _init_kalman(self, x):
        self.my_filter = KalmanFilter(dim_x=1, dim_z=1)
        self.my_filter.x = x
        self.my_filter.F = np.eye(1)
        self.my_filter.H = np.eye(1)
        self.my_filter.P = 1 * np.eye(1)
        self.my_filter.R = 100 * np.eye(1)  # observation noise
        self.my_filter.Q = 0.01 * np.eye(1)  # process noise

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
                accum_gradient = self.filtered_kl_cache[-1] - self.filtered_kl_cache[0]
            if accum_gradient > self.threshold:
                self.detect_flag = True
        return x.item()

    def _getKey(self):
        if os.name != 'nt':
            settings = termios.tcgetattr(sys.stdin)
        if os.name == 'nt':
            timeout = 0.1
            startTime = time.time()
            while (1):
                if msvcrt.kbhit():
                    if sys.version_info[0] >= 3:
                        return msvcrt.getch().decode()
                    else:
                        return msvcrt.getch()
                elif time.time() - startTime > timeout:
                    return ''

        tty.setraw(sys.stdin.fileno())
        rlist, _, _ = select.select([sys.stdin], [], [], 0.1)
        if rlist:
            key = sys.stdin.read(1)
        else:
            key = ''

        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
        return key


