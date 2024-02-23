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


def convbn(in_channels, out_channels, kernel_size, stride, padding, bias):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )


def convgn(in_channels, out_channels, kernel_size, stride, padding, bias):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
        nn.GroupNorm(num_groups=32, num_channels=out_channels),
        nn.ReLU(inplace=True)
    )


class ConvLSTMCell(nn.Module):
    def __init__(self, in_size, in_c, hid_c, dropout_keep, kernel_size=3):
        super().__init__()
        self.height, self.width = in_size
        self.in_size = in_size
        self.in_c = in_c
        self.hid_c = hid_c
        self.conv = nn.Sequential(
            nn.Conv2d(self.in_c + self.hid_c, 4 * self.hid_c, kernel_size, 1, padding=kernel_size // 2),
            norm(4 * self.hid_c)
        )
        self.dropout = nn.Dropout2d(p=1 - dropout_keep)  # dp2d instead of dropout for 2d feature map (Ma Xiao)
        print(f'ConvLSTMCell0 in_size = {in_size}, dropout {dropout_keep}')

    def forward(self, ins, seq_len, prev=None):
        if prev is None:
            h, c = self.init_states(ins.size(0))  # ins and prev not None at the same time (guaranteed)
        else:
            h, c = prev

        hs, cs = [], []  # store all intermediate h and c
        for i in range(seq_len):
            # prepare x: create one zero tensor if x is None (decoder mode)
            if ins is not None:
                x = ins[:, i]
            else:
                x = torch.zeros(h.size(0), self.in_c, self.height, self.width).cuda()

            combined = self.dropout(torch.cat([x, h], dim=1))
            combined_conv = self.conv(combined)

            i_t, f_t, o_t, g_t = torch.split(combined_conv, self.hid_c, dim=1)

            i_t = torch.sigmoid(i_t)
            f_t = torch.sigmoid(f_t)
            o_t = torch.sigmoid(o_t)
            g_t = torch.tanh(g_t)

            c_t = f_t * c + i_t * self.dropout(g_t)
            h_t = o_t * torch.tanh(c_t)

            h, c = h_t, c_t

            hs.append(h)
            # cs.append(c)

        return hs, cs

    def init_states(self, batch_size):
        states = (torch.zeros(batch_size, self.hid_c, self.height, self.width),
                  torch.zeros(batch_size, self.hid_c, self.height, self.width))
        states = (states[0].cuda(), states[1].cuda())
        return states


class ConvLSTMCellPeep(nn.Module):
    def __init__(self, in_size, in_c, hid_c, dropout_keep, kernel_size=3):
        super().__init__()
        self.height, self.width = in_size
        self.in_size = in_size
        self.in_c = in_c
        self.hid_c = hid_c
        self.conv1 = nn.Sequential(
            nn.Conv2d(self.in_c + self.hid_c * 2, 2 * self.hid_c, kernel_size, 1, padding=kernel_size // 2),
            norm(2 * self.hid_c)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(self.in_c + self.hid_c * 2, self.hid_c, kernel_size, 1, padding=kernel_size // 2),
            norm(self.hid_c)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(self.in_c + self.hid_c, self.hid_c, kernel_size, 1, padding=kernel_size // 2),
            norm(self.hid_c)
        )
        self.dropout = nn.Dropout2d(p=1 - dropout_keep)  # dp2d instead of dropout for 2d feature map (Ma Xiao)
        print(f'ConvLSTMCell with peephole in_size = {in_size}, dropout {dropout_keep}')

    def forward(self, ins, seq_len, prev=None):
        if prev is None:
            h, c = self.init_states(ins.size(0))  # ins and prev not None at the same time (guaranteed)
        else:
            h, c = prev

        hs, cs = [], []  # store all intermediate h and c
        for i in range(seq_len):
            # prepare x: create one zero tensor if x is None (decoder mode)
            if ins is not None:
                x = ins[:, i]
            else:
                x = torch.zeros(h.size(0), self.in_c, self.height, self.width).cuda()

            # f, i gates
            combined_conv = self.conv1(self.dropout(torch.cat([x, h, c], dim=1)))
            i_t, f_t = torch.split(combined_conv, self.hid_c, dim=1)
            i_t = torch.sigmoid(i_t)
            f_t = torch.sigmoid(f_t)

            # g gate
            g_t = self.conv3(torch.cat([x, h], dim=1))
            g_t = torch.tanh(g_t)

            # update cell state
            c_t = f_t * c + i_t * self.dropout(g_t)

            # o gate
            o_t = self.conv2(torch.cat([x, h, c_t], dim=1))
            o_t = torch.sigmoid(o_t)

            h_t = o_t * torch.tanh(c_t)

            h, c = h_t, c_t

            hs.append(h)
            cs.append(c)

        hs = torch.stack(hs, dim=1)  # stack along t dim
        return hs, cs

    def init_states(self, batch_size):
        states = (torch.zeros(batch_size, self.hid_c, self.height, self.width),
                  torch.zeros(batch_size, self.hid_c, self.height, self.width))
        states = (states[0].cuda(), states[1].cuda())
        return states


class BottleneckLSTMCell(nn.Module):
    def __init__(self, in_size, in_c, hid_c, dropout_keep, peephole):
        super().__init__()
        self.hid_c = hid_c

        cell = ConvLSTMCellPeep if peephole else ConvLSTMCell
        self.cell = cell(in_size, in_c, hid_c, dropout_keep)

    def forward(self, x):
        bs, t, c, h, w = x.shape
        x, cs = self.cell(x, t, prev=self.cell_state)  # prev could be None
        self.cell_state = (x[-1], cs[-1])  # only need the last dim
        return x

    def reset_states(self):
        self.cell_state = None


class SimpleConvNet(nn.Module):
    CHANNELS = [3, 64, 256, 512, 1024]

    def __init__(self, channels):
        super().__init__()
        layer1 = nn.Sequential(
            convbn(self.CHANNELS[0], self.CHANNELS[1], kernel_size=7, stride=2, padding=3, bias=False),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            convbn(self.CHANNELS[1], channels[0], kernel_size=3, stride=1, padding=1, bias=True)
        )
        layer2 = convbn(channels[0], channels[1], kernel_size=3, stride=2, padding=1, bias=True)
        layer3 = convbn(channels[1], channels[2], kernel_size=3, stride=2, padding=1, bias=True)
        self.layers = nn.ModuleList([layer1, layer2, layer3])

    def forward(self, x):
        return self.layers(x)


class OneViewModel(nn.Module):
    NUM_INTENTION = 4

    def __init__(self, spatial_size, channels, lstm_dropout_keep, peephole, skip, sep_lstm, skip_depth,
                 num_intention, stack_deepest):
        super().__init__()
        self.spatial_size = spatial_size
        self.skip = skip
        self.sep_lstm = sep_lstm
        self.skip_depth = skip_depth
        self.stack_deepest = stack_deepest
        if skip:
            self.dropout2d = nn.Dropout2d(p=1 - lstm_dropout_keep)

        if num_intention is not None:
            self.NUM_INTENTION = num_intention

        sizes = [(int(math.ceil(spatial_size[0] / 2 / (2 ** i))),
                  int(math.ceil(spatial_size[1] / 2 / (2 ** i)))) for i in range(1, 4)]

        self.conv_layers = self._get_conv_layers(channels)
        lstms = []
        for i in range(3):
            if i in self.skip_depth or stack_deepest:
                continue
            lstm_branches = []
            for j in range(self.NUM_INTENTION if sep_lstm else 1):
                cell = BottleneckLSTMCell(sizes[i], channels[i], channels[i], lstm_dropout_keep, peephole)
                lstm_branches.append(cell)
            lstms.append(nn.ModuleList(lstm_branches))
        self.lstms = nn.ModuleList(lstms)

        # if ConvLSTMs stacked at the deepest layer
        lstms = []
        if stack_deepest:
            lstm_branches = []
            for i in range(3):
                for j in range(self.NUM_INTENTION if sep_lstm else 1):
                    cell = BottleneckLSTMCell(sizes[-1], channels[-1],
                                              channels[-1], lstm_dropout_keep, peephole)
                    lstm_branches.append(cell)
                lstms.append(nn.ModuleList(lstm_branches))
            self.lstms = nn.ModuleList(lstms)

        self.pool_size = (2, 2)
        self.pool = nn.AdaptiveAvgPool2d(self.pool_size)

    def _get_conv_layers(self, channels):
        return SimpleConvNet(channels).layers

    def forward(self, x, idx):
        idx = idx if self.sep_lstm else 0
        convlstm_count = 0

        for i in range(3):
            bs, t, c, h, w = x.shape
            x = x.view(bs * t, c, h, w)
            x = self.conv_layers[i](x)
            _, c, h, w = x.shape
            x = x.view(bs, t, c, h, w)
            if i in self.skip_depth or self.stack_deepest:
                continue
            # x_2 = self.lstms[i - len(self.skip_depth)][idx](x)
            x_2 = self.lstms[convlstm_count][idx](x)
            convlstm_count += 1
            x = (x + x_2) if self.skip else x_2
            if self.skip:
                x = x.view(bs * t, c, h, w)
                x = self.dropout2d(x)
                x = x.view(bs, t, c, h, w)

        if self.stack_deepest:
            for i in range(3 - len(self.skip_depth)):
                # x = self.lstms[i - len(self.skip_depth)][idx](x)
                x = self.lstms[i][idx](x)

        x = self.pool(x.view(bs * t, c, h, w)).flatten(1).view(bs, t, c * self.pool_size[0] * self.pool_size[1])

        return x

    def reset_states(self):
        for lstm_set in self.lstms:
            for lstm in lstm_set:
                lstm.reset_states()


class ConvLSTMNet(nn.Module):
    """
    Plug lstm_model into resnets
    """
    NUM_INTENTION = 4
    NUM_VIEWS = 3
    FC_DROPOUT_KEEP = 0.6
    LSTM_DROPOUT_KEEP = 0.7
    PEEPHOLE = True
    SKIP_CONNECTION = False

    def __init__(self, spatial_size, channels, sep_lstm, sep_fc, skip_depth, nviews, num_intention,
                 stack_deepest):
        super(ConvLSTMNet, self).__init__()
        self.sep_fc = sep_fc
        self.NUM_VIEWS = nviews

        if num_intention is not None:
            self.NUM_INTENTION = num_intention

        views = []
        for i in range(self.NUM_VIEWS):
            views.append(
                OneViewModel(spatial_size, channels, self.LSTM_DROPOUT_KEEP, self.PEEPHOLE, self.SKIP_CONNECTION,
                             sep_lstm, skip_depth, num_intention, stack_deepest)
            )
        self.view_models = nn.ModuleList(views)

        h, w = (2, 6)
        fc_in, fc_interm = channels[-1] * h * w, 2 * 32

        LATENT_SPACE_SIZE = 2048
        
        mu_latent = []
        logvar_latent = []
        for i in range(self.NUM_INTENTION if self.sep_fc else 1):
            mu_latent.append(
                nn.Sequential(
                    nn.Dropout(p=1 - self.FC_DROPOUT_KEEP),
                    nn.Linear(fc_in, LATENT_SPACE_SIZE)
                )
            )
            logvar_latent.append(
                nn.Sequential(
                    nn.Dropout(p=1 - self.FC_DROPOUT_KEEP),
                    nn.Linear(fc_in, LATENT_SPACE_SIZE)
                )
            )
        self.mu_latent = nn.ModuleList(mu_latent)
        self.logvar_latent = nn.ModuleList(logvar_latent)

        decoders = []
        for i in range(self.NUM_INTENTION if self.sep_fc else 1):
            decoders.append(
                nn.Sequential(
                    nn.Dropout(p=1 - self.FC_DROPOUT_KEEP),
                    nn.Linear(LATENT_SPACE_SIZE, fc_interm, bias=True),
                    nn.LeakyReLU(negative_slope=0.2, inplace=True),
                    nn.Dropout(p=1 - self.FC_DROPOUT_KEEP),
                    nn.Linear(fc_interm, 2, bias=True)
                )
            )
        self.decoders = nn.ModuleList(decoders)

        print(f'model: assume 3-view inputs (bs, t, c, h, w), all intentions in a batch are the same , '
              f'fc_dropout_keep {self.FC_DROPOUT_KEEP}, lstm_dropout_keep {self.LSTM_DROPOUT_KEEP}, '
              f'peephole {self.PEEPHOLE}, skip connection {self.SKIP_CONNECTION}, sep_lstm {sep_lstm}, '
              f'skip_depth {skip_depth}, sep_fc = {sep_fc}, channels {channels}, nviews {nviews}, '
              f'num of intentions {self.NUM_INTENTION}, stack cells at deepest {stack_deepest}'
              f'\nWarning: Remember to manually reset/detach cell states!')

    def forward(self, left, mid, right, intent):
        # assume input (bs, t, c, h, w)
        # assume all intents are the same
        idx = int(intent[0][0].item()) if self.sep_fc else 0
        views = [left, mid, right]

        feats = []
        for i, view in enumerate(views):
            x = self.view_models[i % self.NUM_VIEWS](view, idx)  # 0 and 2 share the same view
            feats.append(x)
        x = torch.cat(feats, dim=2)  # bs, t, c

        bs, t, c = x.shape
        x = x.view(bs * t, c)

        x_mu = self.mu_latent[idx](x)
        x_logvar = self.logvar_latent[idx](x)

        latent = self._latent_sample(x_mu, x_logvar)

        x = self.decoders[idx](latent)
        x = x.view(bs, t, -1)

        return x, x_mu, x_logvar

    def _latent_sample(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = torch.empty_like(std).normal_()
            return eps.mul(std).add_(mu)
        else:
            return mu

    @staticmethod
    def detach_states(states):
        for depth_states in states:
            for i, (h, c) in enumerate(depth_states):
                h, c = h.detach(), c.detach()
                h.requires_grad, c.requires_grad = True, True
                depth_states[i] = (h, c)
        return states

    @staticmethod
    def derive_grad(y, x):
        for depth_y, depth_x in zip(y, x):
            for (yh, yc), (xh, xc) in zip(depth_y, depth_x):
                yc.backward(xc.grad, retain_graph=True)

    def reset_states(self):
        for view_model in self.view_models:
            view_model.reset_states()
            # print(f'ConvLSTMNet reset states')


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
        

class ConvLSTMInference(object):
    SPATIAL_SIZE = (112, 112)
    INTENTION_MAPPING = {0: 'forward', 1: 'left', 2: 'right', 3: 'unknown'}
    BATCH_SIZE = 1
    DEFAULT_OUTPUT = (0.0, 0.0)
    DEFAULT_KL = 0
    W_RECOVER = 0.3

    def __init__(self, ckpt_path, cycle, interval, sep_lstm, sep_fc, skip_depth, nviews, num_intention, stack_deepest,
                 failure_rate):
        self.model = ConvLSTMNet((112, 112), [128, 192, 256], sep_lstm, sep_fc, skip_depth, nviews,
                                 num_intention, stack_deepest).cuda()
        self.kl_divergence = kl_divergence()
        self._load_model_dic(ckpt_path)
        self.to_tensor = Compose([
            ToPILImage(),
            Resize(self.SPATIAL_SIZE),
            ToTensor(),
            Normalize(mean=[0.5071, 0.4866, 0.4409], std=[0.2675, 0.2565, 0.2761]),
        ])
        self.model.eval()
        self.cycle = cycle
        self.interval = interval
        self.last_pred = self.DEFAULT_OUTPUT
        self.last_kl = self.DEFAULT_KL
        self.ticks_count = 0
        self.last_intent = None
        self.gap_count = -1
        self.failure_rate = failure_rate
        self.debug_count = 0
        import time
        self.debug_time = time.time()

        self.my_filter = KalmanFilter(dim_x=1, dim_z=1)
        self.init_filter = True
        # 40, 15
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

        # test sensor failure only #
        self.total_step = 0
        ############################

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
        self.resize_raw = Compose([ToPILImage(), Resize((112, 112))])
        
        self.recovery_action = self.DEFAULT_OUTPUT
        self.backtrack_keep = False
        self.switch_threshold = 1500
        self.max_rotate_step = 60
        self.first_pd = True
        self.ask_for_help = False
        print(f'ConvLSTM based on Simple-INet: loaded from {ckpt_path}, , cycle = {cycle}, interval = {interval},'
              f'simulated sensor failure rate = {self.failure_rate}')

    def _load_model_dic(self, ckpt_path):
        checkpoint = torch.load(ckpt_path)
        self.model.load_state_dict(checkpoint['model'], strict=True)
        print(f'Model loaded: {ckpt_path}')

    def _preprocess_input(self, left, mid, right, intention):
        # import cv2
        # left, mid, right = cv2.cvtColor(left, cv2.COLOR_BGR2RGB), cv2.cvtColor(mid, cv2.COLOR_BGR2RGB), cv2.cvtColor(right, cv2.COLOR_BGR2RGB)
        # cv2.imwrite(f'/data/debug_save_figs/left/{self.debug_count}.jpg', left)
        # cv2.imwrite(f'/data/debug_save_figs/mid/{self.debug_count}.jpg', mid)
        # cv2.imwrite(f'/data/debug_save_figs/right/{self.debug_count}.jpg', right)
        self.debug_count += 1

        # assume the input is a numpy array (cv2 image)
        assert 0 <= intention <= 3
        left, mid, right = self.to_tensor(left).unsqueeze(0), self.to_tensor(mid).unsqueeze(0), \
                           self.to_tensor(right).unsqueeze(0)

        intention = torch.tensor(intention).unsqueeze(0)
        return left, mid, right, intention

    def __call__(self, left, mid, right, intention, yaw):
        self.gap_count += 1

        self.left_raw, self.mid_raw, self.right_raw = self.resize_raw(left), self.resize_raw(mid), self.resize_raw(right)
        left, mid, right, intention = self._preprocess_input(left, mid, right, intention)
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
        return self.infer_from_full_view(left, mid, right, intention, yaw)

    def _tick(self):
        if self.ticks_count % self.cycle == 0:
            self._init_states()
            print(f'cycle count reached')
        self.ticks_count += 1

    def infer_from_full_view(self, left, mid, right, intention, yaw):
        # check states
        self._tick()
        # print(f'history length count {self.ticks_count}')
        assert left.shape == mid.shape == right.shape == (1, 3,) + self.SPATIAL_SIZE, \
            f'visual data shape {left.shape} incorrect'\
        
        key = self._getKey()
        if key == ' ':
            self._init_detection()
            print("reset detection")
            #self.detect_flag = True
            #print("Trigger Anomaly!!")
            
        # recovery mode      
        if self.detect_flag:
            self.recovery_step += 1
            if self.recovery_step == 0:
                d_ = itertools.islice(self.kl_cache, 0, 10)
                self.kl_base = sum(d_) / 10
                
                #!!!!!!!!!!!!!!!!!!! remove !!!!!!!!!!!!!!!!!!!!
                #self.kl_base = 0
                # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

                self.yaw_base = yaw
            (v, w), kl, filtered_kl, visualization, ask_for_help = self._recovery_policy(left, mid, right, intention, yaw)
            return (v, w), kl, filtered_kl, visualization, ask_for_help

        # normal mode
        # check intention
        if intention != self.last_intent:
            # print(f'intention different')
            self.last_intent = intention
            self._init_states()
            self.init_filter = True

        # inference
        with torch.no_grad():
            left, mid, right, intention = left.unsqueeze(1).cuda(), mid.unsqueeze(1).cuda(), \
                                          right.unsqueeze(1).cuda(), intention.unsqueeze(1).cuda()
            out, mu, logvar = self.model(left, mid, right, intention)
            kl = self.kl_divergence(mu, logvar)

        # print(f'intention {intention.item()}, ticks count {self.ticks_count}, pred', (out[0][0][0].item(), out[0][0][1].item()))
        pred = (torch.tensor(0) if out[0][0][0] < 0.08 else out[0][0][0], out[0][0][1])
        # check gap
        if self.gap_count % self.interval != 0:
            pass
        else:
            self.last_pred = pred
            self.last_kl = kl

        filtered_kl = self._filter_kl(kl)
        self.action_cache.append((self.last_pred[0].item(), self.last_pred[1].item()))
        return (self.last_pred[0].item(), self.last_pred[1].item()), self.last_kl.item(), filtered_kl, None, False
   
    def _recovery_policy(self, left, mid, right, intention, yaw):
        print("Anomaly Detected!")
        print("Base value" + str(self.kl_base))
        self._init_states()
        left, mid, right, intention = left.unsqueeze(1).cuda(), mid.unsqueeze(1).cuda(), right.unsqueeze(1).cuda(), intention.unsqueeze(1).cuda()

        out, mu, logvar = self.activations_and_grads(left, mid, right, intention)
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
            
            #if count_left <= count_right:
            #    action_mode = 'left'
            #else:
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
        if os.name != 'nt':
            settings = termios.tcgetattr(sys.stdin)
        if os.name == 'nt':
            timeout = 0.1
            startTime = time.time()
            while(1):
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


