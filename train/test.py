import os
from vint_train.models.gnm.gnm import GNM
from vint_train.models.vint.vint import ViNT
from vint_train.models.gnm.gnm_vae import GNM_VAE
import torch
from typing import Tuple
from PIL import Image
import torchvision.transforms.functional as TF
import pickle
import random
import numpy as np


IMAGE_ASPECT_RATIO = (
    4 / 3
)  # all images are centered cropped to a 4:3 aspect ratio in training

def img_path_to_data(path, image_resize_size: Tuple[int, int]) -> torch.Tensor:
    """
    Load an image from a path and transform it
    Args:
        path (str): path to the image
        image_resize_size (Tuple[int, int]): size to resize the image to
    Returns:
        torch.Tensor: resized image as tensor
    """
    # return transform_images(Image.open(path), transform, image_resize_size, aspect_ratio)
    return resize_and_aspect_crop(Image.open(path), image_resize_size)

def resize_and_aspect_crop(
    img: Image.Image, image_resize_size: Tuple[int, int], aspect_ratio: float = IMAGE_ASPECT_RATIO
):
    w, h = img.size
    if w > h:
        img = TF.center_crop(img, (h, int(h * aspect_ratio)))  # crop to the right ratio
    else:
        img = TF.center_crop(img, (int(w / aspect_ratio), w))
    img = img.resize(image_resize_size)
    resize_img = TF.to_tensor(img)
    return resize_img

def load_obs(id):
    obs_path_start = '/home/zishuo/anomaly_violate3/forwardtempblock/mid_color/' + str(id) + '.jpg'
    return img_path_to_data(obs_path_start, (85, 64))

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

setup_seed(0)
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = GNM().to(device)
ckpt_path = './logs/gnm/gnm_2024_01_22_01_15_28/9.pth'
# ckpt_path = './gnm.pth'
checkpoint = torch.load(ckpt_path)

# model = ViNT().to(device)
# ckpt_path = './vint.pth'
# checkpoint = torch.load(ckpt_path)

# model = GNM_VAE().to(device)
# ckpt_path = './logs/gnm_vae_1e-6/gnm_vae_1e-6_2024_01_23_14_25_57/15.pth'
# checkpoint = torch.load(ckpt_path)

try:
    model.load_state_dict(checkpoint['model'].module.state_dict())
except:
    model.load_state_dict(checkpoint['model'].state_dict())

# goal_path = './mid_color/670.jpg'
goal_path = '/home/zishuo/anomaly_violate3/forwardtempblock/mid_color/230.jpg'
for start_id in range(220, 225, 1):
    model.eval()
    with torch.no_grad():
        obs = torch.cat([load_obs(start_id + i) for i in range(6)])
        goal = img_path_to_data(goal_path, (85, 64))
        obs = obs.unsqueeze(0)
        goal = goal.unsqueeze(0)
        obs = torch.as_tensor(obs, dtype=torch.float32).to(device)
        goal = torch.as_tensor(goal, dtype=torch.float32).to(device)
        try:
            dist, action = model(obs, goal)
        except:
            dist, action, mu, logvar = model(obs, goal)
            kl = torch.mean(-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1), dim=0)
            # print(kl)
            # print(start_id)
        print(dist)
        print(action * 20)



# path = '/home/zishuo/inet_gnm_data/t00000/traj_data.pkl'
# with open(path, "rb") as f:
#     traj_data = pickle.load(f)