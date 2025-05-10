import os
import wandb
import argparse
import numpy as np
import yaml
import time
import pdb

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset
from torch.optim import Adam, AdamW
from torchvision import transforms
import torch.backends.cudnn as cudnn

"""
IMPORT YOUR MODEL HERE
"""
from vint_train.models.gnm.gnm_vae import GNM_VAE
from vint_train.data.vint_dataset import ViNT_Dataset_Calibration as ViNT_Dataset
import random
from typing import List, Optional, Dict
import tqdm
import itertools
import torch.nn.functional as F


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.deterministic = True


def _compute_losses_vae(
        mu,
        logvar,
):
    kldivergence = torch.mean(-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1), dim=0)
    return kldivergence


def evaluate_vae(
        eval_type: str,
        model: nn.Module,
        dataloader: DataLoader,
        transform: transforms,
        device: torch.device,
        beta: float = 1.0,
        alpha: float = 0.5,
        save_pred_path = '/home/zishuo/GNM_VAE/train'
):
    model.eval()
    num_batches = len(dataloader)
    with torch.no_grad():
        tqdm_iter = tqdm.tqdm(
            itertools.islice(dataloader, num_batches),
            total=num_batches,
            dynamic_ncols=True,
            desc=f"Evaluating {eval_type}",
        )

        for i, data in enumerate(tqdm_iter):
            (
                obs_list,
                goal_image,
            ) = data

            kl_list = []

            goal_image = transform(goal_image).to(device)
            for id_obs in range(len(obs_list)):
                obs_image = obs_list[id_obs] 
                obs_images = torch.split(obs_image, 3, dim=1)

                # save_img = obs_images[5].squeeze().cpu().numpy()
                # save_img = np.transpose(save_img, (1, 2, 0))
                # save_img = (save_img * 255).astype(np.uint8)
                # import PIL
                # save_img = PIL.Image.fromarray(save_img)
                # save_img.save(os.path.join(save_pred_path, f"obs_image_{i}_{id_obs}.png"))

                obs_images = [transform(obs_image).to(device) for obs_image in obs_images]
                obs_image = torch.cat(obs_images, dim=1)

                model_outputs = model(obs_image, goal_image)

                dist_pred, action_pred, mu, logvar = model_outputs                

                kl = _compute_losses_vae(
                    mu=mu,
                    logvar=logvar,
                )

                kl_list.append(kl.cpu().numpy())
            
            pred_kl_save = np.array(kl_list)
            if i == 0:
                stacked = pred_kl_save
            else:
                stacked = np.column_stack((stacked, pred_kl_save))

            if (i + 1) % 200 == 0:
                np.savetxt(os.path.join(save_pred_path, f"kl_calibtration.csv"), stacked, fmt='%.2f', delimiter=',')


def calibrate(
        model: nn.Module,
        test_dataloaders: Dict[str, DataLoader],
        transform: transforms,
        device: torch.device,
        beta: float = 1.0,
        alpha: float = 0.5,
):

    for dataset_type in test_dataloaders:
        loader = test_dataloaders[dataset_type]

        evaluate_vae(
            eval_type=dataset_type,
            model=model,
            dataloader=loader,
            transform=transform,
            device=device,
            beta=beta,
            alpha=alpha,
        )

       
            

def main(config):
    assert config["distance"]["min_dist_cat"] < config["distance"]["max_dist_cat"]
    assert config["action"]["min_dist_cat"] < config["action"]["max_dist_cat"]

    setup_seed(0)
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    cudnn.benchmark = True  # good if input sizes don't vary
    transform = ([
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    transform = transforms.Compose(transform)

    test_dataloaders = {}

    if "context_type" not in config:
        config["context_type"] = "temporal"

    if "clip_goals" not in config:
        config["clip_goals"] = False

    for dataset_name in config["datasets"]:
        data_config = config["datasets"][dataset_name]
        if "negative_mining" not in data_config:
            data_config["negative_mining"] = True
        if "goals_per_obs" not in data_config:
            data_config["goals_per_obs"] = 1
        if "end_slack" not in data_config:
            data_config["end_slack"] = 0
        if "waypoint_spacing" not in data_config:
            data_config["waypoint_spacing"] = 1

        for data_split_type in ["test"]:
            if data_split_type in data_config:
                    dataset = ViNT_Dataset(
                        data_folder=data_config["data_folder"],
                        data_split_folder=data_config[data_split_type],
                        dataset_name=dataset_name,
                        image_size=config["image_size"],
                        waypoint_spacing=data_config["waypoint_spacing"],
                        min_dist_cat=config["distance"]["min_dist_cat"],
                        max_dist_cat=config["distance"]["max_dist_cat"],
                        min_action_distance=config["action"]["min_dist_cat"],
                        max_action_distance=config["action"]["max_dist_cat"],
                        negative_mining=data_config["negative_mining"],
                        len_traj_pred=config["len_traj_pred"],
                        learn_angle=config["learn_angle"],
                        context_size=config["context_size"],
                        context_type=config["context_type"],
                        end_slack=data_config["end_slack"],
                        goals_per_obs=data_config["goals_per_obs"],
                        normalize=config["normalize"],
                        goal_type=config["goal_type"],
                    )

                    dataset_type = f"{dataset_name}_{data_split_type}"
                    if dataset_type not in test_dataloaders:
                        test_dataloaders[dataset_type] = {}
                    test_dataloaders[dataset_type] = dataset

    for dataset_type, dataset in test_dataloaders.items():
        test_dataloaders[dataset_type] = DataLoader(
            dataset,
            batch_size=1,
            shuffle=True,
            num_workers=32,
            drop_last=False,
        )

    model = GNM_VAE(
        config["context_size"],
        config["len_traj_pred"],
        config["learn_angle"],
        config["obs_encoding_size"],
        config["goal_encoding_size"],
    )

    model = model.to(device)

    ckpt_path = './logs/gnm_vae/gnm_vae_2025_04_18_00_56_03/26.pth'
    # ckpt_path = './logs/gnm_vae_1e-6/gnm_vae_1e-6_2024_01_23_14_25_57/latest.pth'
    checkpoint = torch.load(ckpt_path)
    try:
        model.load_state_dict(checkpoint['model'].module.state_dict())
    except:
        model.load_state_dict(checkpoint['model'].state_dict())

    calibrate(
        model=model,
        test_dataloaders=test_dataloaders,
        transform=transform,
        device=device,
        beta=config["beta"],
        alpha=config["alpha"],
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visual Navigation Transformer")

    # project setup
    parser.add_argument(
        "--config",
        "-c",
        default="config/gnm_vae.yaml",
        type=str,
        help="Path to the config file in train_config folder",
    )
    args = parser.parse_args()

    with open("config/defaults.yaml", "r") as f:
        default_config = yaml.safe_load(f)

    config = default_config

    with open(args.config, "r") as f:
        user_config = yaml.safe_load(f)

    config.update(user_config)

    config["run_name"] += "_" + time.strftime("%Y_%m_%d_%H_%M_%S")
    config["project_folder"] = os.path.join(
        "logs", config["project_name"], config["run_name"]
    )

    print(config)
    main(config)
