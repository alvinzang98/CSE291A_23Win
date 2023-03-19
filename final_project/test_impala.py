# Import required packages
from mani_skill2.utils.wrappers import RecordEpisode
import argparse
import os.path as osp
import pickle
import os

import gym
import numpy as np
import h5py
import torch as th
import torch.nn as nn
from gym.wrappers import TimeLimit
from tqdm.notebook import tqdm

import mani_skill2.envs
from mani_skill2.utils.wrappers import RecordEpisode
from torch.utils.data import Dataset, DataLoader
from impala import ImpalaNetwork

def convert_observation(observation):
    # flattens the original observation by flattening the state dictionaries
    # and combining the rgb and depth images

    # image data is not scaled here and is kept as uint16 to save space
    image_obs = observation["image"]
    rgb = image_obs["base_camera"]["rgb"]
    depth = image_obs["base_camera"]["depth"]
    rgb2 = image_obs["hand_camera"]["rgb"]
    depth2 = image_obs["hand_camera"]["depth"]

    # we provide a simple tool to flatten dictionaries with state data
    from mani_skill2.utils.common import flatten_state_dict
    state = np.hstack(
        [
            flatten_state_dict(observation["agent"]),
            flatten_state_dict(observation["extra"]),
        ]
    )

    # combine the RGB and depth images
    rgbd = np.concatenate([rgb, depth, rgb2, depth2], axis=-1)
    obs = dict(rgbd=rgbd, state=state)
    return obs
def rescale_rgbd(rgbd):
    # rescales rgbd data and changes them to floats
    rgb1 = rgbd[..., 0:3] / 255.0
    rgb2 = rgbd[..., 4:7] / 255.0
    depth1 = rgbd[..., 3:4] / (2**10)
    depth2 = rgbd[..., 7:8] / (2**10) 
    return np.concatenate([rgb1, depth1, rgb2, depth2], axis=-1)


device = 'cuda' if th.cuda.is_available() else 'cpu'
env_id = 'LiftCube-v1'
policy = ImpalaNetwork()
print(f'Testing Impala {env_id}')
path = os.path.join('./model/impala',f"{env_id}/ckpt_best.pt")
policy.load_state_dict(th.load(path)["policy"])
policy = policy.to(device)

obs_mode = "rgbd"
control_mode = "pd_ee_delta_pose"
env = gym.make(env_id, obs_mode=obs_mode, control_mode=control_mode)
# RecordEpisode wrapper auto records a new video once an episode is completed
env = RecordEpisode(env, output_dir=f"logs/rgbd_{env_id}/videos")
obs = env.reset(seed=42)

successes = []
num_episodes = 200
i = 0
pbar = tqdm(total=num_episodes)
pre_action = th.tensor([0,0,0,0,0,0,0]).unsqueeze(0).to(device)

while i < num_episodes:
    #print(i)
    # convert observation to our desired shape and move to appropriate device
    obs = convert_observation(obs)
    obs_device = dict()
    obs['rgbd'] = rescale_rgbd(obs['rgbd'])
    # unsqueeze adds an extra batch dimension and we permute rgbd since PyTorch expects the channel dimension to be first
    obs_device['rgbd'] = th.from_numpy(obs['rgbd']).float().permute(2,0,1).unsqueeze(0).to(device)
    obs_device['state'] = th.from_numpy(obs['state']).float().unsqueeze(0).to(device)
    with th.no_grad():
        action = policy(obs_device,pre_action).cpu().numpy()[0]
        pre_action = th.from_numpy(action).unsqueeze(0).to(device)
    obs, reward, done, info = env.step(action)

    if done:
        successes.append(info['success'])
        obs = env.reset()
        i += 1
        pbar.update(1)
print("Success Rate:", np.mean(successes))
#print(successes)