import gym
import numpy as np
import torch
from RLmaster.util.save_load_hyperparameters import load_hyperparameters
#from tianshou.data.buffer.base import ReplayBuffer
from tianshou.data import Batch, ReplayBuffer, to_numpy, to_torch

import os
import cv2
import matplotlib.pyplot as plt


#features_dim = 576
features_dim = 3136

log_path = "../../experiments/latent_only/log/PongNoFrameskip-v4/unlabelled_experiment/"
buffer_path = os.path.join(log_path, "buffer.h5")
args = load_hyperparameters(log_path)
buffer = ReplayBuffer.load_hdf5(buffer_path)

# would be much better if you stretched and scaled 
# the output image
# 84x84 is hard to see
# and you need to see

def showAsVideo(buffer):
    for i in range(5000):
        obs = buffer.obs[i]
        #obs = np.ceil(obs * 255).astype(np.uint8).reshape((84,84))
        obs = obs.reshape((84,84))
        cv2.imshow("cat", obs)
        cv2.waitKey(1)

def showSampleAsImg(buffer):
    batch_size = 8
    frames_stack = 4
    buffer._size = args.buffer_size
    buffer.stack_num = frames_stack
    batch, indeces = buffer.sample(batch_size)
    batch.obs = batch.obs[:, -1, :, :].reshape((batch_size, 84, 84))
    fig = plt.figure(figsize=(batch_size // 2, 2))
    for i in np.arange(batch_size):
        ax = fig.add_subplot(2, 8//2, i + 1, xticks=[], yticks=[])
        plt.imshow(batch.obs[i])
    plt.show()

showSampleAsImg(buffer)

