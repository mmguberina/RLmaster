import gym
import numpy as np
import torch
from PIL import Image
import imagehash
from atari_wrapper import WarpFrame, AutoencoderEnv
from autoencoder_nn import NatureCNNEncoder, CNNDecoder

import torch
import torch.nn as nn
from torchvision import transforms
from atari_dataset import AtariImageDataset
#import matplotlib.pyplot as plt
import cv2

# TODO just save the god damn observation space
# in the encoder class and stop this nonsense
dir_name = "pong_dataset"
transform = transforms.ToTensor()
num_workers = 0
batch_size = 20
train_dataset = AtariImageDataset(root_dir="/home/gospodar/chalmers/MASTER/RLmaster/", 
                                  dir_name="pong_dataset", transform=transform, train=True)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers)

dataiter = iter(train_loader)
images = dataiter.next()

observation_shape = images[0].shape

encoder = NatureCNNEncoder(observation_shape=observation_shape)
decoder = CNNDecoder(observation_shape=observation_shape, n_flatten=encoder.n_flatten)
encoder.load_state_dict(torch.load("./encoder.pt", map_location=torch.device('cpu')))
decoder.load_state_dict(torch.load("./decoder.pt", map_location=torch.device('cpu')))

env = gym.make('PongNoFrameskip-v4')
env = WarpFrame(env)
#exit()
env = AutoencoderEnv(env, encoder, decoder)
env.reset()
print(env)


for i in range(500):
    obs, reward, done, info = env.step(env.action_space.sample())
    obs = env.observation(obs)
    obs = np.ceil(obs * 255).astype(np.uint8).reshape((84,84))
    if done:
        env.reset()
    cv2.imshow("cat", obs)
    cv2.waitKey(1)
