import gym
import numpy as np
import torch
from PIL import Image
import imagehash
from RLmaster.util.atari_wrapper import WarpFrame, AutoencoderEnv
from RLmaster.latent_representations.autoencoder_nn import CNNEncoder, CNNDecoder

import torch
import torch.nn as nn
from torchvision import transforms
from RLmaster.util.atari_dataset import AtariImageDataset
#import matplotlib.pyplot as plt
import cv2

# TODO just save the god damn observation space
# in the encoder class and stop this nonsense
dir_name = "Pong-v4_dataset"
transform = transforms.ToTensor()
num_workers = 0
batch_size = 20
train_dataset = AtariImageDataset(root_dir="/home/gospodar/chalmers/MASTER/RLmaster/RLmaster/util/", 
                                  dir_name=dir_name, transform=transform, train=True)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers)

dataiter = iter(train_loader)
images = dataiter.next()

observation_shape = images[0].shape

features_dim = 576

encoder = CNNEncoder(observation_shape=observation_shape, features_dim=features_dim)
decoder = CNNDecoder(observation_shape=observation_shape, n_flatten=encoder.n_flatten, features_dim=features_dim)
encoder.load_state_dict(torch.load("../../experiments/latent_only/encoder_features_dim_{}.pt".format(features_dim), map_location=torch.device('cpu')))
decoder.load_state_dict(torch.load("../../experiments/latent_only/decoder_features_dim_{}.pt".format(features_dim), map_location=torch.device('cpu')))

env = gym.make('PongNoFrameskip-v4')
#env = gym.make('Breakout-v4')
env = WarpFrame(env)
#exit()
env = AutoencoderEnv(env, encoder, decoder)
env.reset()
print(env)

# would be much better if you stretched and scaled 
# the output image
# 84x84 is hard to see
# and you need to see
for i in range(500):
    obs, reward, done, info = env.step(env.action_space.sample())
    obs = np.ceil(obs * 255).astype(np.uint8).reshape((84,84))
    if done:
        env.reset()
    cv2.imshow("cat", obs)
    cv2.waitKey(1)
