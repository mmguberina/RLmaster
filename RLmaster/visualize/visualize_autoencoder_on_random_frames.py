import gym
import numpy as np
import torch
from PIL import Image
import imagehash
from RLmaster.util.atari_wrapper import make_atari_env, AutoencoderEnv
#from RLmaster.latent_representations.autoencoder_nn import CNNEncoder, CNNDecoder
from RLmaster.latent_representations.autoencoder_nn import CNNEncoderNew, CNNDecoderNew
from RLmaster.util.save_load_hyperparameters import load_hyperparameters

import torch
import torch.nn as nn
from torchvision import transforms
from RLmaster.util.atari_dataset import AtariImageDataset
#import matplotlib.pyplot as plt
import cv2

# TODO just save the god damn observation space
# in the encoder class and stop this nonsense
transform = transforms.ToTensor()

#features_dim = 576
features_dim = 3136

#log_path = "../../experiments/latent_only/log/PongNoFrameskip-v4/unlabelled_experiment/"
log_path = "../../experiments/latent_only/log/PongNoFrameskip-v4/training_preloaded_buffer_fs_1/"
args = load_hyperparameters(log_path)
env = make_atari_env(args)
observation_shape = env.observation_space.shape or env.observation_space.n

#encoder = CNNEncoder(observation_shape=observation_shape, features_dim=features_dim)
#decoder = CNNDecoder(observation_shape=observation_shape, n_flatten=encoder.n_flatten, features_dim=features_dim)
#encoder.load_state_dict(torch.load("../../experiments/latent_only/encoder_features_dim_{}.pt".format(features_dim), map_location=torch.device('cpu')))
#decoder.load_state_dict(torch.load("../../experiments/latent_only/decoder_features_dim_{}.pt".format(features_dim), map_location=torch.device('cpu')))

encoder_name = "checkpoint_encoder_epoch_30.pth"
decoder_name = "checkpoint_decoder_epoch_30.pth"
#encoder_name = "encoder.pth"
#decoder_name = "decoder.pth"
encoder = CNNEncoderNew(observation_shape=observation_shape, features_dim=features_dim, device='cpu')
decoder = CNNDecoderNew(observation_shape=observation_shape, n_flatten=encoder.n_flatten, features_dim=features_dim)
print(torch.load(log_path + encoder_name, map_location=torch.device('cpu')))
#encoder.load_state_dict(torch.load(log_path + encoder_name, map_location=torch.device('cpu'))['encoder'])
#decoder.load_state_dict(torch.load(log_path + decoder_name, map_location=torch.device('cpu'))['decoder'])
encoder.load_state_dict(torch.load(log_path + encoder_name, map_location=torch.device('cpu')))
decoder.load_state_dict(torch.load(log_path + decoder_name, map_location=torch.device('cpu')))


#env = gym.make('PongNoFrameskip-v4')
#env = gym.make('Breakout-v4')
#env = WarpFrame(env)
#env.reset()
#obs, rew, done,info = env.step(0)
#obs, rew, done,info = env.step(0)
#obs, rew, done,info = env.step(0)
#print(obs)
#print(obs.shape)
#exit()
env = AutoencoderEnv(env, encoder, decoder, args.frames_stack)
env.reset()
print(env)

# would be much better if you stretched and scaled 
# the output image
# 84x84 is hard to see
# and you need to see
for i in range(5000):
    obs, reward, done, info = env.step(env.action_space.sample())
    obs = np.ceil(obs * 255).astype(np.uint8).reshape((84,84))
    if done:
        env.reset()
    cv2.imshow("cat", obs)
    cv2.waitKey(1)
