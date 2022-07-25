import gym
import numpy as np
import torch
from PIL import Image
import imagehash
from RLmaster.util.atari_wrapper import make_atari_env, AutoencoderEnv
#from RLmaster.latent_representations.autoencoder_nn import CNNEncoder, CNNDecoder
from RLmaster.latent_representations.autoencoder_nn import CNNEncoderNew, CNNDecoderNew, RAE_ENC, RAE_DEC
from RLmaster.util.save_load_hyperparameters import load_hyperparameters
from collections import deque

import torch
import torch.nn as nn
from torchvision import transforms
from RLmaster.util.atari_dataset import AtariImageDataset
#import matplotlib.pyplot as plt
import cv2
from time import sleep

# TODO just save the god damn observation space
# in the encoder class and stop this nonsense

#features_dim = 576
features_dim = 3136
#features_dim = 50

#log_path = "../../experiments/latent_only/log/PongNoFrameskip-v4/unlabelled_experiment/"
#log_path = "../../experiments/latent_only/log/PongNoFrameskip-v4/training_preloaded_buffer_fs_1/"
#log_path = "../../experiments/latent_only/log/PongNoFrameskip-v4/ae_trained_as_policy/"
#log_path = "../../log/dqn_ae_parallel_good_arch_fs_4_passing_q_grads/"
#log_path = "../../log/ae_single-frame-trained_as_policy_3136_4/"
#log_path = "../../log/rae_single-frame-trained_as_policy_1/"
#log_path = "../../log/raibow_ae_parallel_good_arch_fs_4_passing_q_grads_6/"
log_path = "../../log/latent_only/PongNoFrameskip-v4/ae_compressed-frame-trained_as_policy_3136/"
#log_path = "../../experiments/latent_only/log/PongNoFrameskip-v4/ae_single-frame-trained_as_policy_3136/"
#log_path_enc_dc = "../../experiments/latent_only/log/PongNoFrameskip-v4/"
args = load_hyperparameters(log_path)
# TODO fix this bug where frames_stack is not done in make_atari_env, but elsewhere
#args.frames_stack = 2
env, test_envs = make_atari_env(args.task, args.seed, 1, 1, frames_stack=args.frames_stack)
observation_shape = env.observation_space.shape or env.observation_space.n
print(observation_shape)
if args.latent_space_type == "single-frame-predictor":
    observation_shape = list(args.state_shape)
    observation_shape[0] = 1 
    observation_shape = tuple(observation_shape)

if args.latent_space_type == "forward-frame-predictor":
    observation_shape = list(args.state_shape)
    observation_shape[0] = 2 
    observation_shape = tuple(observation_shape)
#print(observation_shape)
#encoder = CNNEncoder(observation_shape=observation_shape, features_dim=features_dim)
#decoder = CNNDecoder(observation_shape=observation_shape, n_flatten=encoder.n_flatten, features_dim=features_dim)
#encoder.load_state_dict(torch.load("../../experiments/latent_only/encoder_features_dim_{}.pt".format(features_dim), map_location=torch.device('cpu')))
#decoder.load_state_dict(torch.load("../../experiments/latent_only/decoder_features_dim_{}.pt".format(features_dim), map_location=torch.device('cpu')))

encoder_name = "checkpoint_encoder_epoch_8.pth"
decoder_name = "checkpoint_decoder_epoch_8.pth"
#encoder_name = "encoder.pth"
#decoder_name = "decoder.pth"
#encoder_name = "encoder_features_dim_3136.pt"
#decoder_name = "decoder_features_dim_3136.pt"
encoder = CNNEncoderNew(observation_shape=observation_shape, features_dim=features_dim, device='cpu')
#encoder = RAE_ENC("cpu", observation_shape, features_dim)
decoder = CNNDecoderNew(observation_shape=observation_shape, n_flatten=encoder.n_flatten, features_dim=features_dim)
#decoder = RAE_DEC("cpu", observation_shape, features_dim)
#print(torch.load(log_path + encoder_name, map_location=torch.device('cpu')))
encoder.load_state_dict(torch.load(log_path + encoder_name, map_location=torch.device('cpu'))['encoder'])
decoder.load_state_dict(torch.load(log_path + decoder_name, map_location=torch.device('cpu'))['decoder'])
#encoder.load_state_dict(torch.load(log_path + encoder_name, map_location=torch.device('cpu')))
#decoder.load_state_dict(torch.load(log_path + decoder_name, map_location=torch.device('cpu')))


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
#env = AutoencoderEnv(env, encoder, decoder, args.frames_stack)
#env = make_atari_env(args)
env.reset()
#print(env)

# would be much better if you stretched and scaled 
# the output image
# 84x84 is hard to see
# and you need to see
#queue = deque([], maxlen=args.frames_stack)
#print(queue)
#exit()
for i in range(5000):
    sleep(0.02)
    obs, reward, done, info = env.step(np.array([env.action_space.sample()]))
    #obs = torch.tensor(obs)
    #obs = torch.tensor(obs)[:,:-2,:,:].view(1, args.frames_stack, 84, 84)
    #obs = torch.tensor(obs)[:,-1,:,:].view(1,1, 84, 84)
    obs = torch.tensor(obs)
#    print(obs.shape)
#    obs = obs[:,-1,:,:].view(1,1,84,84) / 255
    #obs = np.ceil(obs * 255).astype(np.uint8).reshape((84,84))
    with torch.no_grad():
        #obs = decoder(encoder(obs.reshape((1,1,84,84))))
        obs = decoder(encoder(obs))
    obs = obs[:,-1:,:].reshape((84,84)).numpy()
    if done:
        env.reset()
    cv2.imshow("cat", obs)
    cv2.waitKey(1)
