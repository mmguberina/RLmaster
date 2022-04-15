import argparse
import datetime
import os
import pprint

import numpy as np
import torch
from atari_network import Rainbow
from atari_wrapper import wrap_deepmind
from torch.utils.tensorboard import SummaryWriter

from tianshou.data import Collector, PrioritizedVectorReplayBuffer, VectorReplayBuffer
from tianshou.env import ShmemVectorEnv
from tianshou.policy import RainbowPolicy
from tianshou.trainer import offpolicy_trainer
from tianshou.utils import TensorboardLogger

import tianshou as ts


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='PongNoFrameskip-v4')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--eps-test', type=float, default=0.005)
    parser.add_argument('--eps-train', type=float, default=1.)
    parser.add_argument('--eps-train-final', type=float, default=0.05)
    parser.add_argument('--buffer-size', type=int, default=100000)
    parser.add_argument('--lr', type=float, default=0.0000625)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--num-atoms', type=int, default=51)
    parser.add_argument('--v-min', type=float, default=-10.)
    parser.add_argument('--v-max', type=float, default=10.)
    parser.add_argument('--noisy-std', type=float, default=0.1)
    parser.add_argument('--no-dueling', action='store_true', default=False)
    parser.add_argument('--no-noisy', action='store_true', default=False)
    parser.add_argument('--no-priority', action='store_true', default=False)
    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--beta', type=float, default=0.4)
    parser.add_argument('--beta-final', type=float, default=1.)
    parser.add_argument('--beta-anneal-step', type=int, default=5000000)
    parser.add_argument('--no-weight-norm', action='store_true', default=False)
    parser.add_argument('--n-step', type=int, default=3)
    parser.add_argument('--target-update-freq', type=int, default=500)
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--step-per-epoch', type=int, default=100000)
    parser.add_argument('--step-per-collect', type=int, default=10)
    parser.add_argument('--update-per-step', type=float, default=0.1)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--training-num', type=int, default=10)
    parser.add_argument('--test-num', type=int, default=10)
    parser.add_argument('--logdir', type=str, default='log')
    parser.add_argument('--render', type=float, default=1.)
    parser.add_argument('--resume_path', type=str, default="/home/gospodar/chalmers/MASTER/tianshou/examples/atari/log/PongNoFrameskip-v4/rainbow/seed_0_0214-093958/policy.pth")
    parser.add_argument(
        '--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu'
    )
    parser.add_argument('--frames-stack', type=int, default=4)
    parser.add_argument('--resume-path', type=str, default=None)
    parser.add_argument(
        '--watch',
        default=False,
        action='store_true',
        help='watch the play of pre-trained policy only'
    )
    parser.add_argument('--save-buffer-name', type=str, default=None)
    return parser.parse_args()


def make_atari_env(args):
    return wrap_deepmind(args.task, frame_stack=args.frames_stack)


def make_atari_env_watch(args):
    return wrap_deepmind(
        args.task,
        frame_stack=args.frames_stack,
        episode_life=False,
        clip_rewards=False
    )


def test_rainbow(args=get_args()):
    env = make_atari_env(args)
    args.state_shape = env.observation_space.shape or env.observation_space.n
    args.action_shape = env.action_space.shape or env.action_space.n
    # should be N_FRAMES x H x W
    print("Observations shape:", args.state_shape)
    print("Actions shape:", args.action_shape)
    # make environments
    train_envs = ShmemVectorEnv(
        [lambda: make_atari_env(args) for _ in range(args.training_num)]
    )
    test_envs = ShmemVectorEnv(
        [lambda: make_atari_env_watch(args) for _ in range(args.test_num)]
    )
    # seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    train_envs.seed(args.seed)
    test_envs.seed(args.seed)
    # define model
    net = Rainbow(
        *args.state_shape,
        args.action_shape,
        args.num_atoms,
        args.noisy_std,
        args.device,
        is_dueling=not args.no_dueling,
        is_noisy=not args.no_noisy
    )
    optim = torch.optim.Adam(net.parameters(), lr=args.lr)
    # define policy
    policy = RainbowPolicy(
        net,
        optim,
        args.gamma,
        args.num_atoms,
        args.v_min,
        args.v_max,
        args.n_step,
        target_update_freq=args.target_update_freq
    ).to(args.device)
    # load a previous policy
    if args.resume_path:
        policy.load_state_dict(torch.load(args.resume_path, map_location=args.device))
        print("Loaded agent from: ", args.resume_path)
    # replay buffer: `save_last_obs` and `stack_num` can be removed together
    # when you have enough RAM

def force_cudnn_initialization():
    s = 32
    dev = torch.device('cuda')
    torch.nn.functional.conv2d(torch.zeros(s, s, s, s, device=dev), torch.zeros(s, s, s, s, device=dev))

if __name__ == '__main__':
#    force_cudnn_initialization()
    args = get_args()

    env = make_atari_env(args)
    args.state_shape = env.observation_space.shape or env.observation_space.n
    args.action_shape = env.action_space.shape or env.action_space.n
    # should be N_FRAMES x H x W
    print("Observations shape:", args.state_shape)
    print("Actions shape:", args.action_shape)
    # make environments
    # seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    # define model
    net = Rainbow(
        *args.state_shape,
        args.action_shape,
        args.num_atoms,
        args.noisy_std,
        args.device,
        is_dueling=not args.no_dueling,
        is_noisy=not args.no_noisy
    )
    optim = torch.optim.Adam(net.parameters(), lr=args.lr)
    # define policy
    policy = RainbowPolicy(
        net,
        optim,
        args.gamma,
        args.num_atoms,
        args.v_min,
        args.v_max,
        args.n_step,
        target_update_freq=args.target_update_freq
    ).to(args.device)
    # load a previous policy
    if args.resume_path:
        policy.load_state_dict(torch.load(args.resume_path, map_location=args.device))
        print("Loaded agent from: ", args.resume_path)
    policy.eval()
    policy.set_eps(0.09)
    collector = ts.data.Collector(policy, env, exploration_noise=False)
    collector.collect(n_episode=1, render=1 / 25)
