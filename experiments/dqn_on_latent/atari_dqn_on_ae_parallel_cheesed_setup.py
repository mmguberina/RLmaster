import argparse
import os
import pprint

import numpy as np
import torch
from RLmaster.network.atari_network import DQNNoEncoder
from RLmaster.util.atari_wrapper import wrap_deepmind
from torch.utils.tensorboard import SummaryWriter

from tianshou.data import Collector, VectorReplayBuffer
from tianshou.env import ShmemVectorEnv, RayVectorEnv
from RLmaster.policy.reconstruction_loss_on_features_layers_q_on_last_ones import DQNOnReconstructionEncoderPolicy
from RLmaster.latent_representations.autoencoder_nn import CNNEncoderNew, CNNDecoder
#from tianshou.policy.modelbased.icm import ICMPolicy
from tianshou.trainer import offpolicy_trainer
from tianshou.utils import TensorboardLogger, WandbLogger
from tianshou.utils.net.discrete import IntrinsicCuriosityModule


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='PongNoFrameskip-v4')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--eps-test', type=float, default=0.005)
    parser.add_argument('--eps-train', type=float, default=1.)
    parser.add_argument('--eps-train-final', type=float, default=0.05)
#    parser.add_argument('--buffer-size', type=int, default=100000)
    parser.add_argument('--buffer-size', type=int, default=1000)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--n-step', type=int, default=3)
    parser.add_argument('--target-update-freq', type=int, default=500)
    parser.add_argument('--epoch', type=int, default=100)
#    parser.add_argument('--step-per-epoch', type=int, default=100000)
    parser.add_argument('--step-per-epoch', type=int, default=1000)
    #parser.add_argument('--step-per-collect', type=int, default=8)
    parser.add_argument('--step-per-collect', type=int, default=8)
    parser.add_argument('--update-per-step', type=float, default=0.1)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--training-num', type=int, default=8)
#    parser.add_argument('--training-num', type=int, default=1)
    parser.add_argument('--test-num', type=int, default=8)
#    parser.add_argument('--test-num', type=int, default=1)
    parser.add_argument('--logdir', type=str, default='log')
    parser.add_argument('--render', type=float, default=0.)
    parser.add_argument(
        '--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu'
    )
    parser.add_argument('--frames-stack', type=int, default=4)
    parser.add_argument('--resume-path', type=str, default=None)
    parser.add_argument('--resume-id', type=str, default=None)
    parser.add_argument(
        '--logger',
        type=str,
        default="tensorboard",
        choices=["tensorboard", "wandb"],
    )
    parser.add_argument(
        '--watch',
        default=False,
        action='store_true',
        help='watch the play of pre-trained policy only'
    )
    parser.add_argument('--save-buffer-name', type=str, default=None)
    parser.add_argument(
        '--icm-lr-scale',
        type=float,
        default=0.,
        help='use intrinsic curiosity module with this lr scale'
    )
    parser.add_argument(
        '--icm-reward-scale',
        type=float,
        default=0.01,
        help='scaling factor for intrinsic curiosity reward'
    )
    parser.add_argument(
        '--icm-forward-loss-weight',
        type=float,
        default=0.2,
        help='weight for the forward model loss in ICM'
    )
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


def test_dqn(args=get_args()):
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
    q_net = DQNNoEncoder(args.action_shape, args.device).to(args.device)
    # TODO put this features_dim in arguments later, make code clean
    features_dim = 3136 
    encoder = CNNEncoderNew(observation_shape=args.state_shape, features_dim=features_dim, device=args.device).to(args.device)
    decoder = CNNDecoder(observation_shape=args.state_shape, n_flatten=encoder.n_flatten, features_dim=features_dim).to(args.device)
    optim_q = torch.optim.Adam(q_net.parameters(), lr=args.lr)
    optim_encoder = torch.optim.Adam(encoder.parameters(), lr=args.lr)
    optim_decoder = torch.optim.Adam(decoder.parameters(), lr=args.lr)
    reconstruction_criterion = torch.nn.BCELoss()
    # define policy
    # TODO add this to device and solve the whole device situation
    policy = DQNOnReconstructionEncoderPolicy(
        encoder,
        decoder,
        q_net,
        optim_q,
        optim_encoder,
        optim_decoder,
        reconstruction_criterion,
        args.device,
        features_dim,
        args.gamma,
        args.n_step,
        target_update_freq=args.target_update_freq,
    )
    # load a previous policy
    if args.resume_path:
        policy.load_state_dict(torch.load(args.resume_path, map_location=args.device))
        print("Loaded agent from: ", args.resume_path)
    # replay buffer: `save_last_obs` and `stack_num` can be removed together
    # when you have enough RAM
    buffer = VectorReplayBuffer(
        args.buffer_size,
        buffer_num=len(train_envs),
        ignore_obs_next=True,
        save_only_last_obs=True,
        stack_num=args.frames_stack
    )
    # collector
    train_collector = Collector(policy, train_envs, buffer, exploration_noise=True)
    test_collector = Collector(policy, test_envs, exploration_noise=True)
    # log
    log_name = 'DQNOnReconstructionEncoderFromScratch_test'
    log_path = os.path.join(args.logdir, args.task, log_name)
    if args.logger == "tensorboard":
        writer = SummaryWriter(log_path)
        writer.add_text("args", str(args))
        logger = TensorboardLogger(writer)
    else:
        logger = WandbLogger(
            save_interval=1,
            project=args.task,
            name=log_name,
            run_id=args.resume_id,
            config=args,
        )

    def save_fn(policy):
        torch.save(policy.state_dict(), os.path.join(log_path, 'policy.pth'))

    def stop_fn(mean_rewards):
        if env.spec.reward_threshold:
            return mean_rewards >= env.spec.reward_threshold
        elif 'Pong' in args.task:
            return mean_rewards >= 20
        else:
            return False

    def train_fn(epoch, env_step):
        # nature DQN setting, linear decay in the first 1M steps
        if env_step <= 1e6:
            eps = args.eps_train - env_step / 1e6 * \
                (args.eps_train - args.eps_train_final)
        else:
            eps = args.eps_train_final
        policy.set_eps(eps)
        if env_step % 1000 == 0:
            logger.write("train/env_step", env_step, {"train/eps": eps})

    def test_fn(epoch, env_step):
        policy.set_eps(args.eps_test)

    def save_checkpoint_fn(epoch, env_step, gradient_step):
        # see also: https://pytorch.org/tutorials/beginner/saving_loading_models.html
        ckpt_path_q_network = os.path.join(log_path, 'checkpoint_q_network_epoch_' + str(epoch) + '.pth')
        ckpt_path_encoder = os.path.join(log_path, 'checkpoint_encoder_epoch_' + str(epoch) + '.pth')
        ckpt_path_decoder = os.path.join(log_path, 'checkpoint_decoder_epoch_' + str(epoch) + '.pth')
        torch.save({'q_network': policy.state_dict()}, ckpt_path_q_network)
        torch.save({'encoder': encoder.state_dict()}, ckpt_path_encoder)
        torch.save({'decoder': decoder.state_dict()}, ckpt_path_decoder)
        return "useless string for a useless return"

    # watch agent's performance
    def watch():
        print("Setup test envs ...")
        policy.eval()
        policy.set_eps(args.eps_test)
        test_envs.seed(args.seed)
        if args.save_buffer_name:
            print(f"Generate buffer with size {args.buffer_size}")
            buffer = VectorReplayBuffer(
                args.buffer_size,
                buffer_num=len(test_envs),
                ignore_obs_next=True,
                save_only_last_obs=True,
                stack_num=args.frames_stack
            )
            collector = Collector(policy, test_envs, buffer, exploration_noise=True)
            result = collector.collect(n_step=args.buffer_size)
            print(f"Save buffer into {args.save_buffer_name}")
            # Unfortunately, pickle will cause oom with 1M buffer size
            buffer.save_hdf5(args.save_buffer_name)
        else:
            print("Testing agent ...")
            test_collector.reset()
            result = test_collector.collect(
                n_episode=args.test_num, render=args.render
            )
        rew = result["rews"].mean()
        print(f'Mean reward (over {result["n/ep"]} episodes): {rew}')

    if args.watch:
        watch()
        exit(0)


    # test train_collector and start filling replay buffer
    train_collector.collect(n_step=args.batch_size * args.training_num)
    # trainer
    result = offpolicy_trainer(
        policy,
        train_collector,
        test_collector,
        args.epoch,
        args.step_per_epoch,
        args.step_per_collect,
        args.test_num,
        args.batch_size,
        train_fn=train_fn,
        test_fn=test_fn,
        stop_fn=stop_fn,
        save_fn=save_fn,
        logger=logger,
        update_per_step=args.update_per_step,
        test_in_train=False,
        resume_from_log=args.resume_id is not None,
        save_checkpoint_fn=save_checkpoint_fn,
    )

    pprint.pprint(result)
    watch()


if __name__ == '__main__':
    test_dqn(get_args())
