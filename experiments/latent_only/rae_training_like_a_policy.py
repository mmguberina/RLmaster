import argparse
import os
import pprint

import numpy as np
import torch
from RLmaster.network.atari_network import DQNNoEncoder, RainbowNoConvLayers, Rainbow
from RLmaster.util.atari_wrapper import wrap_deepmind, make_atari_env, make_atari_env_watch
from torch.utils.tensorboard import SummaryWriter
from RLmaster.util.save_load_hyperparameters import save_hyperparameters

from tianshou.data import Collector, VectorReplayBuffer
from tianshou.env import ShmemVectorEnv
# TODO write the autoencoder only policy
#from RLmaster.policy.autoencoder_only import AutoencoderOnly
from RLmaster.policy.random import RandomPolicy
from RLmaster.policy.dqn_fixed import RainbowPolicyFixed
from RLmaster.latent_representations.autoencoder_learning_as_policy_wrapper import AutoencoderLatentSpacePolicy
from RLmaster.latent_representations.autoencoder_nn import RAE_ENC, RAE_DEC, CNNEncoderNew, CNNDecoderNew, RAE_predictive_DEC
from RLmaster.util.collector_on_latent import CollectorOnLatent
from tianshou.trainer import offpolicy_trainer
from tianshou.utils import TensorboardLogger
from RLmaster.util.save_load_hyperparameters import load_hyperparameters

"""
action are all random sampled
1 frame (1,84,84) -> pass through autoencoder -> 1 frame (1,84,84)
"""


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='BreakoutNoFrameskip-v4')
    parser.add_argument('--latent-space-type', type=str, default='single-frame-predictor')
    parser.add_argument('--use-reconstruction-loss', type=int, default=True)
    parser.add_argument('--use-q-loss', type=bool, default=False)
    parser.add_argument('--squeeze-latent-into-single-vector', type=bool, default=True)
    parser.add_argument('--use-pretrained-latent', type=int, default=False)
    parser.add_argument('--use-pretrained-rl', type=int, default=True)
    parser.add_argument('--pretrained-rl-log-name', type=str, default='rainbow_only_small_enc_01')
    parser.add_argument('--pass-q-grads-to-encoder', type=bool, default=False)
    parser.add_argument('--use-regularization-on-unsupervised', type=bool, default=True)
    parser.add_argument('--data-augmentation', type=bool, default=True)
    #parser.add_argument('--data-augmentation', type=bool, default=False)
    # NOTE the arg below is not used atm
    parser.add_argument('--forward-prediction-in-latent', type=bool, default=False)
    # TODO implement
    parser.add_argument('--alternating-training-frequency', type=int, default=1)
    #parser.add_argument('--features_dim', type=int, default=50)
    parser.add_argument('--features_dim', type=int, default=512)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument("--scale-obs", type=int, default=0)
    parser.add_argument('--eps-test', type=float, default=0.005)
    parser.add_argument('--eps-train', type=float, default=0.31)
    parser.add_argument('--eps-train-final', type=float, default=0.3)
    parser.add_argument('--buffer-size', type=int, default=100000)
    parser.add_argument('--lr-rl', type=float, default=0.0000625)
    parser.add_argument('--lr-unsupervised', type=float, default=0.0001)
    # TODO understand where exactly this is used and why
    # it's probably how often you update the target policy network in deep-Q
    parser.add_argument('--target-update-freq', type=int, default=500)
    #parser.add_argument('--target-update-freq', type=int, default=5)
#    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--epoch', type=int, default=15)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--num-atoms', type=int, default=51)
    parser.add_argument('--v-min', type=float, default=-10.)
    parser.add_argument('--v-max', type=float, default=10.)
    parser.add_argument("--noisy-std", type=float, default=0.1)
    parser.add_argument("--no-dueling", action="store_true", default=False)
    parser.add_argument("--no-noisy", action="store_true", default=False)
    parser.add_argument("--no-priority", action="store_true", default=False)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--beta", type=float, default=0.4)
    parser.add_argument("--beta-final", type=float, default=1.)
    parser.add_argument("--beta-anneal-step", type=int, default=5000000)
    parser.add_argument("--no-weight-norm", action="store_true", default=False)
    parser.add_argument('--n-step', type=int, default=3)
    #parser.add_argument('--n-step', type=int, default=20)
    parser.add_argument('--step-per-epoch', type=int, default=100000)
    # TODO why 8?
    parser.add_argument('--step-per-collect', type=int, default=8)
    # TODO understand where exactly this is used and why
    # why is this a float?
    parser.add_argument('--update-per-step', type=float, default=0.3)
#    parser.add_argument('--update-per-step', type=float, default=0.6)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--training-num', type=int, default=8)
#    parser.add_argument('--training-num', type=int, default=2)
    # tests aren't necessary as we're free to overfit as much as we want
    # the training domain IS the testing domain
#    parser.add_argument('--test-num', type=int, default=8)
    parser.add_argument('--test-num', type=int, default=1)
    parser.add_argument('--logdir', type=str, default='log')
    parser.add_argument('--log-name', type=str, default='rae_compressor-trained_on-pretrained-rl-feature-dim-512-01')
#    parser.add_argument('--log-name', type=str, default='inverse_dynamics_model_1')
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
# TODO write the watch function
# basically run some functions from visualize
# or don't 'cos you'll be cloud training, whatever
#    parser.add_argument(
#        '--watch',
#        default=False,
#        action='store_true',
#        help='watch the play of pre-trained policy only'
#    )
    parser.add_argument('--save-buffer-name', type=str, default=None)
    args = parser.parse_args()
    return args



if __name__ == '__main__':
#def test_dqn(args=get_args()):
#    torch.set_num_threads(1)
    args=get_args()
    #env = make_atari_env(args)
    train_envs, test_envs = make_atari_env(
        args.task,
        args.seed,
        args.training_num,
        args.test_num,
        scale=args.scale_obs,
        frame_stack=args.frames_stack,
    )
    args.state_shape = train_envs.observation_space.shape or train_envs.observation_space.n
    args.action_shape = train_envs.action_space.shape or train_envs.action_space.n
    # TODO deal with this in a proper place
    observation_shape = args.state_shape
    # should be N_FRAMES x H x W
    print("Observations shape:", args.state_shape)
    print("Actions shape:", args.action_shape)
    # seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    train_envs.seed(args.seed)
    test_envs.seed(args.seed)
    # in this experiment we're using the random policy
    # which is just a placeholder really
    if not args.use_pretrained_rl:
        rl_policy = RandomPolicy(args.action_shape)

    else:
        pretrained_rl_log_path = "../rainbow_on_latent/log/" + args.task + \
        "/" + args.pretrained_rl_log_name + "/"
        pretrained_rl_args = load_hyperparameters(pretrained_rl_log_path)
        encoder_pretrained_rl = CNNEncoderNew(observation_shape=observation_shape, 
                features_dim=pretrained_rl_args.features_dim, device=pretrained_rl_args.device).to(pretrained_rl_args.device)
        rl_input_dim = pretrained_rl_args.features_dim 
        pretrained_rl_net = RainbowNoConvLayers(pretrained_rl_args.action_shape, 
                                     pretrained_rl_args.num_atoms,
                                     pretrained_rl_args.noisy_std,
                                     pretrained_rl_args.device,
                                     not pretrained_rl_args.no_dueling,
                                     not pretrained_rl_args.no_noisy,
                                     rl_input_dim)#.to(pretrained_rl_args.device)
        pretrained_rl_encoder_name = "encoder.pth"
        pretrained_rl_net_name = "rainbow_net.pth"

        encoder_pretrained_rl.load_state_dict(torch.load(pretrained_rl_log_path + pretrained_rl_encoder_name, map_location=torch.device('cpu')))
        #encoder_pretrained_rl.load_state_dict(torch.load(pretrained_rl_log_path + pretrained_rl_encoder_name, map_location=torch.device('cpu'))['encoder'])
        pretrained_rl_net.load_state_dict(torch.load(pretrained_rl_log_path + pretrained_rl_net_name, map_location=torch.device('cpu')))
        #pretrained_rl_net.load_state_dict(torch.load(pretrained_rl_log_path + pretrained_rl_net_name, map_location=torch.device('cpu'))['encoder'])
        encoder_pretrained_rl.to(args.device)
        encoder_pretrained_rl.requires_grad_(False)
        pretrained_rl_net.to(args.device)
        pretrained_rl_net.requires_grad_(False)
        optim_q = None

        #rl_policy = RainbowPolicy(
        rl_policy = RainbowPolicyFixed(
            pretrained_rl_net,
            optim_q,
            args.gamma,
            args.num_atoms,
            args.v_min,
            args.v_max,
            args.n_step,
            target_update_freq=args.target_update_freq
        ).to(args.device)


    if not args.squeeze_latent_into_single_vector:
        # in this case, we don't pass the stacked frames.
        # we unstack them, compress them, the stack the compressed ones and
        # pass that to the policy
        observation_shape = list(args.state_shape)
        observation_shape[0] = 1 
        observation_shape = tuple(observation_shape)
    encoder = RAE_ENC(args.device, observation_shape, 
            args.features_dim).to(args.device)
    print(encoder)
    if args.latent_space_type == "forward-frame-predictor":
        decoder = RAE_predictive_DEC(args.device, observation_shape, 
                args.features_dim, args.squeeze_latent_into_single_vector, args.frames_stack).to(args.device)
    else:
        decoder = RAE_DEC(args.device, observation_shape, 
                args.features_dim).to(args.device)
    print(decoder)
#    encoder = CNNEncoderNew(observation_shape=args.state_shape, features_dim=args.features_dim, device=args.device).to(args.device)
#    decoder = CNNDecoderNew(observation_shape=args.state_shape, n_flatten=encoder.n_flatten, features_dim=args.features_dim).to(args.device)
    optim_encoder = torch.optim.Adam(encoder.parameters(), lr=args.lr_unsupervised)

    if args.use_regularization_on_unsupervised:
        optim_decoder = torch.optim.Adam(decoder.parameters(), lr=args.lr_unsupervised, weight_decay=10**-7)
    else:
        optim_decoder = torch.optim.Adam(decoder.parameters(), lr=args.lr_unsupervised)
    reconstruction_criterion = torch.nn.MSELoss()



    policy = AutoencoderLatentSpacePolicy(
        rl_policy,
        args.latent_space_type,
        encoder,
        decoder,
        optim_encoder,
        optim_decoder,
        reconstruction_criterion,
        args.batch_size,
        args.frames_stack,
        args.device,
        args.squeeze_latent_into_single_vector,
        args.use_reconstruction_loss,
        args.pass_q_grads_to_encoder,
        args.alternating_training_frequency,
        args.data_augmentation,
        args.forward_prediction_in_latent,
        args.use_q_loss,
        args.use_regularization_on_unsupervised,
        encoder_pretrained_rl
    )
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
    log_path = os.path.join(args.logdir, args.task, args.log_name)
    writer = SummaryWriter(log_path)
    writer.add_text("args", str(args))
    logger = TensorboardLogger(writer)

    save_hyperparameters(args)

    def save_fn(policy):
        torch.save(encoder.state_dict(), os.path.join(log_path, 'encoder.pth'))
        torch.save(decoder.state_dict(), os.path.join(log_path, 'decoder.pth'))

    def stop_fn(mean_rewards):
        if train_envs.spec.reward_threshold:
            return mean_rewards >= train_envs.spec.reward_threshold
        elif 'Pong' in args.task:
            return mean_rewards >= 20
        else:
            return False

    # nature DQN setting, linear decay in the first 1M steps
    # NOTE none of this is a thing here because we're using a random policy,
    def train_fn(epoch, env_step):
        pass
        # and not q-learning
#        if env_step <= 1e6:
#            eps = args.eps_train - env_step / 1e6 * \
#                (args.eps_train - args.eps_train_final)
#        else:
#            eps = args.eps_train_final
#        policy.set_eps(eps)
#        if env_step % 1000 == 0:
#            logger.write("train/env_step", env_step, {"train/eps": eps})

    def test_fn(epoch, env_step):
        pass
        # NOTE none of this is a thing here because we're using a random policy,
        #policy.set_eps(args.eps_test)

    def save_checkpoint_fn(epoch, env_step, gradient_step):
        # see also: https://pytorch.org/tutorials/beginner/saving_loading_models.html
        ckpt_path_encoder = os.path.join(log_path, 'checkpoint_encoder_epoch_' + str(epoch) + '.pth')
        ckpt_path_decoder = os.path.join(log_path, 'checkpoint_decoder_epoch_' + str(epoch) + '.pth')
        torch.save({'encoder': encoder.state_dict()}, ckpt_path_encoder)
        torch.save({'decoder': decoder.state_dict()}, ckpt_path_decoder)
        return "useless string for a useless return"



    # test train_collector and start filling replay buffer
    #train_collector.collect(n_step=args.batch_size * args.training_num)
    train_collector.collect(n_step=args.buffer_size)
#    buffer.save_hdf5(os.path.join(log_path, 'buffer.h5'))
#    print('buffer = saved to disk')
#    exit()
    # trainer
    # don't need to run it right now
    # just want to check the buffer out
    
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
    


#if __name__ == '__main__':
#    test_dqn(get_args())
