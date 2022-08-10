
import argparse
import os
import pprint

import numpy as np
import torch
from RLmaster.network.atari_network import DQNNoEncoder, RainbowNoConvLayers
from RLmaster.util.atari_wrapper import wrap_deepmind, make_atari_env, make_atari_env_watch
from torch.utils.tensorboard import SummaryWriter
from RLmaster.util.save_load_hyperparameters import save_hyperparameters

from tianshou.data import Collector, VectorReplayBuffer, PrioritizedVectorReplayBuffer
from tianshou.env import ShmemVectorEnv
from RLmaster.policy.dqn_fixed import DQNPolicy
#from tianshou.policy import RainbowPolicy
from RLmaster.policy.dqn_fixed import RainbowPolicyFixed
from RLmaster.latent_representations.autoencoder_learning_as_policy_wrapper import AutoencoderLatentSpacePolicy
from RLmaster.latent_representations.autoencoder_nn import RAE_ENC, RAE_DEC, CNNEncoderNew, CNNDecoderNew
from RLmaster.util.collector_on_latent import CollectorOnLatent
from tianshou.trainer import offpolicy_trainer
from tianshou.utils import TensorboardLogger

def get_args():
    parser = argparse.ArgumentParser()
    #parser.add_argument('--task', type=str, default='PongNoFrameskip-v4')
    parser.add_argument('--task', type=str, default='SeaquestNoFrameskip-v4')
    parser.add_argument('--latent-space-type', type=str, default='single-frame-predictor')
    #parser.add_argument('--use-reconstruction-loss', type=int, default=True)
    parser.add_argument('--use-reconstruction-loss', type=int, default=False)
    parser.add_argument('--squeeze-latent-into-single-vector', type=bool, default=True)
    parser.add_argument('--use-pretrained', type=int, default=False)
    parser.add_argument('--pass-q-grads-to-encoder', type=bool, default=True)
    parser.add_argument('--data-augmentation', type=bool, default=True)
    # TODO implement this lel
    parser.add_argument('--forward-prediction-in-latent', type=bool, default=False)
    # TODO implement
    parser.add_argument('--alternating-training-frequency', type=int, default=1)
    parser.add_argument('--features-dim', type=int, default=50)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument("--scale-obs", type=int, default=0)
    parser.add_argument('--eps-test', type=float, default=0.005)
    parser.add_argument('--eps-train', type=float, default=1.)
    parser.add_argument('--eps-train-final', type=float, default=0.05)
    parser.add_argument('--buffer-size', type=int, default=100000)
    parser.add_argument('--lr-rl', type=float, default=0.0000625)
    parser.add_argument('--lr-unsupervised', type=float, default=0.001)
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
    parser.add_argument('--target-update-freq', type=int, default=500)
    parser.add_argument('--epoch', type=int, default=50)
#    parser.add_argument('--epoch', type=int, default=5)
    parser.add_argument('--step-per-epoch', type=int, default=100000)
#    parser.add_argument('--step-per-epoch', type=int, default=100)
    # TODO why 8?
    #parser.add_argument('--step-per-collect', type=int, default=12)
    parser.add_argument('--step-per-collect', type=int, default=10)
    # TODO having a different update frequency for the autoencoder 
    # and the policy is probably a smart thing to do
    parser.add_argument('--update-per-step', type=float, default=0.1)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--training-num', type=int, default=10)
    #parser.add_argument('--training-num', type=int, default=6)
    #parser.add_argument('--test-num', type=int, default=8)
    parser.add_argument('--test-num', type=int, default=10)
    parser.add_argument('--logdir', type=str, default='log')
    parser.add_argument('--log-name', type=str, default='raibow_only')
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



#def test_dqn(args=get_args()):
if __name__ == "__main__": 
#    torch.set_num_threads(1)
    args=get_args()
    print(args.task)
    #env = make_atari_env(args)
    # we have another way now, should be faster
    #env, train_envs, test_envs = make_atari_env(
    train_envs, test_envs = make_atari_env(
        args.task,
        args.seed,
        args.training_num,
        args.test_num,
        scale=args.scale_obs,
        frame_stack=args.frames_stack,
    )
    # this gives (frames_stack,84,84) w/ pixels in 0-255 range
    #args.state_shape = env.observation_space.shape or env.observation_space.n
    #args.action_shape = env.action_space.shape or env.action_space.n
    args.state_shape = train_envs.observation_space.shape or train_envs.observation_space.n
    args.action_shape = train_envs.action_space.shape or train_envs.action_space.n
    # should be N_FRAMES x H x W
    print("Observations shape:", args.state_shape)
    print("Actions shape:", args.action_shape)
    # seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    #train_envs.seed(args.seed)
    #test_envs.seed(args.seed)
    #rainbow_net = DQNNoEncoder(args.action_shape, args.frames_stack, args.device).to(args.device)
    if not args.squeeze_latent_into_single_vector:
        # in this case, we don't pass the stacked frames.
        # we unstack them, compress them, the stack the compressed ones and
        # pass that to the policy
        observation_shape = list(args.state_shape)
        observation_shape[0] = 1 
        observation_shape = tuple(observation_shape)
        rl_input_dim = args.features_dim * args.frames_stack
    else:
        rl_input_dim = args.features_dim 
        # in this case, we pass the stacked frames.
        observation_shape = args.state_shape

    # TODO unify this somehow
    # TODO add the 3rd network
    if args.features_dim == 50:
        encoder = RAE_ENC(args.device, observation_shape, args.features_dim).to(args.device)
        decoder = RAE_DEC(args.device, observation_shape, args.features_dim).to(args.device)
    if args.features_dim == 3136:
        encoder = CNNEncoderNew(observation_shape=observation_shape, 
                features_dim=args.features_dim, device=args.device).to(args.device)
        decoder = CNNDecoderNew(observation_shape=observation_shape, 
                n_flatten=encoder.n_flatten, features_dim=args.features_dim).to(args.device)
        #print("encoder.n_flatten")
        #print(encoder.n_flatten)
    if args.use_pretrained:
        encoder_name = "checkpoint_encoder_epoch_2.pth"
        decoder_name = "checkpoint_decoder_epoch_2.pth"
        log_path = "./log/PongNoFrameskip-v4/raibow_rae_all_fast/"
        encoder.load_state_dict(torch.load(log_path + encoder_name)['encoder'])
        decoder.load_state_dict(torch.load(log_path + decoder_name)['decoder'])

    optim_encoder = torch.optim.Adam(encoder.parameters(), lr=args.lr_unsupervised)
    optim_decoder = torch.optim.Adam(decoder.parameters(), lr=args.lr_unsupervised, weight_decay=10**-7)
    reconstruction_criterion = torch.nn.MSELoss()

    # TODO FINISH FIX
    rainbow_net = RainbowNoConvLayers(args.action_shape, 
                                 args.num_atoms,
                                 args.noisy_std,
                                 args.device,
                                 not args.no_dueling,
                                 args.noisy_std,
                                 rl_input_dim)#.to(args.device)

    if args.pass_q_grads_to_encoder == False:
        optim_q = torch.optim.Adam(rainbow_net.parameters(), lr=args.lr_rl)
    else:
        optim_q = torch.optim.Adam([{'params': rainbow_net.parameters()}, 
                {'params': encoder.parameters()}], lr=args.lr_rl)

    #rl_policy = RainbowPolicy(
    rl_policy = RainbowPolicyFixed(
        rainbow_net,
        optim_q,
        args.gamma,
        args.num_atoms,
        args.v_min,
        args.v_max,
        args.n_step,
        target_update_freq=args.target_update_freq
    ).to(args.device)
#    print(rl_policy.__class__)
#    if isinstance(rl_policy, DQNPolicy):
#        print("i4t is")
#    print(type(rl_policy.__class__))
#    exit()
#    no_nex()
    # the rl_policy is then passed into our autoencoder-wrapper policy
    # it's done this way because the compression to latent spaces
    # comes before using the rl policy.
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
        args.forward_prediction_in_latent
    )
    # TODO write this out
    if args.resume_path:
        policy.load_state_dict(torch.load(args.resume_path, map_location=args.device))
        print("Loaded agent from: ", args.resume_path)
    # replay buffer: `save_last_obs` and `stack_num` can be removed together
    # when you have enough RAM
    if args.no_priority:
        buffer = VectorReplayBuffer(
            args.buffer_size,
            buffer_num=len(train_envs),
            ignore_obs_next=True,
            save_only_last_obs=True,
            stack_num=args.frames_stack
        )
    else:
        buffer = PrioritizedVectorReplayBuffer(
            args.buffer_size,
            buffer_num=len(train_envs),
            ignore_obs_next=True,
            save_only_last_obs=True,
            stack_num=args.frames_stack,
            alpha=args.alpha,
            beta=args.beta,
            weight_norm=not args.no_weight_norm
        )
    # collector
    #train_collector = CollectorOnLatent(policy, train_envs, buffer, exploration_noise=True)
    #test_collector = CollectorOnLatent(policy, test_envs, exploration_noise=True)
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
        torch.save(rainbow_net.state_dict(), os.path.join(log_path, 'rainbow_net.pth'))

    def stop_fn(mean_rewards):
        if train_envs.spec.reward_threshold:
            return mean_rewards >= train_envs.spec.reward_threshold
        elif 'Pong' in args.task:
            return mean_rewards >= 20
        else:
            return False

    # nature DQN setting, linear decay in the first 1M steps
    # TODO why the fuck is there a pass here ????????????????
    # it was uncommented. and this is all clearly very important
    def train_fn(epoch, env_step):
        #pass
        if env_step <= 1e6:
            eps = args.eps_train - env_step / 1e6 * \
                (args.eps_train - args.eps_train_final)
        else:
            eps = args.eps_train_final
        policy.set_eps(eps)
        if env_step % 1000 == 0:
            logger.write("train/env_step", env_step, {"train/eps": eps})
        if not args.no_priority:
            if env_step <= args.beta_anneal_step:
                beta = args.beta - env_step / args.beta_anneal_step * \
                        (args.beta - args.beta_final)
            else:
                beta = args.beta_final
            buffer.set_beta(beta)
            if env_step % 1000 == 0:
                logger.write("train/env_step", env_step, {"train/beta": beta})

    def test_fn(epoch, env_step):
        policy.set_eps(args.eps_test)

    def save_checkpoint_fn(epoch, env_step, gradient_step):
        # see also: https://pytorch.org/tutorials/beginner/saving_loading_models.html
        ckpt_path_encoder = os.path.join(log_path, 'checkpoint_encoder_epoch_' + str(epoch) + '.pth')
        ckpt_path_decoder = os.path.join(log_path, 'checkpoint_decoder_epoch_' + str(epoch) + '.pth')
        ckpt_path_rainbow_net = os.path.join(log_path, 'checkpoint_rainbow_net_epoch_' + str(epoch) + '.pth')
        torch.save({'encoder': encoder.state_dict()}, ckpt_path_encoder)
        torch.save({'decoder': decoder.state_dict()}, ckpt_path_decoder)
        torch.save({'rainbow_net': decoder.state_dict()}, ckpt_path_rainbow_net)
        return "useless string for a useless return"

    # watch agent's performance
    # TODO write this or decide to delete it
#    def watch():
#        print("Setup test envs ...")
#        test_envs.seed(args.seed)
#        if args.save_buffer_name:
#            print(f"Generate buffer with size {args.buffer_size}")
#            buffer = VectorReplayBuffer(
#                args.buffer_size,
#                buffer_num=len(test_envs),
#                ignore_obs_next=True,
#                save_only_last_obs=True,
#                stack_num=args.frames_stack
#            )
#            collector = Collector(policy, test_envs, buffer, exploration_noise=True)
#            result = collector.collect(n_step=args.buffer_size)
#            print(f"Save buffer into {args.save_buffer_name}")
#            # Unfortunately, pickle will cause oom with 1M buffer size
#            buffer.save_hdf5(args.save_buffer_name)
#        else:
#            print("Testing agent ...")
#            test_collector.reset()
#            result = test_collector.collect(
#                n_episode=args.test_num, render=args.render
#            )
#        rew = result["rews"].mean()
#        print(f'Mean reward (over {result["n/ep"]} episodes): {rew}')

# TODO write the watch function
# basically run some functions from visualize
# or don't 'cos you'll be cloud training, whatever
#    if args.watch:
#        watch()
#        exit(0)


    # test train_collector and start filling replay buffer
    train_collector.collect(n_step=args.batch_size * args.training_num)
    #train_collector.collect(n_step=args.buffer_size // 3)
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
#    watch()
    


#if __name__ == '__main__':
#    test_dqn(get_args())
