import argparse
import os
import pprint

import numpy as np
import torch
from RLmaster.network.atari_network import DQNNoEncoder
from RLmaster.util.atari_wrapper import wrap_deepmind, make_atari_env, make_atari_env_watch
from torch.utils.tensorboard import SummaryWriter
from RLmaster.util.save_load_hyperparameters import save_hyperparameters

from tianshou.data import Collector, VectorReplayBuffer
from tianshou.env import ShmemVectorEnv
# TODO write the autoencoder only policy
#from RLmaster.policy.autoencoder_only import AutoencoderOnly
from RLmaster.policy.random import RandomPolicy
from RLmaster.latent_representations.autoencoder_learning_as_policy_wrapper import AutoencoderLatentSpacePolicy
from RLmaster.latent_representations.autoencoder_nn import CNNEncoderNew, CNNDecoderNew
from RLmaster.util.collector_on_latent import CollectorOnLatent
from tianshou.trainer import offpolicy_trainer
from tianshou.utils import TensorboardLogger
from tianshou.data import Batch, ReplayBuffer, to_numpy, to_torch

"""
action are all random sampled
1 frame (1,84,84) -> pass through autoencoder -> 1 frame (1,84,84)
"""


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='PongNoFrameskip-v4')
    parser.add_argument('--features_dim', type=int, default=3136)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--buffer-size', type=int, default=100000)
#    parser.add_argument('--buffer-size', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.001)
    # TODO understand where exactly this is used and why
    # it's probably how often you update the target policy network in deep-Q
#    parser.add_argument('--target-update-freq', type=int, default=500)
    parser.add_argument('--target-update-freq', type=int, default=5)
    parser.add_argument('--epoch', type=int, default=100)
#    parser.add_argument('--epoch', type=int, default=5)
    parser.add_argument('--step-per-epoch', type=int, default=100000)
    # TODO why 8?
    parser.add_argument('--step-per-collect', type=int, default=8)
    # TODO understand where exactly this is used and why
    # why is this a float?
#    parser.add_argument('--update-per-step', type=float, default=0.1)
    parser.add_argument('--update-per-step', type=float, default=0.6)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--training-num', type=int, default=8)
#    parser.add_argument('--training-num', type=int, default=2)
    # tests aren't necessary as we're free to overfit as much as we want
    # the training domain IS the testing domain
#    parser.add_argument('--test-num', type=int, default=8)
    parser.add_argument('--test-num', type=int, default=1)
    parser.add_argument('--logdir', type=str, default='log')
    parser.add_argument('--log-name', type=str, default='training_preloaded_buffer_fs_1')
    parser.add_argument('--render', type=float, default=0.)
    parser.add_argument(
        '--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu'
    )
    # NOTE: frame stacking needs to be 1 for what we're doing now
    # but let's keep it like a parameter here to avoid unnecessary code
#    parser.add_argument('--frames-stack', type=int, default=2)
    parser.add_argument('--frames-stack', type=int, default=1)
#    parser.add_argument('--frames-stack', type=int, default=4)
    parser.add_argument('--resume-path', type=str, default=None)
    parser.add_argument('--resume-id', type=str, default=None)
    parser.add_argument(
        '--logger',
        type=str,
        default="tensorboard",
        choices=["tensorboard", "wandb"],
    )

    parser.add_argument('--save-buffer-name', type=str, default=None)
    args = parser.parse_args()
    return args



if __name__ == '__main__':
#def test_dqn(args=get_args()):
    args=get_args()
    env = make_atari_env(args)
    # this gives (1,84,84) w/ pixels in 0-1 range, as it should
    args.state_shape = env.observation_space.shape or env.observation_space.n
    args.action_shape = env.action_space.shape or env.action_space.n
    # should be N_FRAMES x H x W
    print("Observations shape:", args.state_shape)
    print("Actions shape:", args.action_shape)
    # make environments
    # seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    # in this experiment we're using the random policy
    # which is just a placeholder really
    rl_policy = RandomPolicy(args.action_shape)
    encoder = CNNEncoderNew(observation_shape=args.state_shape, features_dim=args.features_dim, device=args.device).to(args.device)
    decoder = CNNDecoderNew(observation_shape=args.state_shape, n_flatten=encoder.n_flatten, features_dim=args.features_dim).to(args.device)
    optim_encoder = torch.optim.Adam(encoder.parameters(), lr=args.lr)
    optim_decoder = torch.optim.Adam(decoder.parameters(), lr=args.lr)
    reconstruction_criterion = torch.nn.BCELoss()
    # the rl_policy is then passed into our autoencoder-wrapper policy
    # it's done this way because the compression to latent spaces
    # comes before using the rl policy.
    policy = AutoencoderLatentSpacePolicy(
        rl_policy,
        encoder,
        decoder,
        optim_encoder,
        optim_decoder,
        reconstruction_criterion,
        args.batch_size,
        args.frames_stack,
        args.device
    )

    # load filled buffer
    # we're doing this only for faster debugging, otherwise 
    # we'd just fill a new one with collector quite fast
    log_path_buffer = "../../experiments/latent_only/log/PongNoFrameskip-v4/unlabelled_experiment/"
    buffer_path = os.path.join(log_path_buffer, "buffer.h5")
    buffer = VectorReplayBuffer.load_hdf5(buffer_path)

    buffer._size = args.buffer_size
    buffer.stack_num = args.frames_stack

    save_hyperparameters(args)

    
    log_path = os.path.join(args.logdir, args.task, args.log_name)
    for epoch in range(1, args.epoch + 1):
        if epoch % 10 == 0:
            print("did", epoch, "epochs")
            ckpt_path_encoder = os.path.join(log_path, 'checkpoint_encoder_epoch_' + str(epoch) + '.pth')
            ckpt_path_decoder = os.path.join(log_path, 'checkpoint_decoder_epoch_' + str(epoch) + '.pth')
            torch.save(encoder.state_dict(), ckpt_path_encoder)
            torch.save(decoder.state_dict(), ckpt_path_decoder)

        train_loss = 0.0
        policy.train()
        losses_epoch = 0
        # TODO makes this works
        for i in range(args.buffer_size // args.batch_size):
#            losses = policy.update(args.batch_size, buffer)
#            losses_epoch += losses["loss/autoencoder"]
#        print("loss at epoch", epoch, " = ", losses_epoch)

            # but it doesn't work ,so let's go line by line
            # policy.update
            batch, indices = buffer.sample(args.batch_size)
            policy.updating = True
            batch = policy.process_fn(batch, buffer, indices) # just returns batch
            # policy.learn
            if policy.frames_stack == 1:
                #batch.obs = to_torch(batch.obs, device=policy.device).view(policy.batch_size, policy.frames_stack, 84, 84)
                batch.obs = torch.tensor(batch.obs, device=policy.device).view(policy.batch_size, policy.frames_stack, 84, 84)
            else:
                #batch.obs = to_torch(batch.obs, device=policy.device)
                batch.obs = torch.tensor(batch.obs, device=policy.device)
                # this does nothing to the autoencoder 
    #        with torch.no_grad():
    #            batch.embedded_obs = self.encoder(batch.obs)
    # this policy is the rl policy
    #        res = self.policy.learn(batch, input="embedded_obs", **kwargs)

            # and now here pass again through encoder, pass trough decoder and do the update
            policy.optim_encoder.zero_grad()
            policy.optim_decoder.zero_grad()
            decoded_obs = policy.decoder(policy.encoder(batch.obs))
            # this line works in shell, but maybe i did something wrong there?
            reconstruction_loss = policy.reconstruction_criterion(decoded_obs, batch.obs[:, -1, :, :].view(-1, 1, 84, 84))
            reconstruction_loss.backward()
            policy.optim_encoder.step()
            policy.optim_decoder.step()
            losses_epoch += reconstruction_loss.item()
            # policy.update
            policy.post_process_fn(batch, buffer, indices) # does weighting, which is = 1 
            policy.updating = False
        print("loss at epoch", epoch, " = ", losses_epoch / (args.buffer_size // args.batch_size))

                #samples, indeces = buffer.sample(args.batch_size)
                #images = images.to(device)
                #optimizer_encoder.zero_grad()
                #optimizer_decoder.zero_grad()
                #encoded = encoder(images)
                #decoded = decoder(encoded)
                #loss = criterion(decoded, images)
                #loss.backward()
                #optimizer_encoder.step()
                #optimizer_decoder.step()
                #train_loss += loss.item() * images.size(0)
