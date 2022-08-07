import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
from typing import Any, Callable, Dict, Optional, Tuple, Union
import kornia.augmentation 

from tianshou.data import Batch, ReplayBuffer, to_numpy, to_torch, to_torch_as
from tianshou.policy import BasePolicy
from tianshou.policy.base import _gae_return, _nstep_return
from RLmaster.latent_representations.autoencoder_nn import CNNEncoderNew, CNNDecoderNew
from RLmaster.policy.dqn_fixed import DQNPolicyFixed

class AutoencoderLatentSpacePolicy(BasePolicy):
    """
    AutoencoderLatentSpace is the autoencoder which learns on image
    reconstruction loss alongside a reinforcement learning policy (any rl alg really).
    Its purpose is to provide a lower dimensional embedding 
    for the reinforcement learning alg.
    It inherits from BasePolicy because that is the most convenient way
    to set up learning within tianshou.
    Furthermore, it takes the reinforcement learning policy as an argument 
    and acts as a wrapper around it (as it should, given that its purpose is to
    stand between the agent and environment).
    This makes parallel learning not only convenient, but also, ideally, 
    enables switching between different reinforcement learning algorithms
    as simple as passing a different parameter (as it should be, given that
    it should act as an independent wrapper).

    :param BasePolicy policy: a base policy to add AutoencoderLatentSpace to.
    :param CNNEncoderNew encoder: the encoder part of the autoencoder.
    :param CNNDecoderNew decoder: the encoder part of the autoencoder.
    :param torch.optim.Optimizer optim_encoder: a torch.optim for optimizing the encoder.
    :param torch.optim.Optimizer optim_decoder: a torch.optim for optimizing the decoder.
    :param float lr_scale: the scaling factor for autoencoder learning.
    """

    def __init__(
        self,
        rl_policy: BasePolicy,
        latent_space_type: str,
        encoder: CNNEncoderNew,
        decoder: CNNDecoderNew,
        optim_encoder: torch.optim.Optimizer,
        optim_decoder: torch.optim.Optimizer,
        reconstruction_criterion,
        batch_size: int,
        frames_stack: int,
        device: str = "cpu",
        squeeze_latent_into_single_vector: bool = True,
        use_reconstruction_loss: bool = True,
        pass_policy_grad_to_encoder: bool = False,
        alternating_training_frequency: int = 1000,
        lr_scale: float = 0.001,
        data_augmentation: bool = True,
        forward_prediction_in_latent: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.rl_policy = rl_policy
        self.latent_space_type = latent_space_type
        # here only for testing
        self.squeeze_latent_into_single_vector = squeeze_latent_into_single_vector
        self.use_reconstruction_loss = use_reconstruction_loss
        self.pass_policy_grad_to_encoder = pass_policy_grad_to_encoder
        self.alternating_training_frequency = alternating_training_frequency
        self.encoder = encoder
        self.decoder = decoder
        self.optim_encoder = optim_encoder
        self.optim_decoder = optim_decoder
        self.reconstruction_criterion = reconstruction_criterion
        self.device = device
        self.batch_size = batch_size
        self.frames_stack = frames_stack
        self.lr_scale = lr_scale
        # you could have an additional randomcrop (80,80) before replicationpad
        self.data_augmentation = data_augmentation
        random_shift = nn.Sequential(nn.ReplicationPad2d(4), 
                kornia.augmentation.RandomCrop((84, 84)))
        self.augmentation = random_shift
        self.forward_prediction_in_latent = forward_prediction_in_latent

    def train(self, mode: bool = True) -> "AutoencoderLatentSpacePolicy":
        """Set the module in training mode."""
        self.rl_policy.train(mode)
        self.training = mode
        self.encoder.train(mode)
        self.decoder.train(mode)
        return self


    def forward(
        self,
        batch: Batch,
        state: Optional[Union[dict, Batch, np.ndarray]] = None,
        input: str = "obs",
        **kwargs: Any,
    ) -> Batch:
        """First embed the observations and then 
        compute action over the given embedded observations with the inner policy.
        This function is called in the collector under torch.no_grad()
        and it's purpose is to select actions when filling to fill the ReplayBuffer.
        The type of embedding depends on latent_space_type.
        Since the various variations are so similar,
        we just pass this parameter and implement them here.

        .. seealso::

            Please refer to :meth:`~tianshou.policy.BasePolicy.forward` for
            more detailed explanation.
            Also check out the forward function of your specific policy
            if you want details on that.
        """
        # so te FINAL SOLUTION is: 
        # save original obs under new key called, say, orig_obs
        # copy-paste collector code into CollectorOnEmbeddedSpace
        # update the self.data batch not with obs=obs, but with obs=obs_orig or whatever
        # but to avoid needing to change policy code it is:
        # here shape the observations to fit the chosen latent space type
        if self.latent_space_type == 'single-frame-predictor':
            # encode each one separately
            obs = batch[input].reshape((-1, 1, 84, 84))
            # and then restack
            #batch.embedded_obs = to_numpy(self.encoder(obs).view(-1, self.frames_stack, 
            # we stack by combining into a single vector
            # because now the first row is linear
            # TODO make it work with cnn layers too
            #batch.embedded_obs = to_numpy(self.encoder(obs).view(-1, 
            batch.embedded_obs = self.encoder(obs).view(-1, 
                self.frames_stack * self.encoder.features_dim)
        else:
        # TODO: write out the other cases (ex. forward prediction)
            obs = batch[input]
            #batch.embedded_obs = to_numpy(self.encoder(obs))
            batch.embedded_obs = self.encoder(obs)
        #batch.obs = to_numpy(embedded_obs)
        #print(batch.orig_obs.shape)
        return self.rl_policy.forward(batch, state, input="embedded_obs", **kwargs)

    def set_eps(self, eps: float) -> None:
        """Set the eps for epsilon-greedy exploration."""
        if hasattr(self.rl_policy, "set_eps"):
            self.rl_policy.set_eps(eps)  # type: ignore
        else:
            raise NotImplementedError()

#    @staticmethod
    def compute_nstep_return(
        self,
        batch: Batch,
        buffer: ReplayBuffer,
        indice: np.ndarray,
        target_q_fn: Callable[[ReplayBuffer, np.ndarray], torch.Tensor],
        gamma: float = 0.99,
        n_step: int = 1,
        rew_norm: bool = False,
    ) -> Batch:
        r"""Compute n-step return for Q-learning targets.

        .. math::
            G_t = \sum_{i = t}^{t + n - 1} \gamma^{i - t}(1 - d_i)r_i +
            \gamma^n (1 - d_{t + n}) Q_{\mathrm{target}}(s_{t + n})

        where :math:`\gamma` is the discount factor, :math:`\gamma \in [0, 1]`,
        :math:`d_t` is the done flag of step :math:`t`.

        :param Batch batch: a data batch, which is equal to buffer[indice].
        :param ReplayBuffer buffer: the data buffer.
        :param function target_q_fn: a function which compute target Q value
            of "obs_next" given data buffer and wanted indices.
        :param float gamma: the discount factor, should be in [0, 1]. Default to 0.99.
        :param int n_step: the number of estimation step, should be an int greater
            than 0. Default to 1.
        :param bool rew_norm: normalize the reward to Normal(0, 1), Default to False.

        :return: a Batch. The result will be stored in batch.returns as a
            torch.Tensor with the same shape as target_q_fn's return tensor.
        """
        assert not rew_norm, \
            "Reward normalization in computing n-step returns is unsupported now."
        rew = buffer.rew
        bsz = len(indice)
        indices = [indice]
        for _ in range(n_step - 1):
            indices.append(buffer.next(indices[-1]))
        indices = np.stack(indices)
        # terminal indicates buffer indexes nstep after 'indice',
        # and are truncated at the end of each episode
        terminal = indices[-1]
        batch_for_computing_returns = buffer[terminal]  # batch.obs_next: s_{t+n}
        #print(type(batch_for_computing_returns))
        #print(batch_for_computing_returns.shape)
        #print(batch_for_computing_returns)
        with torch.no_grad():
            if self.latent_space_type == 'single-frame-predictor':
                batch_for_computing_returns.obs_next = batch_for_computing_returns.obs_next.reshape(
                        (-1, 1, 84, 84))
                #batch_for_computing_returns.obs_next = to_numpy(
                #        self.encoder(batch_for_computing_returns.obs_next).view(-1, 
                #    self.frames_stack * self.encoder.features_dim))
                batch_for_computing_returns.obs_next = self.encoder(batch_for_computing_returns.obs_next).view(-1, 
                    self.frames_stack * self.encoder.features_dim)
            else:
                batch_for_computing_returns.obs_next = self.encoder(batch_for_computing_returns.obs_next).view(-1, 
                    self.encoder.features_dim)

            target_q_torch = target_q_fn(batch_for_computing_returns)  # (bsz, ?)
        target_q = to_numpy(target_q_torch.reshape(bsz, -1))
        target_q = target_q * BasePolicy.value_mask(buffer, terminal).reshape(-1, 1)
        end_flag = buffer.done.copy()
        end_flag[buffer.unfinished_index()] = True
        target_q = _nstep_return(rew, end_flag, target_q, indices, gamma, n_step)

        batch.returns = to_torch_as(target_q, target_q_torch)
        if hasattr(batch, "weight"):  # prio buffer update
            batch.weight = to_torch_as(batch.weight, target_q_torch)
        return batch

    def process_fn(
        self, batch: Batch, buffer: ReplayBuffer, indices: np.ndarray
    ) -> Batch:
        """Pre-process the data from the provided replay buffer.

        Used in :meth:`update`. Check out :ref:`process_fn` for more information.
        """
        # TODO make this work for future prediction too (maybe you want the same aug)
        # NOTE: maybe u need to swith to torch first
        if self.data_augmentation:
            batch.obs = self.augmentation(torch.tensor(batch.obs, dtype=torch.float))
            batch.obs_next = self.augmentation(torch.tensor(batch.obs_next, dtype=torch.float))
        # this ripped from dqn method
        # no, it shouldn't be
        # but we went to disgusting hacking so here we are man
        # do the below depending on underlying policy
        if isinstance(self.rl_policy, DQNPolicyFixed):
            batch = self.compute_nstep_return(
                batch, buffer, indices, self.rl_policy._target_q, self.rl_policy._gamma, 
                self.rl_policy._n_step, self.rl_policy._rew_norm
            )
            return batch
        else:
            # we don't do anything here, just pass it further to inner policy pre-processing
            return self.rl_policy.process_fn(batch, buffer, indices)


    def post_process_fn(
        self, batch: Batch, buffer: ReplayBuffer, indices: np.ndarray
    ) -> None:
        """Post-process the data from the provided replay buffer.

        Typical usage is to update the sampling weight in prioritized
        experience replay. Used in :meth:`update`.
        """
        # we don't do anything here, just pass it further to inner policy post-processing
        self.rl_policy.post_process_fn(batch, buffer, indices)

    def learn(self, batch: Batch, **kwargs: Any) -> Dict[str, float]:
        # pass through encoder with no_grad here
# NOTE: stupid hack for a stupid problem...
        if self.frames_stack == 1:
            batch.obs = to_torch(batch.obs, device=self.device).view(self.batch_size, self.frames_stack, 84, 84)
        else:
            # it's the right shape if frames_stack != 1
            #batch.obs = to_torch(batch.obs, device=self.device, dtype=torch.float)
            batch.obs = torch.tensor(batch.obs, device=self.device, dtype=torch.float)
            # added for rainbow
            #batch.obs_next = to_torch(batch.obs_next, device=self.device, dtype=torch.float)
            batch.obs_next = torch.tensor(batch.obs_next, device=self.device, dtype=torch.float)

        # we zero grad this here in because maybe we want both grads
        self.rl_policy.zero_this_grad()
        # TODO use this to implement forward prediction
        #obs_next = torch.tensor(batch.obs_next, device=self.device)

        if self.pass_policy_grad_to_encoder == False:
            with torch.no_grad():
                if self.latent_space_type == 'single-frame-predictor':
                    # encode each one separately
                    obs = batch.obs.reshape((-1, 1, 84, 84))
                    obs_next = batch.obs_next.reshape((-1, 1, 84, 84))
                    batch.obs = self.encoder(obs).view(-1, 
                        self.frames_stack * self.encoder.features_dim)
                    batch.obs_next = self.encoder(obs_next).view(-1, 
                        self.frames_stack * self.encoder.features_dim)
                else:
                    obs = batch.obs
                    obs_next = batch.obs_next
                    batch.obs = self.encoder(obs)
                    batch.obs_next = self.encoder(obs_next)
        else:
            if self.latent_space_type == 'single-frame-predictor':
                # encode each one separately
                obs = batch.obs.reshape((-1, 1, 84, 84))
                obs_next = batch.obs_next.reshape((-1, 1, 84, 84))
                batch.obs = self.encoder(obs).view(-1, 
                    self.frames_stack * self.encoder.features_dim)
                batch.obs_next = self.encoder(obs_next).view(-1, 
                    self.frames_stack * self.encoder.features_dim)
            else:
                obs = batch.obs
                obs_next = batch.obs_next
                batch.obs = self.encoder(obs)
                batch.obs_next = self.encoder(obs_next)

        # this will also pass q-grads through the encoder if encoder params are given to q_optim
        res = self.rl_policy.learn(batch, **kwargs)

        # just for testing:
        # don't update with reconstruction loss if not we don't pass that
        if not self.use_reconstruction_loss:
            return res

        self.optim_encoder.zero_grad()
        self.optim_decoder.zero_grad()
        encoded_obs = self.encoder(obs)
        if self.latent_space_type == 'forward-frame-predictor':
            decoded_obs = self.decoder(encoded_obs, batch.act)
        else:
            decoded_obs = self.decoder(encoded_obs)


        # batch.obs is of shape (batch_size, frames_stack, 84, 84)
        # decoded_obs is of shape (batch_size, 1, 84, 84) and we want it to learn, say, the last frame only
        # which means we have to somehow correctly slice batch.obs so that only the last frames are left
        # tried it in shell, this worked 
        #reconstruction_loss = self.reconstruction_criterion(decoded_obs, batch.obs[:, -1, :, :].view(-1, 1, 84, 84))

        #if self.latent_space_type == 'single-frame-predictor':
            #reconstruction_loss = self.reconstruction_criterion(decoded_obs, obs[:, -1, :, :].view(-1, 1, 84, 84))
        if self.latent_space_type == 'single-frame-predictor':
            reconstruction_loss = self.reconstruction_criterion(decoded_obs, obs / 255)
        if self.latent_space_type == 'forward-frame-predictor':
            # TODO: delete, this is just a check
            reconstruction_loss = self.reconstruction_criterion(decoded_obs, obs_next / 255)
#        if self.latent_space_type == 'forward-frame-predictor':
#            batch.obs_next = torch.tensor(batch.obs_next, device=self.device)
#            print(batch.obs_next.shape)
#            print(batch.obs_next[:, -1, :, :].shape)
#            reconstruction_loss = self.reconstruction_criterion(decoded_obs, batch.obs_next[:, -1, :, :].view(-1, 1, 84, 84) / 255)
        # L2 penalty on latent representation:
        # this is wrong for single-frame-predictor
        latent_loss = (0.5 * encoded_obs.pow(2).sum(1)).mean()
        # throwing a sample loss in there to see what happens
        loss = reconstruction_loss + latent_loss * 10**-6


        #reconstruction_loss.backward()
        loss.backward()
        self.optim_encoder.step()
        self.optim_decoder.step()
        res.update(
            {
                "loss/autoencoder": loss.item(),
            }
        )
        return res
