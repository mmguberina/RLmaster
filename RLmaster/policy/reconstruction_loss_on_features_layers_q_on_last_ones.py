from typing import Any, Dict, Optional, Union
from copy import deepcopy

import numpy as np
import torch
from tianshou.data import Batch, ReplayBuffer, to_numpy, to_torch_as
from tianshou.policy import BasePolicy

class DQNOnReconstructionEncoderPolicy(BasePolicy):
    """
    there are 2 networks. the first is an autoencoder
    which is trained on image reconstruction loss.
    we use the encoder part of it for feature
    embedding.
    to make this a bit easier,
    the autoencoder is split into the encoder and the decoder portion
    beforehand.
    the other network is a DQN network which takes
    as input the embedded observations, passes them through
    its own FC layers and trains those with expected return
    loss (i.e. does Q-learning on them)
    """

    def __init__(
        self,
        encoder: torch.nn.Module,
        decoder: torch.nn.Module,
        q_network: torch.nn.Module,
        optim_q: torch.optim.Optimizer,
        optim_encoder: torch.optim.Optimizer,
        optim_decoder: torch.optim.Optimizer,
        reconstruction_criterion,
        features_dim: int = 3136,
        discount_factor: float = 0.99,
        estimation_step: int = 1,
        target_update_freq: int = 0,
        reward_normalization: bool = False,
        is_double: bool = True,
        **kwargs: Any,
        ) -> None:
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.q_network = q_network
        self.optim_encoder = optim_encoder
        self.optim_decoder = optim_decoder
        self.optim_q = optim_q
        self.reconstruction_criterion = reconstruction_criterion
        self.eps = 0.0
        assert 0.0 <= discount_factor <= 1.0, "discount has to be in [0,1]"
        self._gamma = discount_factor
        assert estimation_step > 0, "estimation_step has to be greater than 0"
        self._n_step = estimation_step
        self._target = target_update_freq > 0
        self._freq = target_update_freq
        self._iter = 0
        if self._target:
            self.q_network_old = deepcopy(self.q_network)
            self.q_network_old.eval()
        self._rew_norm = reward_normalization
        self._is_double = is_double

    def set_eps(self, eps: float) -> None:
        """ setting the epsilon in epsilon-greedy exploration"""
        self.eps = eps

    def train(self, mode: bool = True) -> "DQNOnReconstructionEncoder":
        """set modules in training mode, expect for the target network"""
        self.training = mode
        self.q_network.train(mode)
        self.encoder.train(mode)
        self.decoder.train(mode)
        # can't say i understand this returns a new instance, but ok
        return self
    
    def sync_weight(self) -> None:
        """sync weight of the target network"""
        self.q_network_old.load_state_dict(self.q_network.state_dict())

    def _target_q(self, buffer: ReplayBuffer, indices: np.array) -> torch.Tensor:
        batch = buffer[indices]
        # batch.obs_next: s_{t+n} 
        # TODO understand why this is necessary - why not just take the next
        # batch, they're supposed to be stored sequentially anyway
        # also
        # this class inherits from nn.module (and ABC, but that's not relevant for this point)
        # and so calling on the instance of the class like instance(arg) is
        # in fact equivalent to instance.forward(arg)
        # note that this has been overwritten in tianshou so that it
        # accepts tianshou.data.batch and a state (which is None unless your
        # network requires a hidden state as well)
        result = self(batch, input="obs_next")
        if self._target:
            # target_Q = Q_old(s_, argmax(Q_new(s_, *)))
            target_q = self(batch, q_network="q_network_old", input="obs_next").logits
        else:
            target_q = result.logits
        if self._is_double:
            return target_q[np.arange(len(result.act)), result.act]
        else:
            return target_q.max(dim=1)[0]

    def process_fn(
        self, batch: Batch, buffer: ReplayBuffer, indices: np.ndarray
    ) -> Batch:
        """compute the n-step return for Q-learning targets."""
        batch = self.compute_nstep_return(
            batch, buffer, indices, self._target_q, self._gamma, self._n_step,
            self._rew_norm
        )
        return batch

    def compute_q_value(
        self, logits: torch.Tensor, mask: Optional[np.ndarray]
    ) -> torch.Tensor:
        """compute the q value based on raw network output and action mask"""
        if mask is not None:
            # the masked q value should be smaller than logits.min() TODO: why?
            # mask is a bool array.
            # interestingly, True == 1 and False == 0 and you can do arithmetic
            # operations on them with no complaints from the interpreter (what a language).
            # also why are we doing this (probably because there is nothing
            # after the last linear layer in the q-network architecture, but why this
            # specifically TODO i don't know)
            min_value = logits.min() - logits.max() - 1.0
            logits = logits + to_torch_as(1 - mask, logits) * min_value
        return logits

    # TODO: why am i passing the q_network here?
    # should i do the same for encoder and decoder then?
    def forward(
        self,
        batch: Batch,
        state: Optional[Union[dict, Batch, np.ndarray]] = None,
        q_network: str = "q_network",
        input: str = "obs",
        **kwargs: Any,
    ) -> Batch:
        """
        compute action over the given batch of data.
        if a mask is required (because say some actions are not available in some state),
        it is to be passed as 
        batch == Batch(
            obs=Batch(
                obs="...original obs... (with batch_size=1 here for simpler notation)",
                mask=np.array([[False, True, False]]),
                # meaning action 1 is available and 0 and 2 are not
            ),
            ...
        )

        returns tiashou.data.Batch with 3 keys:
            - ``act`` - the action
            - ``logits`` - the network's raw output
            - ``state`` - the hidden state (should default to None)

        For this particular policy, we pass the observations through
        the encoder and the pass it's output through both the q_network
        and the decoder. the decoder output is only used to calculate the
        loss for both the encoder and the decoder,
        and the q_network output of course is used as the state-action advantage
        estimate.
        """
        q_network = getattr(self, q_network)
        obs = batch[input]
#        print("well i did get it once")
#        print(batch)
        obs_next = obs.obs if hasattr(obs, "obs") else obs
        embedded_obs = self.encoder(obs_next)
        # TODO these comments below,
        # yeah they didn't work...
        # decoded_obs will be used just to calculate loss
        # in training mode, the forward pass triggers autograd
        # so just from this you have the gradients you're looking for.
        # we need to pass that on so that we can calculate the loss in the learn method.
        # fortunatelly we can just pass it as an additional key-value pair in
        # the batch this method's returning.
        # this also allows for easy rendering later, which is a nice plus.
        #decoded_obs = self.decoder(embedded_obs)
        logits, hidden = q_network(embedded_obs, state=state, info=batch.info)
        q = self.compute_q_value(logits, getattr(obs, "mask", None))
        if not hasattr(self, "max_action_num"):
            self.max_action_num = q.shape[1]
        act = to_numpy(q.max(dim=1)[1])
        #return Batch(logits=logits, act=act, state=hidden, decoded_obs=decoded_obs)
        return Batch(logits=logits, act=act, state=hidden)

    # here the batch is obtained from the replay buffer
    # and this is where you perform learning.
    # in fact, when collecting, torch.no_grad over forward is used
    def learn(self, batch: Batch, **kwargs: Any) -> Dict[str, float]:
        if self._target and self._iter % self._freq == 0:
            self.sync_weight()
        self.optim_encoder.zero_grad()
        self.optim_decoder.zero_grad()
        self.optim_q.zero_grad()
        # pop works 'cos Batch is kinda like an overloaded dictionary
        weight = batch.pop("weight", 1.0)

        # can't use the forward method as it is now to get what i want
        # so i'll hack another version of it right here
        q_network = getattr(self, "q_network")
        # TODO i have no idea what this input is
        # it is not a key, that's for sure
        #obs = batch[input]
        obs = batch.obs
        obs_next = obs.obs if hasattr(obs, "obs") else obs
        embedded_obs = self.encoder(obs_next)
        decoded_obs = self.decoder(embedded_obs)
        logits, hidden = q_network(embedded_obs, state=None, info=batch.info)
        q = self.compute_q_value(logits, getattr(obs, "mask", None))
        if not hasattr(self, "max_action_num"):
            self.max_action_num = q.shape[1]
        act = to_numpy(q.max(dim=1)[1])
        #Batch(logits=logits, act=act, state=hidden)

        #forward_result_batch = self(batch)
        #decoded_obs = forward_result_batch.decoded_obs
        q = logits
        q = q[np.arange(len(q)), batch.act]
        returns = to_torch_as(batch.returns.flatten(), q)
        td_error = returns - q
        q_loss = (td_error.pow(2) * weight).mean()
        batch.weight = td_error
        q_loss.backward()
        self.optim_q.step()

        # reconstruction criteon does not work
        # says:
        #    if target.size() != input.size():
        #    TypeError: 'int' object is not callable
        # the sizes are certainly correct tho....

        print("decoded_obs.shape")
        print(decoded_obs.shape)
        print(decoded_obs)
        print("obs_next.shape")
        print(obs_next.shape)
        print(obs_next)
        reconstruction_loss = self.reconstruction_criterion(decoded_obs, obs_next)
        reconstruction_loss.backward()
        self.optim_encoder.step()
        self.optim_decoder.step()

        self._iter += 1
        # TODO need to change return so that it also given encoder (and decoder?) loss
        return {"loss": loss.item()}

    def exploration_noise(self, act: Union[np.ndarray, Batch],
                            batch: Batch) -> Union[np.ndarray, Batch]:
        if isinstance(act, np.ndarray) and not np.isclose(self.eps, 0.0):
            bsz = len(act)
            rand_mask = np.random.rand(bsz) < self.eps
            q = np.random.rand(bsz, self.max_action_num)
            if hasattr(batch.obs, "mask"):
                q += batch.obs.mask
            rand_act = q.argmax(axis=1)
            act[rand_mask] = rand_act[rand_mask]
        return act
