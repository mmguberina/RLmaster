import numpy as np
import torch
import torch.nn.functional as F
from copy import deepcopy

from tianshou.data import Batch, ReplayBuffer, to_numpy, to_torch
from tianshou.policy import BasePolicy
from RLmaster.latent_representations.autoencoder_nn import CNNEncoderNew, CNNDecoderNew

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
        policy: BasePolicy,
        encoder: CNNEncoderNew,
        decoder: CNNEncoderNew,
        optim_encoder: torch.optim.Optimizer,
        optim_decoder: torch.optim.Optimizer,
        lr_scale: float,
        pass_policy_grad: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.policy = policy
        self.encoder = encoder
        self.decoder = decoder
        self.optim_encoder = optim_encoder
        self.optim_decoder = optim_decoder
        self.pass_policy_grad = pass_policy_grad
        self.lr_scale = lr_scale

    def train(self, mode: bool = True) -> "AutoencoderLatentSpacePolicy":
        """Set the module in training mode."""
        self.policy.train(mode)
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

        .. seealso::

            Please refer to :meth:`~tianshou.policy.BasePolicy.forward` for
            more detailed explanation.
            Also check out the forward function of your specific policy
            if you want details on that.
        """
        # just replace the observation in the batch with embedded observations
        # TODO check if this actually works. there's no reason why it shouldn't,
        # but i'm not 100% on this.
        batch[input] = self.encoder(batch[input])
        return self.policy.forward(batch, state, **kwargs)

    def set_eps(self, eps: float) -> None:
        """Set the eps for epsilon-greedy exploration."""
        if hasattr(self.policy, "set_eps"):
            self.policy.set_eps(eps)  # type: ignore
        else:
            raise NotImplementedError()


    def process_fn(
        self, batch: Batch, buffer: ReplayBuffer, indices: np.ndarray
    ) -> Batch:
        """Pre-process the data from the provided replay buffer.

        Used in :meth:`update`. Check out :ref:`process_fn` for more information.
        """
        # we don't do anything here, just pass it further to inner policy pre-processing
        return self.policy.process_fn(batch, buffer, indices)


    def post_process_fn(
        self, batch: Batch, buffer: ReplayBuffer, indices: np.ndarray
    ) -> None:
        """Post-process the data from the provided replay buffer.

        Typical usage is to update the sampling weight in prioritized
        experience replay. Used in :meth:`update`.
        """
        # we don't do anything here, just pass it further to inner policy post-processing
        self.policy.post_process_fn(batch, buffer, indices)

# TODO rewrite this so that it works for the current usecase    
    def learn(self, batch: Batch, **kwargs: Any) -> Dict[str, float]:
        # pass through encoder with no_grad here
        obs = deepcopy(batch.obs)
        with torch.no_grad():
            batch.obs = self.encoder(batch.obs)
        #FFNINSINSIFH THIS TIMORROWWWWWWWWWWWWww
            

        # then do this policy learn thing
        res = self.policy.learn(batch, **kwargs)
        # and now here pass again through necoder, pass trough decoder and do the update
        self.optim.zero_grad()
        act_hat = batch.policy.act_hat
        act = to_torch(batch.act, dtype=torch.long, device=act_hat.device)
        inverse_loss = F.cross_entropy(act_hat, act).mean()  # type: ignore
        forward_loss = batch.policy.mse_loss.mean()
        loss = (
            (1 - self.forward_loss_weight) * inverse_loss +
            self.forward_loss_weight * forward_loss
        ) * self.lr_scale
        loss.backward()
        self.optim.step()
        res.update(
            {
                "loss/icm": loss.item(),
                "loss/icm/forward": forward_loss.item(),
                "loss/icm/inverse": inverse_loss.item()
            }
        )
        return res
