import numpy as np
import torch
import torch.nn.functional as F
from copy import deepcopy
from typing import Any, Callable, Dict, Optional, Tuple, Union

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
        latent_space_type: str,
        encoder: CNNEncoderNew,
        decoder: CNNEncoderNew,
        optim_encoder: torch.optim.Optimizer,
        optim_decoder: torch.optim.Optimizer,
        reconstruction_criterion,
        batch_size: int,
        frames_stack: int,
        device: str = "cpu",
        lr_scale: float = 0.001,
        pass_policy_grad: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.policy = policy
        self.latent_space_type = latent_space_type
        self.encoder = encoder
        self.decoder = decoder
        self.optim_encoder = optim_encoder
        self.optim_decoder = optim_decoder
        self.reconstruction_criterion = reconstruction_criterion
        self.device = device
        self.batch_size = batch_size
        self.frames_stack = frames_stack
        self.pass_policy_grad = pass_policy_grad
        self.lr_scale = lr_scale

    def train(self, mode: bool = True) -> "AutoencoderLatentSpacePolicy":
        """Set the module in training mode."""
        self.policy.train(mode)
        self.training = mode
        self.encoder.train(mode)
        self.decoder.train(mode)
        return self

    # TODO write this out
    # the point of this function is to act as the preprocessing
    # function in collector.
    # but, crucially, you need to ensure you keep the original observations as well
    def embed_obs(self, obs_orig):
        if self.latent_space_type == 'single-frame-predictor':
            # encode each one separately
            obs = obs_orig.reshape((-1, 1, 84, 84))
            # and then restack
            #batch.embedded_obs = to_numpy(self.encoder(obs).view(-1, self.frames_stack, 
            # we stack by combining into a single vector
            # because now the first row is linear
            # TODO make it work with cnn layers too
            #batch.embedded_obs = to_numpy(self.encoder(obs).view(-1, 
            #    self.frames_stack * self.encoder.n_flatten))
            obs = to_numpy(self.encoder(obs).view(
                -1, self.frames_stack * self.encoder.n_flatten))
        return obs



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
        # just replace the observation in the batch with embedded observations
        # TODO check if this actually works. there's no reason why it shouldn't,
        # but i'm not 100% on this.
        #####################################33
        # TODO
        # it does not work. it manages to fuck with the replay buffer.
        # the replay buffer does not save what it should because of this.
        # it saves the batch to the replay buffer AFTER the forward pass, 
        # which is actually good because you want to store the action as well.
        # POINT IS you can't change the batch.obs here
        # SOLUTION:
        # add embedded obs as an extra key.
        # yes this means you'll need to rewrite every single policy code to use
        # this key instead of the obs key. and thus copy-paste every tianshou policy
        # just to change this 1 line of code. but that's life bro. even though
        # it's copying a lot of policies, it's better than rewriting how collector works imo.
        # or is it????
        # ---> yes it is, better to change 1 file than 10.
        # so te FINAL SOLUTION is: 
        # save original obs under new key called, say, orig_obs
        # copy-paste collector code into CollectorOnEmbeddedSpace
        # update the self.data batch not with obs=obs, but with obs=obs_orig or whatever
        # ----> nvm, you can send a different key to forward
        # -------> so save embedded_obs in batch and make that the input key
        # ==================> yeah, but only in forward. goes to shit in other places. 
        # ====================> TODO find a way to fix this.
        ################################################################################
        ## SCREW EMBEDDING HERE THEN
        ## DO IT IN THE COLLECTOR AS SOON AS A OBS POPS UP
        # this function should be called action_selector because that's what it does
        # and that's the only thing it should do
        ################################################################################
        # TODO QUESTION: does it make sense to store the embedded observations too?
        # certainly not just to have autoencoder learning, but what when the policy is learning?
        # the policy learning will be more stable ....... maybe?
        # the autoencoder shifts with learning, ergo same obs is not the same embedded_obs with 
        # different autoencoders. thus after learning, the samples in the replay buffer are wrong.
        # if you recalculate the embeddings, the chosen action potentially changes.
        # and then what are you learning and updating?
        # maybe this whole approch works better with on-policy algorithms?
        # TODO YOU NEED TO FIND A WAY TO MEASURE EMBEDDING DISTRIBUTIONAL SHIFT
        # YOU NEED TO QUANTIFY CATASTROPHIC FORGETTING
        #batch.orig_obs = deepcopy(batch.obs)
        #print(batch.orig_obs.shape)
        # the below should be:
        #batch.embedded_obs = self.encoder(batch[input])
        # but to avoid needing to change policy code it is:
        # here shape the observations to fit the chosen latent space type
        #batch.obs_orig = deepcopy(batch.obs)
        #if self.latent_space_type == 'single-frame-predictor':
            # encode each one separately
            #obs = batch[input].reshape((-1, 1, 84, 84))
            # and then restack
            #batch.embedded_obs = to_numpy(self.encoder(obs).view(-1, self.frames_stack, 
            # we stack by combining into a single vector
            # because now the first row is linear
            # TODO make it work with cnn layers too
            #batch.embedded_obs = to_numpy(self.encoder(obs).view(-1, 
            #    self.frames_stack * self.encoder.n_flatten))
            #batch.obs = to_numpy(self.encoder(obs).view(-1, 
            #    self.frames_stack * self.encoder.n_flatten))
        #else:
        # TODO: write out the other cases (ex. forward prediction)
            #obs = batch[input]
            #batch.embedded_obs = to_numpy(self.encoder(obs))
        #batch.obs = to_numpy(embedded_obs)
        #print(batch.orig_obs.shape)
        #return self.policy.forward(batch, state, input="embedded_obs", **kwargs)
        return self.policy.forward(batch, state, **kwargs)

    def set_eps(self, eps: float) -> None:
        """Set the eps for epsilon-greedy exploration."""
        if hasattr(self.policy, "set_eps"):
            self.policy.set_eps(eps)  # type: ignore
        else:
            raise NotImplementedError()

# TODO you need to embed obs_next here
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

    def learn(self, batch: Batch, **kwargs: Any) -> Dict[str, float]:
        # pass through encoder with no_grad here
        # TODO don't really know if deepcopy is necessary here given that the array is stored in a batch (dictionary)
        #print("from learn")
        #print("===================================")
        #obs = deepcopy(batch.obs)
        #print(obs.shape)
        #print("===================================")
#        print(batch)
#        print(batch.shape)
# NOTE: stupid hack for a stupid problem...
        if self.frames_stack == 1:
            #batch.obs = to_torch(batch.obs, device=self.device).view(self.batch_size, self.frames_stack, 84, 84)
            batch.obs_orig = to_torch(batch.obs_orig, device=self.device).view(self.batch_size, self.frames_stack, 84, 84)
        else:
            # it's the right shape if frames_stack != 1
            batch.obs = to_torch(batch.obs, device=self.device)
            batch.obs_orig = to_torch(batch.obs_orig, device=self.device)
    # this was already handeled
    # NOTE however, we might want to re-do the embedding here, dunno tbh
    # TODO try both, see if there's a difference
    #    with torch.no_grad():
    #        if self.latent_space_type == 'single-frame-predictor':
    #            # encode each one separately
    #            obs = batch[input].reshape((-1, 1, 84, 84))
    #            # and then restack
    #            batch.embedded_obs = to_numpy(self.encoder(obs).view(-1, self.frames_stack, 
    #                self.encoder.n_flatten))
    #        else:
    #        # TODO: write out the other cases (ex. forward prediction)
    #            obs = batch[input]
    #            batch.embedded_obs = to_numpy(self.encoder(obs))
            #batch.embedded_obs = self.encoder(to_torch(batch.obs).view(32, 1, 84, 84))
        # do policy learn on embedded
        # TODO make this work:
        # 0) ensure you can actually pass input key through learn (not convinced that you can)
        # ---> a) save embedded_obs in replay buffer, use that
        # ---> b) pass the newly embedded (like we did now (make less sense overall intuitively but that says close to nothing))
        #res = self.policy.learn(batch, input="embedded_obs", **kwargs)
        # ---> obs in buffer are now the embedded obs we had before
        # otherwise recompress the batch you took here and pass that
        res = self.policy.learn(batch, **kwargs)

        # and now here pass again through encoder, pass trough decoder and do the update
        self.optim_encoder.zero_grad()
        self.optim_decoder.zero_grad()
        #decoded_obs = self.decoder(self.encoder(batch.obs))
        decoded_obs = self.decoder(self.encoder(obs))
        
#        print("===================================")
#        print("===================================")
#        print(batch.obs)
#        print(type(batch.obs))
#        print(batch.obs.shape)
#        print("===================================")
#        print(decoded_obs)
#        print(type(decoded_obs))
#        print(decoded_obs.shape)
        # batch.obs is of shape (batch_size, frames_stack, 84, 84)
        # decoded_obs is of shape (batch_size, 1, 84, 84) and we want it to learn, say, the last frame only
        # which means we have to somehow correctly slice batch.obs so that only the last frames are left
        # tried it in shell, this worked 
        reconstruction_loss = self.reconstruction_criterion(decoded_obs, batch.obs[:, -1, :, :].view(-1, 1, 84, 84))
        reconstruction_loss.backward()
        self.optim_encoder.step()
        self.optim_decoder.step()
        res.update(
            {
                "loss/autoencoder": reconstruction_loss.item(),
            }
        )
        return res
