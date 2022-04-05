import torch
import torch.nn as nn
from torchvision import transforms
from ..util.atari_dataset import AtariImageDataset
import matplotlib.pyplot as plt
import numpy as np

# taken from stablebaselines3
# but everyone uses this anyway
# make it usable
class CNNEncoder(nn.Module):

    def __init__(self, observation_shape, 
            features_dim: int = 3136):
        super(CNNEncoder, self).__init__()
        assert features_dim > 0
        assert len(observation_shape) == 3
        #self._observation_space = observation_space
        self.observation_shape = observation_shape
        # TODO this makes sense only if there's a linear layer afterward
        # and there won't be one for now
        # i.e. this will be overwritten
        self._features_dim = features_dim

        # it is assumed that the obversations went through WarpFrame 
        # environment wrapper first
        # so 1xHxW images (channels first) are assumed
        n_input_channels = self.observation_shape[0]
        if self._features_dim == 3136:
            self.encoder = nn.Sequential(
                nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=0),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
                nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
                nn.ReLU(),
                nn.Flatten(),
            )

        if self._features_dim == 576:
            self.encoder = nn.Sequential(
                nn.Conv2d(n_input_channels, 16, kernel_size=8, stride=4, padding=0),
                nn.ReLU(),
                nn.Conv2d(16, 32, kernel_size=8, stride=2, padding=0),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=4, stride=1, padding=0),
                nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=2, stride=1, padding=0),
                nn.ReLU(),
                nn.Flatten(),
            )



        # Compute shape by doing one forward pass
        with torch.no_grad():
            n_flatten = np.prod(self.encoder(torch.zeros(1, observation_shape[0],
                        observation_shape[1], observation_shape[2])).shape[1:])
        self.n_flatten = n_flatten
#        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
#        return self.linear(self.encoder(observations))
        return self.encoder(observations)

class CNNDecoder(nn.Module):
    def __init__(self, observation_shape, n_flatten, features_dim: int = 3136):
        super(CNNDecoder, self).__init__()
        self.observation_shape = observation_shape
        self._features_dim = features_dim
        n_input_channels = self.observation_shape[0]

#        self.linear = nn.Sequential(
#            nn.Linear(features_dim, n_flatten),
#            nn.ReLU(),
#        )

        if self._features_dim == 3136:
            self.deconv = nn.Sequential(
                nn.ConvTranspose2d(64, 64, kernel_size=3, stride=1, padding=0),
                nn.ReLU(),
                nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=0),
                nn.ReLU(),
                nn.ConvTranspose2d(32, n_input_channels, kernel_size=8, stride=4, padding=0),
                nn.Sigmoid(),
            )

        if self._features_dim == 576:
            self.deconv = nn.Sequential(
                nn.ConvTranspose2d(64, 64, kernel_size=2, stride=1, padding=0),
                nn.ReLU(),
                nn.ConvTranspose2d(64, 32, kernel_size=4, stride=1, padding=0),
                nn.ReLU(),
                nn.ConvTranspose2d(32, 16, kernel_size=8, stride=2, padding=0),
                nn.ReLU(),
                nn.ConvTranspose2d(16, n_input_channels, kernel_size=8, stride=4, padding=0),
                nn.Sigmoid(),
            )

    def forward(self, latent_observations: torch.Tensor) -> torch.Tensor:
#        after_lin = self.linear(latent_observations)
        if self._features_dim == 3136:
            deconv = latent_observations.view(-1, 64, 7, 7)

        if self._features_dim == 576:
            deconv = latent_observations.view(-1, 64, 3, 3)
        obs = self.deconv(deconv)
#        print(obs.shape)
        return obs



