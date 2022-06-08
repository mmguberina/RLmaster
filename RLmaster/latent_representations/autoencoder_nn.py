import torch
import torch.nn as nn
from tianshou.data import to_torch
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
        self.features_dim = features_dim

        # it is assumed that the obversations went through WarpFrame 
        # environment wrapper first
        # so 1xHxW images (channels first) are assumed
        n_input_channels = self.observation_shape[0]
        if self.features_dim == 3136:
            self.encoder = nn.Sequential(
                nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=0),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
                nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
                nn.ReLU(),
                nn.Flatten(),
            )

        if self.features_dim == 576:
            self.encoder = nn.Sequential(
                nn.Conv2d(n_input_channels, 64, kernel_size=8, stride=4, padding=0),
                nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=8, stride=2, padding=0),
                nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=4, stride=1, padding=0),
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
        self.features_dim = features_dim
        n_input_channels = self.observation_shape[0]

#        self.linear = nn.Sequential(
#            nn.Linear(features_dim, n_flatten),
#            nn.ReLU(),
#        )

        if self.features_dim == 3136:
            self.deconv = nn.Sequential(
                nn.ConvTranspose2d(64, 64, kernel_size=3, stride=1, padding=0),
                nn.ReLU(),
                nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=0),
                nn.ReLU(),
                nn.ConvTranspose2d(32, n_input_channels, kernel_size=8, stride=4, padding=0),
                nn.Sigmoid(),
            )

        if self.features_dim == 576:
            self.deconv = nn.Sequential(
                nn.ConvTranspose2d(64, 64, kernel_size=2, stride=1, padding=0),
                nn.ReLU(),
                nn.ConvTranspose2d(64, 64, kernel_size=4, stride=1, padding=0),
                nn.ReLU(),
                nn.ConvTranspose2d(64, 64, kernel_size=8, stride=2, padding=0),
                nn.ReLU(),
                nn.ConvTranspose2d(64, n_input_channels, kernel_size=8, stride=4, padding=0),
                nn.Sigmoid(),
            )

    def forward(self, latent_observations: torch.Tensor) -> torch.Tensor:
#        after_lin = self.linear(latent_observations)
        if self.features_dim == 3136:
            deconv = latent_observations.view(-1, 64, 7, 7)

        if self.features_dim == 576:
            deconv = latent_observations.view(-1, 64, 3, 3)
        obs = self.deconv(deconv)
#        print(obs.shape)
        return obs


# need new ones if i want change and to be able to load old ones
class CNNEncoderNew(nn.Module):
    def __init__(self, observation_shape, device,
            features_dim: int = 3136):
        super(CNNEncoderNew, self).__init__()
        assert features_dim > 0
        assert len(observation_shape) == 3
        #self._observation_space = observation_space
        self.observation_shape = observation_shape
        self.device = device
        # TODO this makes sense only if there's a linear layer afterward
        # and there won't be one for now
        # i.e. this will be overwritten
        self.features_dim = features_dim

        # it is assumed that the obversations went through WarpFrame 
        # environment wrapper first
        # so 1xHxW images (channels first) are assumed
        n_input_channels = self.observation_shape[0]
        if self.features_dim == 3136:
            self.encoder = nn.Sequential(
                nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=0),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
                nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
                nn.ReLU(),
                nn.Flatten(),
            )

        if self.features_dim == 576:
            self.encoder = nn.Sequential(
                nn.Conv2d(n_input_channels, 64, kernel_size=8, stride=4, padding=0),
                nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=8, stride=2, padding=0),
                nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=4, stride=1, padding=0),
                nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=2, stride=1, padding=0),
                nn.ReLU(),
                nn.Flatten(),
            )

        # Compute shape by doing one forward pass
        # this is useless as this is the input lel
        with torch.no_grad():
            n_flatten = np.prod(self.encoder(torch.zeros(1, observation_shape[0],
                        observation_shape[1], observation_shape[2])).shape[1:])
        self.n_flatten = n_flatten
#        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

# TODO change this to be convert via torch.data.to_tensor
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        # the / 255 is now necessary because we switched to envpool and it does not downscale
        observations = torch.tensor(observations, dtype=torch.float32, device=self.device) / 255
#        return self.linear(self.encoder(observations))
        return self.encoder(observations)

class CNNDecoderNew(nn.Module):
    def __init__(self, observation_shape, n_flatten, features_dim: int = 3136):
        super(CNNDecoderNew, self).__init__()
        self.observation_shape = observation_shape
        self.features_dim = features_dim
        n_input_channels = self.observation_shape[0]

#        self.linear = nn.Sequential(
#            nn.Linear(features_dim, n_flatten),
#            nn.ReLU(),
#        )
        # this always returns a single frame, as it should
        if self.features_dim == 3136:
            self.deconv = nn.Sequential(
                nn.ConvTranspose2d(64, 64, kernel_size=3, stride=1, padding=0),
                nn.ReLU(),
                nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=0),
                nn.ReLU(),
                nn.ConvTranspose2d(32, 1, kernel_size=8, stride=4, padding=0),
                nn.Sigmoid(),
            )

        # last layer needs to give 1 frame back, regardless of the stacking
        if self.features_dim == 576:
            self.deconv = nn.Sequential(
                nn.ConvTranspose2d(64, 64, kernel_size=2, stride=1, padding=0),
                nn.ReLU(),
                nn.ConvTranspose2d(64, 64, kernel_size=4, stride=1, padding=0),
                nn.ReLU(),
                nn.ConvTranspose2d(64, 64, kernel_size=8, stride=2, padding=0),
                nn.ReLU(),
                nn.ConvTranspose2d(64, 1, kernel_size=8, stride=4, padding=0),
                nn.Sigmoid(),
            )

    def forward(self, latent_observations: torch.Tensor) -> torch.Tensor:
#        after_lin = self.linear(latent_observations)
        if self.features_dim == 3136:
            deconv = latent_observations.view(-1, 64, 7, 7)

        if self.features_dim == 576:
            deconv = latent_observations.view(-1, 64, 3, 3)
        obs = self.deconv(deconv)
#        print(obs.shape)
        return obs

class RAE_ENC(nn.Module):
    def __init__(self, device, observation_shape, features_dim, num_layers=4, num_filters=32):
        super().__init__()
        assert len(observation_shape) == 3
        self.device = device
        self.features_dim = features_dim
        self.num_layers = num_layers
        self.conv_layers = nn.ModuleList(
            [nn.Conv2d(observation_shape[0], num_filters, 3, stride=2)])
        for i in range(num_layers -1):
            self.conv_layers.append(nn.Conv2d(num_filters, num_filters, 3, stride=1))
        out_dim = 35
        self.fc = nn.Linear(num_filters * out_dim * out_dim, self.features_dim)
        self.ln = nn.LayerNorm(self.features_dim)

    def reparametrize(self, mu, logstd):
        std = torch.exp(logstd)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward_conv(self, obs):
        obs = torch.tensor(obs, dtype=torch.float32, device=self.device) / 255
        #obs = to_torch(obs, device=self.device, dtype=torch.float) / 255.
        conv = torch.relu(self.conv_layers[0](obs))
        for i in range(1, self.num_layers):
            conv = torch.relu(self.conv_layers[i](conv))
        h = conv.view(conv.size(0), -1)
        return h

    def forward(self, obs, detach=False):
        h = self.forward_conv(obs)
        if detach:
            h = h.detach()

        h = self.fc(h)
        h = self.ln(h)
        h = torch.tanh(h)

        return h

class RAE_DEC(nn.Module):
    def __init__(self, device, observation_shape, features_dim, num_layers=4, num_filters=32):
        super().__init__()
        self.device = device
        self.features_dim = features_dim
        self.num_layers = num_layers
        self.num_filters = num_filters
        self.out_dim = 35

        self.fc = nn.Linear(features_dim, num_filters * self.out_dim * self.out_dim)
        self.deconv_layers = nn.ModuleList()
        for i in range(self.num_layers - 1):
            self.deconv_layers.append(nn.ConvTranspose2d(num_filters, num_filters, 3, stride=1))
        self.deconv_layers.append(
             nn.ConvTranspose2d(num_filters, observation_shape[0], 3, stride=2, output_padding=1))

    def forward(self, h):
        h = torch.relu(self.fc(h))
        h = h.view(-1, self.num_filters, self.out_dim, self.out_dim)
        for i in range(self.num_layers - 1):
            h = torch.relu(self.deconv_layers[i](h))
        obs = self.deconv_layers[-1](h)
        return obs

# TODO finish this maybe
class RAE_predictive_DEC(nn.Module):
    def __init__(self, device, observation_shape, features_dim, frames_stack, num_layers=4, num_filters=32):
        super().__init__()
        self.device = device
        self.features_dim = features_dim
        self.num_layers = num_layers
        self.num_filters = num_filters
        self.frames_stack = frames_stack
        self.out_dim = 35

        # the + frames_stack -1  is for actions
        self.fc = nn.Linear(features_dim + 1, num_filters * self.out_dim * self.out_dim)
        self.deconv_layers = nn.ModuleList()
        for i in range(self.num_layers - 1):
            self.deconv_layers.append(nn.ConvTranspose2d(num_filters, num_filters, 3, stride=1))
        self.deconv_layers.append(
             nn.ConvTranspose2d(num_filters, observation_shape[0], 3, stride=2, output_padding=1))

    def forward(self, h, a):
        # frames_stack is the same as action repeat
        act_stacked = torch.cat((torch.tensor(a, device=self.device, dtype=torch.float),) * self.frames_stack, device=self.device)
        h = torch.cat((h, act_stacked.view(-1,1)), dim=1, device=self.device)
        h = torch.relu(self.fc(h))
        h = h.view(-1, self.num_filters, self.out_dim, self.out_dim)
        for i in range(self.num_layers - 1):
            h = torch.relu(self.deconv_layers[i](h))
        obs = self.deconv_layers[-1](h)
        return obs



class PredictorInLatent(nn.Module):
    def __init__(self, device, features_dim, frames_stack):
        super().__init__()
        self.device = device
        self.features_dim = features_dim
        self.frames_stack = frames_stack

        # what are these sizes tho?
        self.net = nn.Sequential(
            nn.Linear(args.frames_stack * self.features_dim + 1, 512), nn.ReLU(inplace=True),
            nn.Linear(512, 256), nn.ReLU(inplace=True),
            nn.Linear(256, args.frames_stack * self.features_dim)
            )
# maybe you want to normalize the latent space first
# then you can have a meaningful activation at the end
# but whatever for now
    def forward(self, obs_latent, act):
        obs_latent_and_act = torch.cat((obs_latent, torch.tensor(act).view(-1,1)), dim=1)
        obs_latent_next = self.net(obs_latent_and_act)
        return obs_latent_next

# TODO finish this
class InversePredictorInLatent(nn.Module):
    def __init__(self, device, features_dim, frames_stack, batch_size):
        super().__init__()
        self.device = device
        self.features_dim = features_dim
        self.frames_stack = frames_stack

# give full stacks, it's the same action in between them
        self.net = nn.Sequential(
            nn.Linear(2 * args.frames_stack * self.features_dim, 512), nn.ReLU(inplace=True),
            nn.Linear(512, 256), nn.ReLU(inplace=True),
            nn.Linear(256, batch_size)
            )
# maybe you want to normalize the latent space first
# then you can have a meaningful activation at the end
# but whatever for now
    def forward(self, obs_latent, obs_latent_next):
        # TODO check this concat
        consequtives_obses = torch.cat((obs_latent, obs_latent_next), dim=1)
        obs_latent_next = self.net(obs_latent_and_act)
        return obs_latent_next


