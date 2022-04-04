import torch
import torch.nn as nn
from torchvision import transforms
from atari_dataset import AtariImageDataset
import matplotlib.pyplot as plt
import numpy as np

# taken from stablebaselines3
# but everyone uses this anyway
# make it usable
class CNNEncoder(nn.Module):

    def __init__(self, observation_shape, 
            features_dim: int = 512):
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
        self.encoder = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
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
    def __init__(self, observation_shape, n_flatten, features_dim: int = 512):
        super(CNNDecoder, self).__init__()
        self.observation_shape = observation_shape
        self._features_dim = features_dim
        n_input_channels = self.observation_shape[0]

#        self.linear = nn.Sequential(
#            nn.Linear(features_dim, n_flatten),
#            nn.ReLU(),
#        )

        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.ConvTranspose2d(32, n_input_channels, kernel_size=8, stride=4, padding=0),
            nn.Sigmoid(),
        )

    def forward(self, latent_observations: torch.Tensor) -> torch.Tensor:
#        after_lin = self.linear(latent_observations)
        deconv = latent_observations.view(-1, 64, 7, 7)
        obs = self.deconv(deconv)
#        print(obs.shape)
        return obs

def imshow(img):
#    img = img / 2 + 0.5  # unnormalize
    plt.imshow(np.transpose(img, (1, 2, 0)))  # convert from Tensor image


if __name__ == "__main__":

    dir_name = "pong_dataset"
    transform = transforms.ToTensor()
    num_workers = 0
    batch_size = 20
    train_dataset = AtariImageDataset(root_dir="/home/gospodar/chalmers/MASTER/RLmaster/", 
                                      dir_name="pong_dataset", transform=transform, train=False)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers)

    dataiter = iter(train_loader)
    images = dataiter.next()

    observation_shape = images[0].shape
    encoder = CNNEncoder(observation_shape=observation_shape)
    decoder = CNNDecoder(observation_shape=observation_shape)

    encoder.load_state_dict(torch.load("./encoder90.pt", map_location=torch.device('cpu')))
    decoder.load_state_dict(torch.load("./decoder90.pt", map_location=torch.device('cpu')))
    with torch.no_grad():
        rez = decoder(encoder(images))
        images = rez.numpy()
    fig = plt.figure(figsize=(25, 4))
    for i in np.arange(20):
        ax = fig.add_subplot(2, 20//2, i + 1, xticks=[], yticks=[])
        imshow(images[i])
        #ax.set_title(classes[labels[i]])
    plt.show()
