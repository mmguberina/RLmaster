import torch
import torch.nn as nn
from torchvision import transforms
from RLmaster.util.atari_dataset import AtariImageDataset
from RLmaster.latent_representations.autoencoder_nn import CNNEncoderNew, CNNDecoderNew
import matplotlib.pyplot as plt
import numpy as np

def imshow(img):
#    img = img / 2 + 0.5  # unnormalize
    plt.imshow(np.transpose(img, (1, 2, 0)))  # convert from Tensor image


dir_name = "Pong-v4_dataset"
transform = transforms.ToTensor()
num_workers = 0
batch_size = 32
train_dataset = AtariImageDataset(
        root_dir="/home/gospodar/chalmers/MASTER/RLmaster/RLmaster/util/", 
                                  dir_name=dir_name, transform=transform, train=False)
train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, num_workers=num_workers)

dataiter = iter(train_loader)
images = dataiter.next()

features_dim = 3136

observation_shape = images[0].shape
encoder = CNNEncoderNew(observation_shape=observation_shape, 
        features_dim=features_dim, device="cpu")
decoder = CNNDecoderNew(observation_shape=observation_shape, 
        n_flatten=encoder.n_flatten, features_dim=features_dim)

encoder.load_state_dict(torch.load("../../experiments/latent_only/log/PongNoFrameskip-v4/from_stored_dataset/encoder_features_dim_{}.pt".format(features_dim), map_location=torch.device('cpu')))
decoder.load_state_dict(torch.load("../../experiments/latent_only/log/PongNoFrameskip-v4/from_stored_dataset/decoder_features_dim_{}.pt".format(features_dim), map_location=torch.device('cpu')))

#encoder.load_state_dict(torch.load("../../experiments/dqn_on_latent/log/PongNoFrameskip-v4/synched_ae/checkpoint_encoder_epoch_4.pth", map_location=torch.device('cpu'))['encoder'])
#decoder.load_state_dict(torch.load("../../experiments/dqn_on_latent/log/PongNoFrameskip-v4/synched_ae/checkpoint_decoder_epoch_4.pth", map_location=torch.device('cpu'))['decoder'])
with torch.no_grad():
    rez = decoder(encoder(images))
    images = rez.numpy()
fig = plt.figure(figsize=(4, 2))
for i in np.arange(8):
    ax = fig.add_subplot(2, 8//2, i + 1, xticks=[], yticks=[])
    imshow(images[i])
    #ax.set_title(classes[labels[i]])
#plt.savefig("../../experiments/dqn_on_latent/ae_resulting_images_features_dim_{}.png".format(features_dim), dpi=600)
plt.show()

