import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data.sampler import SubsetRandomSampler

from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import os
from RLmaster.util.atari_dataset import AtariImageDataset, showSample
from RLmaster.latent_representations.autoencoder_nn import CNNEncoder, CNNDecoder

def evaluate(test_loader, encoder, decoder):
    test_loss= 0 
    for images in test_loader:
        images = images.to(device)
        with torch.no_grad():
            encoded = encoder(images)
            decoded = decoder(encoded)
            loss = criterion(decoded, images)
            test_loss += loss.item() * images.size(0)

    total_loss = test_loss / len(test_loader)
    return total_loss



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dir_name = "Pong-v4_dataset"


transform = transforms.ToTensor()

train_dataset = AtariImageDataset(root_dir="/home/gospodar/chalmers/MASTER/RLmaster/RLmaster/util/", 
                                  dir_name=dir_name, transform=transform, train=True)

test_dataset = AtariImageDataset(root_dir="/home/gospodar/chalmers/MASTER/RLmaster/RLmaster/util/", 
                                  dir_name=dir_name, transform=transform, train=False)

#train_dataset = AtariImageDataset('dataset', train=True, download=False, transform=transform)
#test_dataset = datasets.CIFAR10('dataset', train=False, download=False, transform=transform)

num_workers = 4
batch_size = 32
features_dim = 576

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers)

dataiter = iter(train_loader)
images = dataiter.next()


observation_shape = images[0].shape
encoder = CNNEncoder(observation_shape=observation_shape, features_dim=features_dim)
encoder.to(device)
decoder = CNNDecoder(observation_shape=observation_shape, n_flatten=encoder.n_flatten, features_dim=features_dim)
decoder.to(device)

print("=================encoder=================")
print(encoder)
print("=================decoder=================")
print(decoder)


criterion = nn.BCELoss()
optimizer_encoder = torch.optim.Adam(encoder.parameters(), lr=0.001)
optimizer_decoder = torch.optim.Adam(decoder.parameters(), lr=0.001)
n_epochs = 100
#
lowest_test_loss = 10**8
for epoch in range(1, n_epochs + 1):
    if epoch % 10 == 0:
        test_loss = evaluate(test_loader, encoder, decoder)
        print("Current evaluation loss:", test_loss)
        if test_loss < lowest_test_loss:
            lowest_test_loss = test_loss
        else:
            print("overfitting could be (probably is) happening")
        torch.save(encoder.state_dict(), "encoder_features_dim_" + str(features_dim) + "_{}.pt".format(epoch))
        torch.save(decoder.state_dict(), "decoder_features_dim_" + str(features_dim) + "_{}.pt".format(epoch))

    train_loss = 0.0
    for images in train_loader:
        images = images.to(device)
        optimizer_encoder.zero_grad()
        optimizer_decoder.zero_grad()
        encoded = encoder(images)
        decoded = decoder(encoded)
        loss = criterion(decoded, images)
        loss.backward()
        optimizer_encoder.step()
        optimizer_decoder.step()
        train_loss += loss.item() * images.size(0)

    train_loss = train_loss / len(train_loader)
    print('Epoch: {} \tTraining Loss: {:.6f}'.format(epoch, train_loss))
torch.save(encoder.state_dict(), "encoder_features_dim_" + str(features_dim) + ".pt")
torch.save(decoder.state_dict(), "decoder_features_dim_" + str(features_dim) + ".pt")
