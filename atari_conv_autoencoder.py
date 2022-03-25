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
from atari_dataset import AtariImageDataset, showSample
from autoencoder_nn import NatureCNNEncoder, CNNDecoder

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
dir_name = "pong_dataset"


transform = transforms.ToTensor()

train_dataset = AtariImageDataset(root_dir="/home/gospodar/chalmers/MASTER/RLmaster/", 
                                  dir_name="pong_dataset", transform=transform, train=True)

test_dataset = AtariImageDataset(root_dir="/home/gospodar/chalmers/MASTER/RLmaster/", 
                                  dir_name="pong_dataset", transform=transform, train=False)

#train_dataset = AtariImageDataset('dataset', train=True, download=False, transform=transform)
#test_dataset = datasets.CIFAR10('dataset', train=False, download=False, transform=transform)

num_workers = 0
batch_size = 20

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers)

dataiter = iter(train_loader)
images = dataiter.next()

observation_shape = images[0].shape
encoder = NatureCNNEncoder(observation_shape=observation_shape)
encoder.to(device)
decoder = CNNDecoder(observation_shape=observation_shape)
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
        torch.save(encoder.state_dict(), "encoder{}.pt".format(epoch))
        torch.save(decoder.state_dict(), "decoder{}.pt".format(epoch))

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
torch.save(encoder.state_dict(), "encoder.pt")
torch.save(decoder.state_dict(), "decor.pt")
