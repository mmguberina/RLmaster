import subprocess
import os
import torch
from torch.utils.data import Dataset
from skimage import io
import matplotlib.pyplot as plt
import numpy as np

class AtariImageDataset(Dataset):
    def __init__(self, root_dir, dir_name, transform, train):
        self.train = train
        self.image_names = subprocess.check_output("ls " + dir_name, shell=True).decode('utf-8').split('\n')[:-1]
        self.transform = transform
        self.root_dir = os.path.join(root_dir, dir_name)
        # let's say 15% of the data is the test dataset
        self.offset_if_test = int(0.85 *  len(self.image_names)) - 2
        if train:
            self.size = int(0.85 *  len(self.image_names))
        else:
            self.size = int(0.15 *  len(self.image_names))


    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        if self.train == False:
            idx += self.offset_if_test

        img_name = os.path.join(self.root_dir, self.image_names[idx])
        image = io.imread(img_name)
        if self.transform:
            sample = self.transform(image)

        return sample

def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    plt.imshow(np.transpose(img, (1, 2, 0)))  # convert from Tensor image

def showSample(data_loader):
    dataiter = iter(data_loader)
    images = dataiter.next()
    images = images.numpy()

    fig = plt.figure(figsize=(25, 4))
    for i in np.arange(20):
        ax = fig.add_subplot(2, 20//2, i + 1, xticks=[], yticks=[])
        imshow(images[i])
        #ax.set_title(classes[labels[i]])
    plt.show()
