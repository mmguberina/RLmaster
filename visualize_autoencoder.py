import numpy as np
import matplotlib.pyplot as plt
import deeptrack as dt
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torch.nn as nn
from a import ConvAutoencoder, Autoencoder
#from a import Autoencoder


IMAGE_SIZE=64
sequence_length=10#Number of frames per sequence
MIN_SIZE=.5e-6
MAX_SIZE=1.5e-6
MAX_VEL=10 #Maximum velocity. The higher the trickier!
MAX_PARTICLES=3 #Max number of particles in each sequence. The higher the trickier!

#Defining properties of the particles
particle=dt.Sphere(intensity=lambda: 10+10*np.random.rand(),
                   radius=lambda: MIN_SIZE+np.random.rand()*(MAX_SIZE-MIN_SIZE),
                   position=lambda: IMAGE_SIZE*np.random.rand(2),vel=lambda: MAX_VEL*np.random.rand(2),
                   position_unit="pixel")

#Defining an update rule for the particle position
def get_position(previous_value,vel):

    newv=previous_value+vel
    for i in range(2):
        if newv[i]>63:
            newv[i]=63-np.abs(newv[i]-63)
            vel[i]=-vel[i]
        elif newv[i]<0:
            newv[i]=np.abs(newv[i])
            vel[i]=-vel[i]
    return newv

particle=dt.Sequential(particle,position=get_position)
#particle_seq = dt.Sequential(particle,position=get_position)

#Defining properties of the microscope
optics = dt.Fluorescence(NA=1,output_region=(0, 0,IMAGE_SIZE, IMAGE_SIZE), 
    magnification=10,
    resolution=(1e-6, 1e-6),
    wavelength=633e-9)


#Combining everything into a dataset. 
#Note that the sequences are flipped in different directions, so that each unique sequence defines
#in fact 8 sequences flipped in different directions, to speed up data generation
dataset=dt.FlipUD(dt.FlipDiagonal(dt.FlipLR(dt.Sequence(optics(particle**(lambda: 1+np.random.randint(MAX_PARTICLES))),sequence_length=sequence_length))))


#dataset.update().plot(cmap="gray") #This generates a new sequence and plots it
#video=dataset.update().resolve() #This generates a new sequence and stores in in "video"

# nn stuff

NUM_EPOCHS = 100
LEARNING_RATE = 1e-3
BATCH_SIZE = 32


particle_one=dt.Sphere(intensity=lambda: 10+10*np.random.rand(),
                   radius=lambda: MIN_SIZE+np.random.rand()*(MAX_SIZE-MIN_SIZE),
                   position=lambda: IMAGE_SIZE*np.random.rand(2),
                   position_unit="pixel")

dataset_one_img = optics(particle_one**(lambda: 1+np.random.randint(MAX_PARTICLES)))
#dataset_one_img = optics(particle)
# NOTE here you prolly want to normalize but whatever for now
transform = transforms.Compose([
    transforms.ToTensor()
    ])
#while True:
#    dataset_one_img.update()
#    plt.show()
#    output_image = dataset_one_img.resolve()
#    print(trans(output_image).shape)
#    plt.imshow(np.squeeze(output_image))
#    plt.show()


class MovingParticlesDataset(Dataset):
    def __init__(self, deeptrack_dataset_one_img, size, transform):
        self.deeptrack_dataset_one_img = deeptrack_dataset_one_img
        self.transform = transform
        self.size = size # since all of this is generated, we choose how much we want

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        # there is no need for indexing as you can simply generate a 
        # new image every time this is called
        #return self.transform(self.deeptrack_dataset_one_img.update().resolve())
        return torch.tensor(self.deeptrack_dataset_one_img.update().resolve(), dtype=torch.float32).reshape(1, 64, 64) /255

#net = torch.load('Autoencoder_net')
#net = torch.load('Autoencoder_2048', map_location=torch.device('cpu'))
net = torch.load('Autoencoder_2048_w_lin', map_location=torch.device('cpu'))
#net = torch.load('an_early_autoencoder')
        
pytorched_dataset = MovingParticlesDataset(dataset_one_img, 5, transform)

fig, axs = plt.subplots(2, 5)


for i in range(len(pytorched_dataset)):
    sample = pytorched_dataset[i] 
#    im_enc = net.encode(sample.reshape(1,1,64,64))
#    print(im_enc.shape)
#    exit()
#    im_ae = net(sample.reshape(1,1,64,64))
    im_ae = net.decode(net.encode((sample.reshape(1,1,64,64))))
    axs[0, i].imshow(torch.squeeze(sample), cmap='gray')
#    plt.pause(0.001)
#    ax = plt.subplot(2, 4, i + 1)
#    plt.tight_layout()
#    ax.set_title('sample #{}'.format(i))
#    ax.axis('off')
    axs[1, i].imshow(torch.squeeze(im_ae).detach().numpy(), cmap='gray')#.permute(1,2,0))

plt.show()

# this plots each image separately
#for im in video:
#    plt.imshow(np.squeeze(im))
#    plt.show()

#for im in video:
    #t = torch.tensor(im).reshape((1,1,64,64))
#print(torch.tensor(video, dtype=torch.float32).reshape((sequence_length, 1, 64, 64)))
#pytorched_dataset = MovingParticlesDataset(dataset_one_img, 4)
#fig = plt.figure()
#for i in range(len(pytorched_dataset)):
#    sample = pytorched_dataset[i]
#    print(i, sample.shape)
#    ax = plt.subplot(1, 4, i + 1)
#    plt.tight_layout()
#    ax.set_title('sample #{}'.format(i))
#    ax.axis('off')
#    plt.imshow(sample)
#    plt.pause(0.001)
#plt.show()
#


