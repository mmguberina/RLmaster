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

NUM_EPOCHS = 200
LEARNING_RATE = 0.5*1e-3
BATCH_SIZE = 128


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


class MovingParticlesDataset1Img(Dataset):
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
        # we are dividing by 255 to normalize the data
        return torch.tensor(self.deeptrack_dataset_one_img.update().resolve(), dtype=torch.float32).reshape(1, 64, 64) /255

trainset = MovingParticlesDataset1Img(dataset_one_img, 40 * BATCH_SIZE, transform=transform)
testset = MovingParticlesDataset1Img(dataset_one_img, BATCH_SIZE, transform=transform)

trainloader = torch.utils.data.DataLoader(trainset,
                                             batch_size=BATCH_SIZE, shuffle=True,
                                             num_workers=4)

testloader = torch.utils.data.DataLoader(testset,
                                             batch_size=BATCH_SIZE, shuffle=True,
                                             num_workers=4)
        
def get_device():
    if torch.cuda.is_available():
        device = 'cuda:0'
    else:
        device = 'cpu'
    return device


class ConvAutoencoder(nn.Module):
    def __init__(self):
        super(ConvAutoencoder, self).__init__()
        # encoder
        self.enc1 = nn.Conv2d(
            in_channels=1, out_channels=16, kernel_size=2, stride=2
        )
        self.enc2 = nn.Conv2d(
            in_channels=16, out_channels=8, kernel_size=2, stride=2
        )
        # decoder 
        self.dec1 = nn.ConvTranspose2d(
            in_channels=8, out_channels=16, kernel_size=2, stride=2
        )
        self.dec2 = nn.ConvTranspose2d(
            in_channels=16, out_channels=1, kernel_size=2, stride=2
        )
    def forward(self, x):
       x = F.relu(self.enc1(x))
       x = F.relu(self.enc2(x))
       x = F.relu(self.dec1(x))
       x = F.relu(self.dec2(x))
       return x

    def encode(self, x):
       x = F.relu(self.enc1(x))
       x = F.relu(self.enc2(x))
       return x

    def decode(self, x):
       x = F.relu(self.dec1(x))
       x = F.relu(self.dec2(x))
       return x



#class ConvAutoencoderWLinLayer(nn.Module):
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        # encoder
        self.enc1 = nn.Conv2d(
            in_channels=1, out_channels=16, kernel_size=2, stride=2
        )
        self.enc2 = nn.Conv2d(
            in_channels=16, out_channels=8, kernel_size=2, stride=2
        )
        self.dropoutLayer = nn.Dropout(p=0.2)
        self.enc_lin = nn.Linear(2048, 2048)
        # decoder 
        self.dec_lin = nn.Linear(2048, 2048)
        self.dec1 = nn.ConvTranspose2d(
            in_channels=8, out_channels=16, kernel_size=2, stride=2
        )
        self.dec2 = nn.ConvTranspose2d(
            in_channels=16, out_channels=1, kernel_size=2, stride=2
        )
    def forward(self, x):
        x = F.relu(self.enc1(x))
        x = F.relu(self.enc2(x))
        print(x.shape)
        exit()
        x = x.view(-1, 2048)
        x = F.relu(self.enc_lin(x))
        x = F.relu(self.dec_lin(x))
        # TODO check these dimensions
        x = x.view(-1, 8, 16, 16)
        x = F.relu(self.dec1(x))
        x = F.relu(self.dec2(x))
        return x

    def encode(self, x):
        x = F.relu(self.enc1(x))
        x = F.relu(self.enc2(x))
        x = x.view(-1, 2048)
        x = self.enc_lin(x)
        return x

    def decode(self, x):
        x = self.dec_lin(x) 
        x = x.view(-1, 8, 16, 16)
        x = F.relu(self.dec1(x))
        x = F.relu(self.dec2(x))
        return x

if __name__ == "__main__":
    net = Autoencoder()
    #net = ConvAutoencoder()
    #net = torch.load('Autoencoder_2048', map_location=torch.device('cpu'))
    print(net)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)


    def train(net, trainloader, NUM_EPOCHS):
        train_loss = []
        for epoch in range(NUM_EPOCHS):
            running_loss = 0.0
            for data in trainloader:
                img = data # no need for the labels
                img = img.to(device)
                optimizer.zero_grad()
                outputs = net(img)
                loss = criterion(outputs, img)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            
            loss = running_loss / len(trainloader)
            train_loss.append(loss)
            print('Epoch {} of {}, Train Loss: {}'.format(
                epoch+1, NUM_EPOCHS, loss))

            if epoch % 5 == 0:
                #torch.save(net, 'Autoencoder_2048')
                torch.save(net, 'Autoencoder_2048_w_lin')
            #    save_decoded_image(img.cpu().data, name='./Conv_CIFAR10_Images/original{}.png'.format(epoch))
            #    save_decoded_image(outputs.cpu().data, name='./Conv_CIFAR10_Images/decoded{}.png'.format(epoch))
        return train_loss
    device = get_device()
    print(device)
    net.to(device)
    train_loss = train(net, trainloader, NUM_EPOCHS)
    #torch.save(net, 'Autoencoder_2048')
    torch.save(net, 'Autoencoder_2048_w_lin')
    plt.figure()
    plt.plot(train_loss)
    plt.title('Train Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.savefig('conv_ae_loss.png')
    plt.show()
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

    # this plots each image separately
    #for im in video:
    #    plt.imshow(np.squeeze(im))
    #    plt.show()

    #for im in video:
        #t = torch.tensor(im).reshape((1,1,64,64))
    #print(torch.tensor(video, dtype=torch.float32).reshape((sequence_length, 1, 64, 64)))
