import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data
import torchvision
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import numpy as np


torch.manual_seed(1)    # reproducible

# Hyper Parameters
EPOCH = 1
BATCH_SIZE = 64
LR = 0.005         # learning rate
DOWNLOAD_MNIST = False
N_TEST_IMG = 5

train_data = torchvision.datasets.MNIST( root='./data/', train=True, transform=torchvision.transforms.ToTensor(), download=DOWNLOAD_MNIST)

# plot one example
print(train_data.train_data.size())     # (60000, 28, 28)
print(train_data.train_labels.size())   # (60000)
print (torch.max(train_data.train_data[2]))

# Data Loader for easy mini-batch return in training, the image batch shape will be (50, 1, 28, 28)
train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)

class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(28*28, 128),
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, 12),
            nn.Tanh(),
            nn.Linear(12, 3),   # compress to 3 features which can be visualized in plt
        )
        self.decoder = nn.Sequential(
            nn.Linear(3, 12),
            nn.Tanh(),
            nn.Linear(12, 64),
            nn.Tanh(),
            nn.Linear(64, 128),
            nn.Tanh(),
            nn.Linear(128, 28*28),
            nn.Sigmoid(),       # compress to a range (0, 1)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded

class AutoEncoder_2(nn.Module):
    def __init__(self):
        super(AutoEncoder_2, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(28*28, 128),
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, 12),
            nn.Tanh(),
            nn.Linear(12, 3),   # compress to 3 features which can be visualized in plt
        )
        self.decoder = nn.Sequential(
            nn.Linear(3, 12),
            nn.Tanh(),
            nn.Linear(12, 64),
            nn.Tanh(),
            nn.Linear(64, 128),
            nn.Tanh(),
            nn.Linear(128, 28*28),
            nn.Sigmoid(),       # compress to a range (0, 1)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded
        

autoencoder = AutoEncoder()
autoencoder_2 = AutoEncoder_2()
autoencoder = autoencoder.cuda()
autoencoder_2 = autoencoder_2.cuda()

optimizer = torch.optim.Adam(autoencoder.parameters(), lr=LR)
optimizer_2 = torch.optim.Adam(autoencoder_2.parameters(), lr=LR)
loss_func = nn.MSELoss()

for epoch in range(EPOCH):
    for step, (x, y) in enumerate(train_loader):
        # input should have Variable type
        b_x = Variable(x.view(-1, 28*28)).type(torch.FloatTensor)   # batch x, shape (batch, 28*28)
        b_y = Variable(x.view(-1, 28*28)).type(torch.FloatTensor)   # batch y, shape (batch, 28*28)

        b_x = b_x.cuda()
        b_y = b_y.cuda()

        encoded, decoded = autoencoder(b_x)
        encoded_2, decoded_2 = autoencoder_2(decoded)

        loss = loss_func(decoded, b_y)      # mean square error
        optimizer.zero_grad()               # clear gradients for this training step
        loss.backward(retain_variables=True)                     # backpropagation, compute gradients, these variables are true in computational graph
        optimizer.step()                    # apply gradients

        loss_2 = loss_func(decoded_2, b_y)
        optimizer_2.zero_grad()
        loss_2.backward()
        optimizer_2.step()

        if step % 100 == 0:
            print('Step: ', step, '| coarse loss: %.4f' % loss.data[0])
            print('Step: ', step, '| fine loss: %.4f' % loss_2.data[0])
