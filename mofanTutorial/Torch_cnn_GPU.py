import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data
import torchvision
from torchvision import datasets, transforms

torch.manual_seed(1)

EPOCH = 10
BATCH_SIZE = 50
LR = 0.001
DOWNLOAD_MNIST = False

# this downloading process can't work at HPC
trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (1.0,))])
# train_data = datasets.MNIST('../multipleGPU/data', train=True, download=True, transform=trans)
# test_data = datasets.MNIST('../multipleGPU/data', train=False, transform=trans)
train_data = datasets.MNIST('../data', train=True, download=True, transform=trans)
test_data = datasets.MNIST('../data', train=False, transform=trans)
train_loader = torch.utils.data.DataLoader( train_data, batch_size=BATCH_SIZE, shuffle=True)
test_loader = torch.utils.data.DataLoader( test_data, batch_size=BATCH_SIZE, shuffle=True)

print ('==>>> total trainning batch number: {}'.format(len(train_loader)))
print ('==>>> total testing batch number: {}'.format(len(test_loader)))

# !!!!!!!! Change in here !!!!!!!!! # testing data should be on GPU
test_x = Variable(torch.unsqueeze(test_data.test_data, dim=1)).type(torch.FloatTensor)[:2000].cuda()/255.   # Tensor on GPU
test_y = test_data.test_labels[:2000].cuda() 

class CNN(nn.Module):
    def __init__(self):
        # defining layers here
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=2,),
                                   nn.ReLU(), nn.MaxPool2d(kernel_size=2),)
        self.conv2 = nn.Sequential(nn.Conv2d(16, 32, 5, 1, 2), nn.ReLU(), nn.MaxPool2d(2),)
        self.out = nn.Linear(32 * 7 * 7, 10)

    def forward(self, x):
        # combining layers here
        x = self.conv1(x)
        x = self.conv2(x)
        print ('x size', x.size())
        x = x.view(x.size(0), -1)
        output = self.out(x)
        return output

cnn = CNN()

# !!!!!!!! Change in here !!!!!!!!!     # net should be GPU
cnn.cuda()      # Moves all model parameters and buffers to the GPU. why the class type is generator???
                # no mater how many GPU you are going to use, you need to move everything on to the cuda
cnn = torch.nn.DataParallel(module=cnn, device_ids=[0, 1])        # this will let two cpus to run at same time
                # you can't claim 3 devices when you have just only two nodes
                # device_ids=[0] is same as with only on device

optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)
loss_func = nn.CrossEntropyLoss()

for epoch in range(EPOCH):
    for step, (x, y) in enumerate(train_loader):

        # !!!!!!!! Change in here !!!!!!!!! #
        b_x = Variable(x).cuda(0)   # training Tensor on GPU, if this variable is on CPU, it won't have attribute 'get_device()'
        b_y = Variable(y).cuda(0)    # Tensor on GPU

        # do optimization on GPU
        output = cnn(b_x)
        loss = loss_func(output, b_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 50 == 0:
            test_output = cnn(test_x)
            # !!!!!!!! Change in here !!!!!!!!! #
            pred_y = torch.max(test_output, 1)[1].cuda().data.squeeze()  # move the computation in GPU
            # these two variables are on GPU, so this will be on GPU
            accuracy = torch.sum(pred_y == test_y) / test_y.size(0)
            print('Epoch: ', epoch, '| train loss: %.4f' % loss.data[0], '| test accuracy: %.2f' % accuracy)

test_output = cnn(test_x[:10])

# !!!!!!!! Change in here !!!!!!!!! #
pred_y = torch.max(test_output, 1)[1].cuda().data.squeeze() # move the computation in GPU

print(pred_y, 'prediction number')
print(test_y[:10], 'real number')

