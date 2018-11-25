import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution kernel
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)

        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))

        # 2 is ame as (2, 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)

        x = x.view(-1, self.num_flat_features(x))	# GX: do I need to add new function
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        print (size)
        num_features = 1
        for s in size:       # Get the products
            num_features *= s
        return num_features

net = Net()
print(net)

params = list(net.parameters())		# GX: how to check the numerical value of them??
print(len(params))       # 10: 10 sets of trainable parameters

print(params[0].size())  # torch.Size([6, 1, 5, 5])

input = Variable(torch.randn(1, 1, 32, 32))
out = net(input)   # out's size: 1x10.


"""""""""""""""""""""Backward"""""""""""""""""""""""""""
# Doing these iteratively
net.zero_grad()
out.backward()


"""""""""""""""""""""Loss function"""""""""""""""""""""""""""
output = net(input)
target = Variable(torch.arange(1, 11))   # Create a dummy true label Size 10.
criterion = nn.MSELoss()

# Compute the loss by MSE of the output and the true label
loss = criterion(output, target)         # Size 1

net.zero_grad()      # zeroes the gradient buffers of all parameters
loss.backward()


"""""""""""""""""""""Optimizer"""""""""""""""""""""""""""
import torch.optim as optim

# Create a SGD optimizer for gradient descent
optimizer = optim.SGD(net.parameters(), lr=0.01)

# Inside the training loop
for t in range(500):
   output = net(input)
   loss = criterion(output, target)

   optimizer.zero_grad()   # zero the gradient buffers
   loss.backward()

   optimizer.step()        # Perform the training parameters update

learning_rate = 1e-4
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


"""""""""""""""""""""Sequential"""""""""""""""""""""""""""
import torch
from torch.autograd import Variable

N, D_in, H, D_out = 64, 1000, 100, 10

x = Variable(torch.randn(N, D_in))
y = Variable(torch.randn(N, D_out), requires_grad=False)

# GX: using sequential, so that we do not need to build a class and declaim forward backward functions anymore
model = torch.nn.Sequential(
    torch.nn.Linear(D_in, H),
    torch.nn.ReLU(),
    torch.nn.Linear(H, D_out),
)

loss_fn = torch.nn.MSELoss(size_average=False)

learning_rate = 1e-4
for t in range(500):
    y_pred = model(x)

    loss = loss_fn(y_pred, y)
    print(t, loss.data[0])

    model.zero_grad()

    loss.backward()

    for param in model.parameters():
        param.data -= learning_rate * param.grad.data


"""""""""""""""""""""Dynamic model"""""""""""""""""""""""""""
# According to tutorial, at each iteration, we can have different models
class DynamicNet(torch.nn.Module):
    def __init__(self, D_in, H, D_out):
        super(DynamicNet, self).__init__()
        self.input_linear = torch.nn.Linear(D_in, H)
        self.middle_linear = torch.nn.Linear(H, H)
        self.output_linear = torch.nn.Linear(H, D_out)

    def forward(self, x):
        h_relu = self.input_linear(x).clamp(min=0)
        for _ in range(random.randint(0, 3)):
            h_relu = self.middle_linear(h_relu).clamp(min=0)
        y_pred = self.output_linear(h_relu)
        return y_pred


"""""""""""""""""""""Transfer model"""""""""""""""""""""""""""
# Not clear about how he did it...
# https://jhui.github.io/2018/02/09/PyTorch-neural-networks/
