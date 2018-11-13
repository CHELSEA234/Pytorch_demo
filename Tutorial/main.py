import torch 
import torchvision			# I think this is for pretrained model
import torch.nn as nn
import numpy as np
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as dsets
from torch.autograd import Variable

# #======================= Basic autograd example 1 =======================#
# # Create tensors.
# x = Variable(torch.Tensor([1]), requires_grad=True)
# w = Variable(torch.Tensor([2]), requires_grad=True)
# b = Variable(torch.Tensor([3]), requires_grad=True)

# # Build a computational graph.
# y = w * x + b    # y = 2 * x + 3

# # Compute gradients.
# y.backward()

# # Print out the gradients.
# # print(x.grad)    # x.grad = 2 
# # print(w.grad)    # w.grad = 1 
# # print(b.grad)    # b.grad = 1 

# #======================== Basic autograd example 2 =======================#

# torch.manual_seed(1)    # reproducible

# # Create tensors.
# x = Variable(torch.randn(5, 3))		# it is like 5 samples with 3 dim for each
# y = Variable(torch.randn(5, 2))

# # Build a linear layer.
# linear = nn.Linear(3, 2)		# this is parameter, in 5x3, out is 5x2
# print ('w: ', linear.weight)		#weight size is 2x3, transposed is 3x2 for operation 
# print ('b: ', linear.bias)		# bias is 3x1

# # Build Loss and Optimizer.
# criterion = nn.MSELoss()
# optimizer = torch.optim.SGD(linear.parameters(), lr=0.01)	# net here is linear 

# # Forward propagation.
# pred = linear(x)		# in is 5x3, weight is 3x2, out is 5x2

# # Compute loss.
# loss = criterion(pred, y)
# print('loss: ', loss.data[0])

# # Backpropagation.
# loss.backward()

# # Print out the gradients.
# print ('dL/dw: ', linear.weight.grad) 
# print ('dL/db: ', linear.bias.grad)

# # 1-step Optimization (gradient descent).
# optimizer.step()

# # # You can also do optimization at the low level as shown below.
# # # linear.weight.data.sub_(0.01 * linear.weight.grad.data)
# # # linear.bias.data.sub_(0.01 * linear.bias.grad.data)

# # Print out the loss after optimization.
# pred = linear(x)
# loss = criterion(pred, y)
# print('loss after 1 step optimization: ', loss.data[0])

# #======================== Loading data from numpy ========================#
# a = np.array([[1,2], [3,4]])
# b = torch.from_numpy(a)      # convert numpy array to torch tensor
# c = b.numpy()                # convert torch tensor to numpy array


# #===================== Implementing the input pipline =====================#
# # Download and construct dataset.
# train_dataset = dsets.CIFAR10(root='../data/',
#                                train=True, 
#                                transform=transforms.ToTensor(),
#                                download=True)

# # Select one data pair (read data from disk).
# image, label = train_dataset[0]
# # print (len(train_dataset))	the sample number is 50000
# # print (image.size())	the size is 3, 32, 32
# # print (label)		the is 1

# # Data Loader (this provides queue and thread in a very simple way).
# train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=100, shuffle=True, num_workers=2)

# # When iteration starts, queue and thread start to load dataset from files.
# data_iter = iter(train_loader)

# # Mini-batch images and labels. it will iterate automatically in the iteration below
# images, labels = data_iter.next()

# print (images.size())
# print (labels.size())

# # Actual usage of data loader is as below.
# for images, labels in train_loader:
#     # Your training code will be written here
#     pass


#===================== Input pipline for custom dataset =====================#
# You should build custom dataset as below.

# it is like you need to implement custom dataset yourself
class CustomDataset(data.Dataset):
    def __init__(self):
        # TODO
        # 1. Initialize file path or list of file names. 
        pass
    def __getitem__(self, index):
        # TODO
        # 1. Read one data from file (e.g. using numpy.fromfile, PIL.Image.open).
        # 2. Preprocess the data (e.g. torchvision.Transform).
        # 3. Return a data pair (e.g. image and label).
        pass
    def __len__(self):
        # You should change 0 to the total size of your dataset.
        return 0 

# Then, you can just use prebuilt torch's data loader. 
custom_dataset = CustomDataset()
train_loader = torch.utils.data.DataLoader(dataset=custom_dataset, batch_size=100, shuffle=True, num_workers=2)

#========================== Using pretrained model ==========================#
# Download and load pretrained resnet.
resnet = torchvision.models.resnet18(pretrained=True)

# If you want to finetune only top layer of the model.
for param in resnet.parameters():
    param.requires_grad = False
    
# Replace top layer for finetuning.
resnet.fc = nn.Linear(resnet.fc.in_features, 100)  # 100 is for example.

# For test.
images = Variable(torch.randn(10, 3, 256, 256))
outputs = resnet(images)
print (outputs.size())   # (10, 100)

#============================ Save and load the model ============================#
# Save and load the entire model.
torch.save(resnet, 'model.pkl')
model = torch.load('model.pkl')

# Save and load only the model parameters(recommended).
torch.save(resnet.state_dict(), 'params.pkl')
resnet.load_state_dict(torch.load('params.pkl'))

