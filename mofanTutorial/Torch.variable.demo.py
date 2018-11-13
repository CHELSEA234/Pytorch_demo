import torch
import numpy
import torch.optim as optim
from torch.autograd import Variable

np_array = numpy.array([[1,2],[3,4]])
x = Variable(torch.from_numpy(np_array).float(), requires_grad=True)
y = x + 2
z = y * y * 3
out = z.mean()

out.backward()
print (x.grad)
print ("can't display grad on y and z")
