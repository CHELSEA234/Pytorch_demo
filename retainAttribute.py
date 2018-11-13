import torch
from torch.autograd import Variable

x = Variable(torch.ones(2, 2), requires_grad = True)
y = x ** 2
# print (y)
y.backward(torch.ones(2, 2), retain_variables = False)
# print ("first backward of x is:")
# print (x.grad)
print (y)

y.backward(2*torch.ones(2, 2), retain_variables=False)
# print ("second backward of x is:")
# print (x.grad)
