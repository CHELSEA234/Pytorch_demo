import torch
import torch.nn as nn
import torch.autograd as autograd

torch.manual_seed(1)    # reproducible

def mse_loss(input, target):
    result = (input-target).pow(2)
    # print (torch.sum(result).data[0])
    # print (input.data.nelement())
    return torch.sum(result) / input.data.nelement()


loss = nn.MSELoss()
input = autograd.Variable(torch.randn(3, 5), requires_grad=True)
target = autograd.Variable(torch.randn(3, 5))
output = loss(input, target)
output.backward()

print (type(output.data))
print (type(target.data))
print (output.data[0])
print (mse_loss(input, target).data[0])
