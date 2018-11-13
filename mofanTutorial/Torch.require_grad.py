import torch
from torch.autograd import Variable

w1 = Variable(torch.Tensor([1.0,2.0,3.0]),requires_grad=True)	#需要求导的话，requires_grad=True属性是必须的。
w2 = Variable(torch.Tensor([1.0,2.0,3.0]),requires_grad=True)
# print(w1.grad)		# the gradient is zero
# print(w2.grad)

d = torch.mean(w1**2)
d.backward()
# print (w1.grad)

# d.backward()		this will cause erro
# print (w1.grad)

tensor = torch.FloatTensor([[1,2],[3,4]])
variable = Variable(tensor, requires_grad=True)
print (variable.grad)			# this is None
# d = torch.mean(variable)		# single variable won't have problem to do multiple times
d = torch.mean(variable*variable)		# if this is a network, you can't free the buffer, using backward

d.backward()
print (variable.grad)
# print (variable.weights)
