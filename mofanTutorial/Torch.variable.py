import numpy
import torch
from torch.autograd import Variable 	# torch 中 Variable 模块， 这种variable是可以计算梯度的

# 先生鸡蛋 	<class 'torch.FloatTensor'>  torch.Size([2, 2])
a = [[1,2],[3,4]]
tensor = torch.from_numpy(numpy.array(a))
tensor = torch.FloatTensor(a)		# torch.FloatTensor converts list, torch.from_numpy converts nd-array
tensor_numpy = tensor.numpy()

# 把鸡蛋放到篮子里, requires_grad是参不参与误差反向传播, 要不要计算梯度		<class 'torch.autograd.variable.Variable'>	torch.Size([2, 2])
variable = Variable(tensor, requires_grad=True)

t_out = torch.mean(tensor*tensor)       # x^2, this is just float variable, the summation of all 4 variables
# the last sentence is equal to t_out_1 = 1/4.*sum(sum(tensor*tensor))	
n_out = numpy.mean(tensor_numpy*tensor_numpy)
v_out = torch.mean(variable*variable)   	# x^2

# print('tensor operation:', t_out)		# 7.5
# print('ndarray operation:', n_out)
# print('variable operation:', v_out)    # 7.5000 	[torch.FloatTensor of size 1]
# print('variable data:', v_out.data)		# 7.5000	[torch.FloatTensor of size 1]
# print('variable data[0]:', v_out.data[0])	# 7.5

##############################################################################################
v_out.backward()    # backpropagation from v_out, this is output, inputs are variables, each of them will a grad value

v_out = 1/4 * sum(variable*variable)
# the gradients w.r.t the variable, d(v_out)/d(variable) = 1/4*2*variable = variable/2
# the contribution from each point
print(variable.grad)
'''
 0.5000  1.0000
 1.5000  2.0000
'''

print(variable)     # this is data in variable format
"""
Variable containing:
 1  2
 3  4
[torch.FloatTensor of size 2x2]
"""

print(variable.data)    # this is data in tensor format
"""
 1  2
 3  4
[torch.FloatTensor of size 2x2]
"""

print(variable.data.numpy())    # numpy format
"""
[[ 1.  2.]
 [ 3.  4.]]
"""
