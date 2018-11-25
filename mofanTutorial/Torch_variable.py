"""""""""""""""""""""Sample 1"""""""""""""""""""""""""""

import numpy
import torch
from torch.autograd import Variable 	# torch 中 Variable 模块， 这种variable是可以计算梯度的

# 先生鸡蛋 	<class 'torch.FloatTensor'>  torch.Size([2, 2])
a_list = [[1,2],[3,4]]		# <class 'list'>
a_nd = numpy.array(a_list)	# <class 'numpy.ndarray'>

tensor = torch.from_numpy(a_nd) 	# <class 'torch.LongTensor'>
tensor = torch.FloatTensor(a_list)	# <class 'torch.FloatTensor'>

tensor_numpy = tensor.numpy()

variable = Variable(tensor, requires_grad=True)	#  <class 'torch.autograd.variable.Variable'>

t_out = torch.mean(tensor*tensor) 				# operation on the tensor, output is <class 'float'>
n_out = numpy.mean(a_nd*a_nd)					# operation on the numpy
v_out = torch.mean(variable*variable)   		# operation on the autograd.variable, output is <class 'torch.autograd.variable.Variable'>

# to know about the size
print (v_out.size(), tensor.size())

v_out.backward()
print (variable.grad)
# How is this mechanism applied onto loss and variables: https://jhui.github.io/2018/02/09/PyTorch-Variables-functionals-and-Autograd/

"""""""""""""""""""""Some examples"""""""""""""""""""""""""""
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

"""""""""""""""""""""Not every variable has gradients"""""""""""""""""""""""""""
np_array = numpy.array([[1,2],[3,4]])
x = Variable(torch.from_numpy(np_array).float(), requires_grad=True)
y = x + 2
z = y * y * 3
out = z.mean()

out.backward()
print (x.grad)
print ("can't display grad on y and z")


"""""""""""""""""""""how to access data"""""""""""""""""""""""""""
x = torch.randn(3)
x = Variable(x, requires_grad=True)

y = x * 2
while y.data.norm() < 100:
    y = y * 2

"""""""""""""""""""""different backward"""""""""""""""""""""""""""
out = z.mean()
out.backward()    # Same as out.backward(torch.FloatTensor([1.0]))

gradients = torch.FloatTensor([0.1, 1.0, 0.0001])
y.backward(gradients)

