import torch
import numpy as np
import scipy
import matplotlib.pyplot as plt
from torch.autograd import Variable

# In tensorflow, you will have placeholder/ tensor, then you do the calculation (sess)
# In Torch, you just also use tensor holding value, you don't need sess.run and placeholder you can get result

# Conversion is from numpy to tensor, from tensor to variable(with or without autograd)

np_data = np.arange(6).reshape((2, 3))		# (2,3), ndarray
torch_data = torch.from_numpy(np_data)      # this is a tensor, tensor and numpy can convert to each other
											# torch.Size([2, 3]), <class 'torch.LongTensor'>

# torch_input = Variable(np_data)		# RuntimeError: Variable data has to be a tensor, but got numpy.ndarray
torch_input = Variable(torch_data.float())      # this is a grad.variable
												# torch.Size([2, 3]), <class 'torch.autograd.variable.Variable'>

tensor2array = torch_data.numpy()			# np_data is equal to tensor2array
variable2array = torch_input.data.numpy()	# np_data is equal to variable2array	
variable2tensor = torch_input.data 			# from variable to tensor

# print (type(tensor2array))
# print (type(variable2array))
# print (type(variable2tensor))

# print(
#     '\nnumpy array:', np_data,          # [[0 1 2], [3 4 5]]
#     '\ntorch tensor:', torch_data,      # 0  1  2 \n 3  4  5    [torch.LongTensor of size 2x3]
#     '\ntorch variable:', torch_input,	  # variable containing: 0  1  2 \n 3  4  5 
#     '\ntensor to array:', tensor2array, # [[0 1 2], [3 4 5]]
# )

# # abs 绝对值计算
# data = [-1, -2, 1, 2]
# tensor = torch.FloatTensor(data)  # 转换成32位浮点 tensor
# variable = Variable(tensor)		# still variable containing

# print(
#	  '\n This proves torch.abs can be operator on tensor and variable'
#     '\nabs',
#     '\nnumpy: ', np.abs(data),          # [1 2 1 2]
#     '\ntorch: ', torch.abs(tensor),      # [1 2 1 2]
#     '\nvariable: ', torch.abs(variable)		# [1 2 1 2]
# )

# sin   三角函数 sin
# print(
#     '\nsin',
#     '\nnumpy: ', np.sin(data),      # [-0.84147098 -0.90929743  0.84147098  0.90929743]
#     '\ntorch: ', torch.sin(tensor),  # [-0.8415 -0.9093  0.8415  0.9093]
#     '\nvariable: ', torch.sin(variable)	# variable containing:  [-0.8415 -0.9093  0.8415  0.9093]
# )

# mean  均值
# print(
#     '\nmean',
#     '\nnumpy: ', np.mean(data),         # 0.0
#     '\ntorch: ', torch.mean(tensor),     # 0.0
#     '\nvariable: ', torch.mean(variable)
# )

# # matrix multiplication 矩阵点乘
# data = [[1,2], [3,4]]
# tensor = torch.FloatTensor(data)  # 转换成32位浮点 tensor
# variable = Variable(tensor)
# # correct method
# print(
#     '\nmatrix multiplication (matmul)',
#     '\nnumpy: ', np.matmul(data, data),     # [[7, 10], [15, 22]]
#     '\ntorch: ', torch.mm(tensor, tensor),   # [[7, 10], [15, 22]]
#     '\nvariable: ', torch.mm(variable, variable)
# )

# # !!!!  下面是错误的方法 !!!!
# data = np.array(data)
# print(
#     '\nmatrix multiplication (dot)',
#     '\nnumpy: ', data.dot(data),        # [[7, 10], [15, 22]] 在numpy 中可行
#     '\ntorch: ', tensor.dot(tensor)     # torch 会转换成 [1,2,3,4].dot([1,2,3,4) = 30.0
# )
