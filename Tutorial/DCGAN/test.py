import torch

# x = torch.zeros(2,1)
# print (x)
# y = torch.squeeze(x, 1)
# print (y)
# a = torch.FloatTensor(1, 7)
# a.fill_(3.5)


# b = a.add(4.0)
# print (a)
# print (b)
# c = a.add_(5.0)
# print (a)

# class Base(object):		# inherit from object class
#     def __init__(self, name):
#     	self.name = name
#     	print ('base created!', name)


# class ChildA(Base):		# using parent
#     def __init__(self, name):
#         Base.__init__(self, name=name)


# class ChildB(Base):		# using parent
#     def __init__(self, B_name):
#         super(ChildB, self).__init__(name = B_name)

# print (ChildA(name='ChildA'))
# print (ChildB(B_name='ChildB'))
# print (Base(name='dog'))

A = 10
B = 6
print (A/B)
print (A//B)
