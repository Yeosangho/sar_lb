import copy
class A():
	def __init__(self, a):
		self.a = a
As = []
for i in range(4):
	classA = A(i)
	As.append(classA)
for i in range(4):
	print(As[i].a)
