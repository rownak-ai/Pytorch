import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'
my_tensor = torch.tensor([[1,2,3],[4,5,6]],dtype=torch.float32,
                         device=device,requires_grad=True)
print(my_tensor)

x = torch.empty(size=(3,3))
print(x)
x = torch.zeros((3,3))
print(x)
x = torch.rand((3,3))
print(x)
x = torch.ones((3,3))
print(x)
x = torch.eye(5,5)
print(x)
x = torch.arange(start=0,end=5,step=1)
print(x)
x = torch.linspace(start=0.1,end=1,steps=10)
print(x)
x = torch.empty(size=(1,5)).normal_(mean=0, std=1)
print(x)
x = torch.empty(size=(1,5)).uniform_(0,1)
print(x)
x = torch.diag(torch.ones(3))
print(x)

#Converting tensor to other types
tensor = torch.arange(4)
print(tensor.bool())
print(tensor.short())
print(tensor.long()) #int16 Important
print(tensor.half()) #float16
print(tensor.float()) #float32 Important
print(tensor.double()) #float64

#Array to tensor conversation and vice-versa

import numpy as np
np_array = np.zeros((5,5))
tensor = torch.from_numpy(np_array)
np_back = tensor.numpy()


x = torch.tensor([1,2,3])
y = torch.tensor([9,8,7])

z1 = torch.empty(3)
torch.add(x,y,out=z1)
print(z1)

#Subtraction
z = x-y

#Division
z = torch.true_divide(x,y)

#inplace operations
t = torch.zeros(3)
t.add_(x)
#This is another way
t += x

# Exponentiation
z = x.pow(2)
print(x)
# other way to do
z = x**2

#Simple comparision
z = x > 0
print(z)

x1 = torch.rand((2,5))
x2 = torch.rand((5,3))
x3 = torch.mm(x1,x2)

#matrix in multiplies by itself
matrix_exp = torch.rand(5,5)
matrix_exp.matrix_power(3)

#element wise multiplication
z = x*y
print(z)

z = torch.dot(x,y)
print(z)

#batch matrix multiplication
batch = 32
n = 10
m = 20
p = 30

tensor1 = torch.rand((batch,n,m))
tensor2 = torch.rand((batch,m,p))
out_bmm = torch.bmm(tensor1,tensor2) #(batch,n,p)

# Example of broadcasting
x1 = torch.rand((5,5))
x2 = torch.rand((1,5))

z = x1-x2
z = x1**x2

sum_x = torch.sum(x,dim=0)
values,indices = torch.max(x,dim=0)
values,indices = torch.min(x,dim=0)
abs_x = torch.abs(x)
z = torch.argmax(x,dim=0)
mean_x = torch.mean(x.float(),dim=0)
z = torch.eq(x,y)
sorted_y, indices = torch.sort(y,dim=0,descending=False)

z = torch.clamp(x,min=0)

x = torch.tensor([1,0,0,1],dtype=torch.bool)



batch_size = 10
features = 25
x = torch.rand((batch_size,features))
print(x)
print(x[:,0])

x = torch.arange(10)
indices = [2,5,8]
print(x[indices])

x = torch.rand((3,5))
print(x)
rows = torch.tensor([1,0])
cols = torch.tensor([4,0])
print(x[rows,cols])

x = torch.arange(9)
x_3_3 = x.view(3,3)
print(x_3_3)
x_3_3 = x.reshape(3,3)
print(x_3_3)
y = x_3_3.t()
print(y)
print(y.contiguous().view(9))

x1 = torch.rand((2,5))
x2 = torch.rand((2,5))
print(x1)
print(x2)
print(torch.cat((x1,x2),dim=0))
print(torch.cat((x1,x2),dim=1))
print(torch.cat((x1,x2),dim=0).shape)
print(torch.cat((x1,x2),dim=1).shape)

batch = 64
x = torch.rand((batch,2,5))
z = x.view(batch,-1)
print(z.shape)

z = x.permute(0,2,1)

x = torch.arange(10)
print(x)
print(x.unsqueeze(0))
print(x.unsqueeze(1))

x = torch.arange(10).unsqueeze(0).unsqueeze(1)
print(x)