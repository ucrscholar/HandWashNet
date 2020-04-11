from __future__ import print_function
import torch

import numpy as np
kernel_init = np.array(
    [
     [[[-1, 1.0/9, 0]],[[-1, 1.0/9, -1]],[[-1, 1.0/9, 0]]],
     [[[-1, 1.0/9, -1]],[[8, 1.0/9,5]],[[-1, 1.0/9,-1]]],
     [[[-1, 1.0/9,0]],[[-1, 1.0/9,-1]],[[-1, 1.0/9, 0]]]
     ])


x = torch.empty(5, 3)
print(x)

x = torch.rand(5, 3)
print(x)

x = torch.zeros(5, 3, dtype=torch.long)
print(x)

y = torch.rand(5, 3,4)
print(y)

a = torch.ones(5)
print(a)

b = a.numpy()
print(b)

if torch.cuda.is_available():
    device = torch.device("cuda")          # a CUDA device object
    y = torch.ones_like(x, device=device)  # directly create a tensor on GPU
    x = x.to(device)                       # or just use strings ``.to("cuda")``
    z = x + y
    print(z)
    print(z.to("cpu", torch.double))       # ``.to`` can also change dtype together!


x = torch.ones(2, 2, requires_grad=True)
print(x)
y = x + 2
print(y)

z = y * y * 3
out = z.mean()

print(z, out)

a = torch.randn(2, 2)
a = ((a * 3) / (a - 1))
print(a.requires_grad)
a.requires_grad_(True)
print(a.requires_grad)
b = (a * a).sum()
print(b.grad_fn)

out.backward()

print(x.grad)

'''X=torch.randn((100,3))
Y=torch.randn((100))
w1=torch.tensor(0.1, requires_grad=True)
w2=torch.tensor(0.1, requires_grad=True)
w3=torch.tensor(0.1, requires_grad=True)
W=torch.empty(3, requires_grad=False)
W[0]=w1 * w2; W[1]=w2 * w3; W[2]=w3 * w1
Yp=torch.sum(X*W, dim=1)
loss = torch.nn.MSELoss()(Yp, Y)
loss.backward()
print(w1.grad)'''


x = torch.randn(3, requires_grad=True)

y = x * 2
while y.data.norm() < 1000:
    y = y * 2

print(y)

gradients = torch.tensor([0.1, 1.0, 0.0001], dtype=torch.float)
y.backward(gradients)

print(x.grad)


import torch
import torch.nn as nn
import torch.nn.functional as F


m = nn.Linear(20, 30)
input = torch.randn(128, 20)
output = m(input)
print(output.size())

x = torch.randn(128, 20) # 输入的维度是（128，20）
print(x.size())
m = torch.nn.Linear(20, 30) # 20,30是指维度
output = m(x)
print('m.weight.shape:\n ', m.weight.shape)
print('m.bias.shape:\n', m.bias.shape)
print('output.shape:\n', output.shape)

# ans = torch.mm(input,torch.t(m.weight))+m.bias 等价于下面的
ans = torch.mm(x, m.weight.t()) + m.bias
print('ans.shape:\n', ans.shape)

print(torch.equal(ans, output))


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # 1 input image channel, 6 output channels, 3x3 square convolution
        # kernel
        self.conv1 = nn.Conv2d(1, 6, 3)
        self.conv2 = nn.Conv2d(6, 16, 3)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16 * 6 * 6, 120)  # 6*6 from image dimension
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # If the size is a square you can only specify a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


net = Net()
print(net)

params = list(net.parameters())
print(len(params))
print(params[0].size())  # conv1's .weight

input = torch.randn(1, 1, 32, 32)
out = net(input)
print(out)

net.zero_grad()
out.backward(torch.randn(1, 10))

output = net(input)
target = torch.randn(10)  # a dummy target, for example
target = target.view(1, -1)  # make it the same shape as output
criterion = nn.MSELoss()

loss = criterion(output, target)
print(loss)


print(loss.grad_fn)  # MSELoss
print(loss.grad_fn.next_functions[0][0])  # Linear
print(loss.grad_fn.next_functions[0][0].next_functions[0][0])  # ReLU

net.zero_grad()     # zeroes the gradient buffers of all parameters

print('conv1.bias.grad before backward')
print(net.conv1.bias.grad)

loss.backward()

print('conv1.bias.grad after backward')
print(net.conv1.bias.grad)
print('finished')

import torch.optim as optim

# create your optimizer
optimizer = optim.SGD(net.parameters(), lr=0.01)
optimizer.step()    # Does the update