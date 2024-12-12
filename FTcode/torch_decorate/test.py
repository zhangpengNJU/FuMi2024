import torch
from torch.autograd import Function

# 自定义函数，继承自`Function`
class CustomBackward(Function):
    #@staticmethod
    #def forward(ctx, input):
    #    return input

    @staticmethod
    def backward(ctx, *grad_output):
        print("hello")
        return grad_output

# 将自定义的backward函数替换为所有torch算子的默认backward函数
torch.autograd.backward = CustomBackward.apply

# 示例使用
x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
y = torch.tensor([4.0, 5.0, 6.0], requires_grad=True)
z = torch.add(x, y)

z.backward()




import torch
from torch.autograd import Function

# 自定义函数，继承自`Function`
class CustomBackward(Function):
    @staticmethod
    def forward(ctx, input):
        return input

    @staticmethod
    def backward(ctx, grad_output):
        print("hello")
        return grad_output

# 将自定义的backward函数替换为所有torch算子的默认backward函数
torch.autograd.Function.backward = CustomBackward.backward

# 示例使用
x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
y = torch.tensor([4.0, 5.0, 6.0], requires_grad=True)
z = torch.add(x, y)

z.backward()

