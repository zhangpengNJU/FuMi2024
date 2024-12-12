import torch
from functools import  wraps
# 自定义backward函数
def custom_backward(grad_fn, *grad_outputs, retain_graph=False):
    print("hello")
    grad_fn(*grad_outputs, retain_graph=retain_graph)

# 替换torch.add算子的backward函数
def custom_add_backward(*args):
    custom_backward(torch.Tensor._add_backward, *args)

# 将替换后的backward函数注册给torch.add算子
torch.Tensor._add_backward = custom_add_backward

# 示例使用
x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
y = torch.tensor([4.0, 5.0, 6.0], requires_grad=True)
z = torch.add(x, y)

# 执行backward时会先打印"hello"
z.backward()


def decorate_backward_function(api_name, func):
    class CustomFunction(torch.autograd.Function):
        @staticmethod
        def forward(ctx, *args):
            _tensor_index =[]
            _scale_index =[]
            _tensor_data = []
            for i ,v in enumerate(args):
                if torch.is_tensor(v):
                    _tensor_index.append(i)
                    _tensor_data.append(v)
                else:
                    _scale_index.append((i,v))
            ctx.my_tensor_index=tuple(_tensor_index)
            ctx.my_scale_data = tuple(_scale_index)
            ctx.save_for_backward(*_tensor_data)
            return func(*args)

        @staticmethod
        def backward(ctx, *grad_outputs):
            return custom_backward(api_name,ctx,*grad_outputs)



    @wraps(func)
    def custom_function(*args, **kwargs):
        try:
            return CustomFunction.apply(*args,**kwargs)
        except TypeError:
            return func(*args, **kwargs)

    return custom_function




