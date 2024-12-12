
import torch
import os
import re

x=torch.load('/home/users/pzhang/0528/predoo/conv2dpt/12conv2d.pt')
con16=torch.load('/home/users/pzhang/0528/predoo/conv2dpt/12conv2dl.pt')
x_32 = input_withDiffDype(x, torch.float32)
x_16 = input_withDiffDype(x, torch.float16)
x_64 = input_withDiffDype(x, torch.float64).double()
torch_conv_16 = con16
original_weights = torch_conv_16.weight
original_bias = torch_conv_16.bias
float64_weights = original_weights.double()
float64_bias = original_bias.double()
#gpu,64:
#tensor(2.0993984830, device='cuda:0', dtype=torch.float64,       grad_fn=<SelectBackward0>)
torch_conv_64 = nn.Conv2d(3, 8, kernel_size=(3, 3), stride=(2, 2)).cuda()
torch_conv_64.weight = nn.Parameter(float64_weights)
torch_conv_64.bias = nn.Parameter(float64_bias)
out_64_64 = torch_conv_64(x_64)
out_64_64.flatten()[39]
#cpu,64:
#tensor(2.0993984830, dtype=torch.float64, grad_fn=<SelectBackward0>)
cputorch_conv_64 = nn.Conv2d(3, 8, kernel_size=(3, 3), stride=(2, 2)).cpu()
cputorch_conv_64.weight = nn.Parameter(float64_weights.cpu())
cputorch_conv_64.bias = nn.Parameter(float64_bias.cpu())
cpuout_64_64 = cputorch_conv_64(x_64.cpu())
cpuout_64_64.flatten()[39]


#gpu,16:
#tensor(2.1015625000, device='cuda:0', dtype=torch.float16,grad_fn=<SelectBackward0>)
torch_conv_16.weight = nn.Parameter(original_weights.half().cuda())
torch_conv_16.bias = nn.Parameter(original_bias.half().cuda())
out_16_16 = torch_conv_16(x_16)
out_16_16.flatten()[39]

#cpu, 16:
#tensor(2.0996093750, dtype=torch.float16, grad_fn=<SelectBackward0>)
cputorch_conv_16 = nn.Conv2d(3, 8, kernel_size=(3, 3), stride=(2, 2)).cpu()
cputorch_conv_16.weight = nn.Parameter(original_weights.half().cpu())
cputorch_conv_16.bias = nn.Parameter(original_bias.half().cpu())
cpuout_16_16 = cputorch_conv_16(x_16.cpu())
cpuout_16_16.flatten()[39]


