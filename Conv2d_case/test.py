
import torch
import os
import re

group=-1.566390333137739
torch.set_printoptions(precision=10)
# 读取文件列表
files = os.listdir(base_path)
base_path= '.'
# 读取张量文件
def load_tensors(group):
    tensors = []
    for i in range(7):
        tensors.append(torch.load(os.path.join(base_path, f'agrs_{i}_{group}torch.nn.functional.conv2d.pt')))
    kwargs = torch.load(os.path.join(base_path, f'kwargs_{group}torch.nn.functional.conv2d.pt'))
    result_fuzzed = torch.load(os.path.join(base_path, f'result_fuzzed_{group}torch.nn.functional.conv2d.pt'))
    result_original = torch.load(os.path.join(base_path, f'result_original_{group}torch.nn.functional.conv2d.pt'))
    return tensors, kwargs, result_fuzzed, result_original

tensors, kwargs, r1, r2 = load_tensors(group)


print("fuzzed output in index[15631]:"+ r1[15631])

print("original output in index[15631]:"+ r2[15631])



cx0=tensors[0].to("cpu",dtype=torch.float32)
cx1=tensors[1].to("cpu",dtype=torch.float32)
cx2=tensors[2]
cx3=tensors[3]
cx4=tensors[4]
cx5=tensors[5]
cx6=tensors[6]

cpuresult=torch.nn.functional.conv2d(cx0,cx1,cx2,cx3,cx4,cx5,cx6)
print("CPU output in index[15631]:"+ cpuresult.flatten()[15631])


