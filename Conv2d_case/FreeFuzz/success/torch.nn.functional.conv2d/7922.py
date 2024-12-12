results = dict()
import torch
arg_1_tensor = torch.rand([4], dtype=torch.complex64)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = torch.rand([160, 64, 3, 3], dtype=torch.float32)
arg_2 = arg_2_tensor.clone()
arg_3 = 1
arg_4 = 54
try:
  results["res_cpu"] = torch.nn.functional.conv2d(arg_1,arg_2,padding=arg_3,groups=arg_4,)
except Exception as e:
  results["err_cpu"] = "ERROR:"+str(e)
arg_1 = arg_1_tensor.clone().cuda()
arg_2 = arg_2_tensor.clone().cuda()
try:
  results["res_gpu"] = torch.nn.functional.conv2d(arg_1,arg_2,padding=arg_3,groups=arg_4,)
except Exception as e:
  results["err_gpu"] = "ERROR:"+str(e)

print(results)