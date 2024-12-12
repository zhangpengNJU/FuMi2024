results = dict()
import torch
arg_1_tensor = torch.rand([5, 3, 18, 18], dtype=torch.float32)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = torch.rand([3, 1, 3, 3], dtype=torch.float32)
arg_2 = arg_2_tensor.clone()
arg_3 = 93
arg_4 = -42
arg_5 = 1
try:
  results["res_cpu"] = torch.nn.functional.conv2d(arg_1,arg_2,groups=arg_3,padding=arg_4,stride=arg_5,)
except Exception as e:
  results["err_cpu"] = "ERROR:"+str(e)
arg_1 = arg_1_tensor.clone().cuda()
arg_2 = arg_2_tensor.clone().cuda()
try:
  results["res_gpu"] = torch.nn.functional.conv2d(arg_1,arg_2,groups=arg_3,padding=arg_4,stride=arg_5,)
except Exception as e:
  results["err_gpu"] = "ERROR:"+str(e)

print(results)
