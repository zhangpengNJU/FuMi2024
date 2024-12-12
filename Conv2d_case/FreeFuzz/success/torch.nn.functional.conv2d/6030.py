results = dict()
import torch
arg_1_tensor = torch.rand([1, 160, 128, 128], dtype=torch.float32)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = torch.rand([160, 32, 3, 3], dtype=torch.float32)
arg_2 = arg_2_tensor.clone()
arg_3_0 = 1
arg_3_1 = 1
arg_3_2 = 2
arg_3_3 = 0
arg_3 = [arg_3_0,arg_3_1,arg_3_2,arg_3_3,]
arg_4 = "max"
try:
  results["res_cpu"] = torch.nn.functional.conv2d(arg_1,arg_2,padding=arg_3,groups=arg_4,)
except Exception as e:
  results["err_cpu"] = "ERROR:"+str(e)
arg_1 = arg_1_tensor.clone().cuda()
arg_2 = arg_2_tensor.clone().cuda()
arg_3 = [arg_3_0,arg_3_1,arg_3_2,arg_3_3,]
try:
  results["res_gpu"] = torch.nn.functional.conv2d(arg_1,arg_2,padding=arg_3,groups=arg_4,)
except Exception as e:
  results["err_gpu"] = "ERROR:"+str(e)

print(results)