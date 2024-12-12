results = dict()
import torch
arg_1_tensor = torch.rand([1, 2560, 4, 4], dtype=torch.float32)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = torch.rand([2560, 512, 3, 3], dtype=torch.float32)
arg_2 = arg_2_tensor.clone()
arg_3_0 = 3
arg_3_1 = 3
arg_3_2 = 6
arg_3_3 = 6
arg_3_4 = 0
arg_3_5 = 1
arg_3 = [arg_3_0,arg_3_1,arg_3_2,arg_3_3,arg_3_4,arg_3_5,]
arg_4 = 7
try:
  results["res_cpu"] = torch.nn.functional.conv2d(arg_1,arg_2,padding=arg_3,groups=arg_4,)
except Exception as e:
  results["err_cpu"] = "ERROR:"+str(e)
arg_1 = arg_1_tensor.clone().cuda()
arg_2 = arg_2_tensor.clone().cuda()
arg_3 = [arg_3_0,arg_3_1,arg_3_2,arg_3_3,arg_3_4,arg_3_5,]
try:
  results["res_gpu"] = torch.nn.functional.conv2d(arg_1,arg_2,padding=arg_3,groups=arg_4,)
except Exception as e:
  results["err_gpu"] = "ERROR:"+str(e)

print(results)
