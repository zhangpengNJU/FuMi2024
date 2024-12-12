results = dict()
import torch
arg_1_tensor = torch.rand([16, 576, 5, 5], dtype=torch.float32)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = torch.rand([1], dtype=torch.float32)
arg_2 = arg_2_tensor.clone()
arg_3_tensor = torch.rand([96], dtype=torch.float32)
arg_3 = arg_3_tensor.clone()
arg_4_0 = 2
arg_4_1 = 1
arg_4 = [arg_4_0,arg_4_1,]
arg_5_0 = 0
arg_5_1 = 1041
arg_5 = [arg_5_0,arg_5_1,]
arg_6_0 = 1024.0
arg_6_1 = -1024.0
arg_6 = [arg_6_0,arg_6_1,]
arg_7 = -4
try:
  results["res_cpu"] = torch.nn.functional.conv2d(arg_1,arg_2,arg_3,arg_4,arg_5,arg_6,arg_7,)
except Exception as e:
  results["err_cpu"] = "ERROR:"+str(e)
arg_1 = arg_1_tensor.clone().cuda()
arg_2 = arg_2_tensor.clone().cuda()
arg_3 = arg_3_tensor.clone().cuda()
arg_4 = [arg_4_0,arg_4_1,]
arg_5 = [arg_5_0,arg_5_1,]
arg_6 = [arg_6_0,arg_6_1,]
try:
  results["res_gpu"] = torch.nn.functional.conv2d(arg_1,arg_2,arg_3,arg_4,arg_5,arg_6,arg_7,)
except Exception as e:
  results["err_gpu"] = "ERROR:"+str(e)

print(results)
