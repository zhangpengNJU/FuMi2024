results = dict()
import torch
arg_1_tensor = torch.rand([128, 88, 32, 32], dtype=torch.float32)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = torch.rand([88, 1, 7, 7], dtype=torch.float32)
arg_2 = arg_2_tensor.clone()
arg_3_tensor = torch.rand([40], dtype=torch.float32)
arg_3 = arg_3_tensor.clone()
arg_4_0 = 58
arg_4_1 = 22
arg_4 = [arg_4_0,arg_4_1,]
arg_5_0 = 3
arg_5_1 = 3
arg_5 = [arg_5_0,arg_5_1,]
arg_6_0 = 7
arg_6_1 = -1024
arg_6 = [arg_6_0,arg_6_1,]
arg_7 = 88
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