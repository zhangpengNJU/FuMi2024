results = dict()
import torch
arg_1_tensor = torch.rand([128, 80, 32, 32], dtype=torch.float32)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = torch.rand([192, 80, 3, 3], dtype=torch.float32)
arg_2 = arg_2_tensor.clone()
arg_3 = None
arg_4_0 = 35
arg_4_1 = 18
arg_4 = [arg_4_0,arg_4_1,]
arg_5_0 = 1034
arg_5_1 = 71
arg_5 = [arg_5_0,arg_5_1,]
arg_6_0 = -9
arg_6_1 = -40
arg_6 = [arg_6_0,arg_6_1,]
arg_7 = 1
try:
  results["res_cpu"] = torch.nn.functional.conv2d(arg_1,arg_2,arg_3,arg_4,arg_5,arg_6,arg_7,)
except Exception as e:
  results["err_cpu"] = "ERROR:"+str(e)
arg_1 = arg_1_tensor.clone().cuda()
arg_2 = arg_2_tensor.clone().cuda()
arg_4 = [arg_4_0,arg_4_1,]
arg_5 = [arg_5_0,arg_5_1,]
arg_6 = [arg_6_0,arg_6_1,]
try:
  results["res_gpu"] = torch.nn.functional.conv2d(arg_1,arg_2,arg_3,arg_4,arg_5,arg_6,arg_7,)
except Exception as e:
  results["err_gpu"] = "ERROR:"+str(e)

print(results)
