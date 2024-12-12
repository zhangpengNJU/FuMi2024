results = dict()
import torch
arg_1_tensor = torch.rand([16, 352, 16, 40], dtype=torch.complex128)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = torch.rand([88, 352, 1, 1], dtype=torch.float32)
arg_2 = arg_2_tensor.clone()
arg_3 = None
arg_4_0 = 24
arg_4_1 = -43
arg_4 = [arg_4_0,arg_4_1,]
arg_5_0 = -40
arg_5_1 = 39
arg_5 = [arg_5_0,arg_5_1,]
arg_6_0 = -1024
arg_6_1 = 53
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
