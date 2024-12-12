results = dict()
import torch
arg_1_tensor = torch.rand([80, 512, 16, 16], dtype=torch.float32)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = torch.rand([512, 16, 3], dtype=torch.complex128)
arg_2 = arg_2_tensor.clone()
arg_3 = None
arg_4_0 = 2
arg_4_1 = 2
arg_4 = [arg_4_0,arg_4_1,]
arg_5_0 = 0
arg_5_1 = 4
arg_5_2 = 2
arg_5 = [arg_5_0,arg_5_1,arg_5_2,]
arg_6_0 = 73
arg_6_1 = -1097
arg_6 = [arg_6_0,arg_6_1,]
arg_7 = 1
try:
  results["res_cpu"] = torch.nn.functional.conv2d(arg_1,arg_2,arg_3,arg_4,arg_5,arg_6,arg_7,)
except Exception as e:
  results["err_cpu"] = "ERROR:"+str(e)
arg_1 = arg_1_tensor.clone().cuda()
arg_2 = arg_2_tensor.clone().cuda()
arg_4 = [arg_4_0,arg_4_1,]
arg_5 = [arg_5_0,arg_5_1,arg_5_2,]
arg_6 = [arg_6_0,arg_6_1,]
try:
  results["res_gpu"] = torch.nn.functional.conv2d(arg_1,arg_2,arg_3,arg_4,arg_5,arg_6,arg_7,)
except Exception as e:
  results["err_gpu"] = "ERROR:"+str(e)

print(results)
