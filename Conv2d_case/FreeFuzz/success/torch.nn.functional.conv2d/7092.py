results = dict()
import torch
arg_1_tensor = torch.rand([16, 240, 8, 8], dtype=torch.float32)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = torch.rand([240, 1, 3, 3], dtype=torch.float32)
arg_2 = arg_2_tensor.clone()
arg_3_tensor = torch.rand([240], dtype=torch.float32)
arg_3 = arg_3_tensor.clone()
arg_4_0 = 16
arg_4_1 = 23
arg_4 = [arg_4_0,arg_4_1,]
arg_5_0 = -23
arg_5_1 = -19
arg_5 = [arg_5_0,arg_5_1,]
arg_6_0 = -24
arg_6_1 = 27
arg_6 = [arg_6_0,arg_6_1,]
arg_7 = 240
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