results = dict()
import torch
arg_1_tensor = torch.rand([16, 6, 256], dtype=torch.float32)
arg_1 = arg_1_tensor.clone()
arg_2 = None
arg_3_tensor = torch.rand([100], dtype=torch.float32)
arg_3 = arg_3_tensor.clone()
arg_4_0 = 1
arg_4_1 = 1
arg_4 = [arg_4_0,arg_4_1,]
arg_5_0 = 0
arg_5_1 = 0
arg_5 = [arg_5_0,arg_5_1,]
arg_6_0 = 9
arg_6_1 = 16
arg_6 = [arg_6_0,arg_6_1,]
arg_7 = -50
try:
  results["res_cpu"] = torch.nn.functional.conv2d(arg_1,arg_2,arg_3,arg_4,arg_5,arg_6,arg_7,)
except Exception as e:
  results["err_cpu"] = "ERROR:"+str(e)
arg_1 = arg_1_tensor.clone().cuda()
arg_3 = arg_3_tensor.clone().cuda()
arg_4 = [arg_4_0,arg_4_1,]
arg_5 = [arg_5_0,arg_5_1,]
arg_6 = [arg_6_0,arg_6_1,]
try:
  results["res_gpu"] = torch.nn.functional.conv2d(arg_1,arg_2,arg_3,arg_4,arg_5,arg_6,arg_7,)
except Exception as e:
  results["err_gpu"] = "ERROR:"+str(e)

print(results)
