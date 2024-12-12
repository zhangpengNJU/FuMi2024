results = dict()
import torch
arg_1_tensor = torch.randint(0,2,[1, 112, 28, 1], dtype=torch.bool)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = torch.rand([2, 64, 1, 1], dtype=torch.float32)
arg_2 = arg_2_tensor.clone()
arg_3_tensor = torch.rand([2], dtype=torch.float32)
arg_3 = arg_3_tensor.clone()
arg_4_0 = 2
arg_4_1 = 1
arg_4_2 = 1
arg_4 = [arg_4_0,arg_4_1,arg_4_2,]
arg_5 = 0
arg_6_0 = "max"
arg_6_1 = "sum"
arg_6 = [arg_6_0,arg_6_1,]
arg_7 = 1
try:
  results["res_cpu"] = torch.nn.functional.conv2d(arg_1,arg_2,arg_3,arg_4,arg_5,arg_6,arg_7,)
except Exception as e:
  results["err_cpu"] = "ERROR:"+str(e)
arg_1 = arg_1_tensor.clone().cuda()
arg_2 = arg_2_tensor.clone().cuda()
arg_3 = arg_3_tensor.clone().cuda()
arg_4 = [arg_4_0,arg_4_1,arg_4_2,]
arg_6 = [arg_6_0,arg_6_1,]
try:
  results["res_gpu"] = torch.nn.functional.conv2d(arg_1,arg_2,arg_3,arg_4,arg_5,arg_6,arg_7,)
except Exception as e:
  results["err_gpu"] = "ERROR:"+str(e)

print(results)
