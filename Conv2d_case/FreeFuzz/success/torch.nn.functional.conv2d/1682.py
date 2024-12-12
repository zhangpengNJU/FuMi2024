results = dict()
import torch
arg_1_tensor = torch.randint(-128,2048,[3, 6], dtype=torch.int64)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = torch.rand([3, 32, 9, 9], dtype=torch.float32)
arg_2 = arg_2_tensor.clone()
arg_3_tensor = torch.rand([3], dtype=torch.float32)
arg_3 = arg_3_tensor.clone()
arg_4_0 = 16
arg_4_1 = 60
arg_4 = [arg_4_0,arg_4_1,]
arg_5 = 1024
arg_6_0 = -11
arg_6_1 = -32
arg_6 = [arg_6_0,arg_6_1,]
arg_7 = 1
try:
  results["res_cpu"] = torch.nn.functional.conv2d(arg_1,arg_2,arg_3,arg_4,arg_5,arg_6,arg_7,)
except Exception as e:
  results["err_cpu"] = "ERROR:"+str(e)
arg_1 = arg_1_tensor.clone().cuda()
arg_2 = arg_2_tensor.clone().cuda()
arg_3 = arg_3_tensor.clone().cuda()
arg_4 = [arg_4_0,arg_4_1,]
arg_6 = [arg_6_0,arg_6_1,]
try:
  results["res_gpu"] = torch.nn.functional.conv2d(arg_1,arg_2,arg_3,arg_4,arg_5,arg_6,arg_7,)
except Exception as e:
  results["err_gpu"] = "ERROR:"+str(e)

print(results)
