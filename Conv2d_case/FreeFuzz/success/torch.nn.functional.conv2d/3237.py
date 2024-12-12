results = dict()
import torch
arg_1_tensor = torch.rand([5, 64, 130, 130], dtype=torch.float32)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = torch.rand([64, 1, 3, 3], dtype=torch.float32)
arg_2 = arg_2_tensor.clone()
arg_3 = 64
arg_4_0 = False
arg_4_1 = -73.0
arg_4_2 = "max"
arg_4_3 = "max"
arg_4_4 = True
arg_4_5 = -14.0
arg_4 = [arg_4_0,arg_4_1,arg_4_2,arg_4_3,arg_4_4,arg_4_5,]
arg_5 = 1
try:
  results["res_cpu"] = torch.nn.functional.conv2d(arg_1,arg_2,groups=arg_3,padding=arg_4,stride=arg_5,)
except Exception as e:
  results["err_cpu"] = "ERROR:"+str(e)
arg_1 = arg_1_tensor.clone().cuda()
arg_2 = arg_2_tensor.clone().cuda()
arg_4 = [arg_4_0,arg_4_1,arg_4_2,arg_4_3,arg_4_4,arg_4_5,]
try:
  results["res_gpu"] = torch.nn.functional.conv2d(arg_1,arg_2,groups=arg_3,padding=arg_4,stride=arg_5,)
except Exception as e:
  results["err_gpu"] = "ERROR:"+str(e)

print(results)
