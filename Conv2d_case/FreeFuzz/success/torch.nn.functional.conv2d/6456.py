results = dict()
import torch
arg_1_tensor = torch.randint(0,2,[5, 0, 130, 130], dtype=torch.bool)
arg_1 = arg_1_tensor.clone()
arg_2 = None
arg_3 = 29
arg_4 = 16
arg_5 = 1
try:
  results["res_cpu"] = torch.nn.functional.conv2d(arg_1,arg_2,groups=arg_3,padding=arg_4,stride=arg_5,)
except Exception as e:
  results["err_cpu"] = "ERROR:"+str(e)
arg_1 = arg_1_tensor.clone().cuda()
try:
  results["res_gpu"] = torch.nn.functional.conv2d(arg_1,arg_2,groups=arg_3,padding=arg_4,stride=arg_5,)
except Exception as e:
  results["err_gpu"] = "ERROR:"+str(e)

print(results)
