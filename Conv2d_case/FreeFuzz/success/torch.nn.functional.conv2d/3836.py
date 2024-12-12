results = dict()
import torch
arg_1_tensor = torch.rand([1, 320, 64, 64], dtype=torch.float32)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = torch.randint(-4,64,[15, 64, 1, 0], dtype=torch.int32)
arg_2 = arg_2_tensor.clone()
arg_3_0 = 3
arg_3_1 = 1
arg_3 = [arg_3_0,arg_3_1,]
arg_4 = 5
try:
  results["res_cpu"] = torch.nn.functional.conv2d(arg_1,arg_2,padding=arg_3,groups=arg_4,)
except Exception as e:
  results["err_cpu"] = "ERROR:"+str(e)
arg_1 = arg_1_tensor.clone().cuda()
arg_2 = arg_2_tensor.clone().cuda()
arg_3 = [arg_3_0,arg_3_1,]
try:
  results["res_gpu"] = torch.nn.functional.conv2d(arg_1,arg_2,padding=arg_3,groups=arg_4,)
except Exception as e:
  results["err_gpu"] = "ERROR:"+str(e)

print(results)
