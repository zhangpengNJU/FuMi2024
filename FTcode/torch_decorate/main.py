import sys
sys.path.append("..")
from ..fuzz.fuzz_decorate import decorate_functuion


import inspect
from  pathlib import  Path
import torch

#from const import fuzz_config
from torch.jit._state import disable as jit_disable


def fuzz_plugging():
    jit_disable()
    #fuzz_apis=[]
    api_file_path= Path(__file__).parent.joinpath("torch_api.txt")
    with open(api_file_path,"r") as f:
        lines=f.readlines()
        for line in lines:
            hijack(line.strip())





def hijack(func_name_str):
    func_name_list=func_name_str.split(".")
    func_name=func_name_list[-1]
    moudle_obj = torch
    try:
        if len(func_name_list)>2:
            for module_name in func_name_list[1:-1]:
                moudle_obj = getattr(moudle_obj,module_name)
        orig_func = getattr(moudle_obj,func_name)
    except Exception as e:
        print(e)
        return
    if inspect.isclass(orig_func):
        wrapped_obj=orig_func
    elif moudle_obj == torch.Tensor and callable(orig_func):
        wrapped_obj=decorate_functuion(orig_func,func_name_str)
    elif callable(orig_func):
        wrapped_obj = decorate_functuion(orig_func,func_name_str)
    else:
        wrapped_obj=orig_func
    setattr(moudle_obj,func_name,wrapped_obj)