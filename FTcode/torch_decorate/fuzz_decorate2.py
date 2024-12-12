import sys
sys.path.append("..")
import math

import torch
import torch.nn as nn
from typing import Optional, Iterable, Tuple, Any, Callable, List
from ..fuzz import utils
from torch._C import _VariableFunctions as tc
from functools import wraps
from ..fuzz.fuzz_layer import FuzzLayer
from ..fuzz.check_layer import FuncResult
from ..fuzz.check_handler import CheckHandler
from ..fuzz.fix_handler import FixHandler


def type_check(args: Tuple):
    for i,arg in enumerate(args):
        if torch.is_tensor(arg):
            if arg.is_meta:
                continue
            if not torch.is_floating_point(arg):
                continue
            return i
        if isinstance(arg, (Tuple, List)):
            return i
    return -1



ptdir="C:\\Users\\14771\\Desktop\\5\\FuzzTesting\\result"
def decorate_functuion(func, func_id= None):
    @wraps(func)
    def fuzz_wrapper(*args,**kwargs):
        index=type_check(args)
        if index == -1:
            return func(*args,**kwargs)
        #print(func_id+" go into FuzzLayer!")
        fuzz_layer = FuzzLayer(func_id=func_id)
        is_added , fuzzed_input = fuzz_layer(args[index])
        #print(func_id + " go out FuzzLayer!")
        if not is_added:
            return func(*args,**kwargs)
        if index ==0:
            args2=args[1:]
            if not args2:
                result_fuzzed=func(fuzzed_input,**kwargs)
            else:
                #print("type",type(fuzzed_input),type(args[index]))
                result_fuzzed = func(fuzzed_input, *args2, **kwargs)
        else:
            args2_front=args[:index]
            args2_rear=args[index+1:]
            if not args2_rear:
                result_fuzzed = func(*args2_front,fuzzed_input, ** kwargs)
            else:
                result_fuzzed = func(*args2_front, fuzzed_input, *args2_rear,**kwargs)
        result_original = func(*args,**kwargs)
        func_result=FuncResult(result_original,result_fuzzed)
        handler=CheckHandler()
        #handler = FixHandler()
        try:
            e = set_e(result_original)
        except:
            e = 1e-3
        result = handler.handle(args, kwargs, func_id, func_result, ptdir, e)
        return result

    return fuzz_wrapper

def set_e(inputs):
        if str(inputs.dtype) == "torch.float64":
            return 1e-8
        if str(inputs.dtype) == "torch.float32":
            return 1e-4
        if str(inputs.dtype) == "torch.float16":
            return 1e-3
        if str(inputs.dtype) == "torch.bfloat16":
            return 1e-2


