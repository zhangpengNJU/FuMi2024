import math

import torch
import torch.nn as nn
from typing import Optional, Iterable, Tuple, Any, Callable, List

from ..fuzz import utils
from torch._C import _VariableFunctions as tc
from functools import wraps
from ..fuzz.fuzz_layer import FuzzLayer, FuzzLayer3
from ..fuzz.check_layer import FuncResult
from ..fuzz.check_handler import CheckHandler
from pathlib import Path
from ..fuzz.fix_handler import FixHandler

import os
import sys

env_file = "/home/users/pzhang/0528/FuMi/FuMi-9a261abea107a7f579cbcaf6308e03c112116e6f/FTcode/torch_decorate/config.env"
# env_file = os.path.dirname(os.path.realpath(sys.argv[0])).parent.joinpath("torch_decorate").joinpath("config.env")
with open(env_file, "r") as file:
    # 读取第一行
    first_line = file.readline()

perturbation_mode = first_line.split('"')[-2]


def type_check(args: Tuple):
    for i, arg in enumerate(args):
        if torch.is_tensor(arg):
            if arg.is_meta:
                continue
            if not torch.is_floating_point(arg):
                continue
            return i
        if isinstance(arg, (Tuple, List)):
            return i
    return -1


def type_check_all(args: Tuple):
    res = []
    for i, arg in enumerate(args):
        if torch.is_tensor(arg):
            if arg.is_meta:
                continue
            if not torch.is_floating_point(arg):
                continue
            res.append(i)
        if isinstance(arg, (Tuple, List)):
            res.append(i)
    if res == []:
        return -1
    return res


ptdir = "/home/users/pzhang/0528/FuMiOutput/pt"


if perturbation_mode == "custom_function":
  def decorate_functuion(func_id=None, newfunc=None):
    def decorator(func):
        @wraps(func)
        def fuzz_wrapper(*args, **kwargs):
            index = type_check(args)
            if index == -1:
                return func(*args, **kwargs)
            if perturbation_mode == "custom_function":
                result_original = func(*args, **kwargs)
                print(type(newfunc),type(func),func,newfunc)
                result_fuzzed = newfunc(*args, **kwargs)
                func_result = FuncResult(result_original, result_fuzzed)
                handler = CheckHandler()
                try:
                    e = set_e(result_original)
                except:
                    e = 1e-3
                result = handler.handle(args, kwargs, func_id, func_result, ptdir, e)
                return result
        return fuzz_wrapper
    return decorator




if perturbation_mode == "add_noise" or perturbation_mode == "improve_precision":
  def decorate_functuion(func, newfunc=None, func_id=None ):
    @wraps(func)
    def fuzz_wrapper(*args, **kwargs):
        index = type_check(args)
        if index == -1:
            return func(*args, **kwargs)
        # print(func_id+" go into FuzzLayer!")
        if perturbation_mode == "add_noise":
            fuzz_layer = FuzzLayer(func_id=func_id)
        if perturbation_mode == "custom_function":
            result_original = func(*args, **kwargs)
            print(type(newfunc),type(func),func,newfunc)
            result_fuzzed = newfunc(*args, **kwargs)
            func_result = FuncResult(result_original, result_fuzzed)
            handler = CheckHandler()
            try:
                    e = set_e(result_original)
            except:
                    e = 1e-3
            result = handler.handle(args, kwargs, func_id, func_result, ptdir, e)
            return result
        if perturbation_mode == "improve_precision":
            indexes = type_check_all(args)
            fuzz_layer = FuzzLayer3(func_id=func_id)
            #提升精度模式下，要对全部可提升的都提升。
            try:
                is_added=[]
                fuzzed_input=[]
                for ind in indexes:
                    is_adde, fuzzed_inpu = fuzz_layer(args[ind])
                    is_added.append(is_adde)
                    fuzzed_input.append(fuzzed_inpu)
                if True not in is_added:
                    return func(*args, **kwargs)
                if indexes == [0]:
                    args2 = args[1:]
                    if not args2:
                        result_fuzzed = func(fuzzed_input[0], **kwargs)
                    else:
                        # print("type", type(fuzzed_input), type(args[index]))
                        result_fuzzed = func(fuzzed_input[0], *args2, **kwargs)
                else:
                    newarg=()
                    #print(is_added)
                    for i in range(len(args)):
                        if i in indexes:
                            if is_added[indexes.index(i)]:
                                newarg+=(fuzzed_input[indexes.index(i)],)
                            else:
                                newarg +=(args[i],)
                        else:
                            newarg += (args[i],)
                    result_fuzzed = func(*newarg, **kwargs)
                result_original = func(*args, **kwargs)
                func_result = FuncResult(result_original, result_fuzzed)
                handler = CheckHandler()
                try:
                    e = set_e(result_original)
                except:
                    e = 1e-3
                result = handler.handle(args, kwargs, func_id, func_result, ptdir, e)
                return result
            except Exception as s:
                print(s)
                return func(*args, **kwargs)
        try:
            is_added, fuzzed_input = fuzz_layer(args[index])
            # print(func_id + " go out FuzzLayer!")
            if not is_added:
                return func(*args, **kwargs)
            if index == 0:
                args2 = args[1:]
                if not args2:
                    result_fuzzed = func(fuzzed_input, **kwargs)
                else:
                    # print("type", type(fuzzed_input), type(args[index]))
                    result_fuzzed = func(fuzzed_input, *args2, **kwargs)
            else:
                args2_front = args[:index]
                args2_rear = args[index + 1:]
                if not args2_rear:
                    result_fuzzed = func(*args2_front, fuzzed_input, **kwargs)
                else:
                    result_fuzzed = func(*args2_front, fuzzed_input, *args2_rear, **kwargs)
            result_original = func(*args, **kwargs)
            func_result = FuncResult(result_original, result_fuzzed)
            handler = CheckHandler()
        except Exception as s:
            print(s)
            return func(*args, **kwargs)
        # handler = FixHandler()
        try:
            e = set_e(result_original)
        except:
            e = 1e-3
        result = handler.handle(args, kwargs, func_id, func_result, ptdir, e)
        return result
    return fuzz_wrapper



'''

def decorate_functuion(func_id=None, newfunc=None):
  def decorator(func):
    @wraps(func)
    def fuzz_wrapper(*args, **kwargs):
        index = type_check(args)
        if index == -1:
            return func(*args, **kwargs)
        # print(func_id+" go into FuzzLayer!")
        if perturbation_mode == "add_noise":
            fuzz_layer = FuzzLayer(func_id=func_id)
        if perturbation_mode == "custom_function":
            result_original = func(*args, **kwargs)
            result_fuzzed = newfunc(*args, **kwargs)
            func_result = FuncResult(result_original, result_fuzzed)
            handler = CheckHandler()
            try:
                    e = set_e(result_original)
            except:
                    e = 1e-3
            result = handler.handle(args, kwargs, func_id, func_result, ptdir, e)
            return result
        if perturbation_mode == "improve_precision":
            indexes = type_check_all(args)
            fuzz_layer = FuzzLayer3(func_id=func_id)
            #提升精度模式下，要对全部可提升的都提升。
            try:
                is_added=[]
                fuzzed_input=[]
                for ind in indexes:
                    is_adde, fuzzed_inpu = fuzz_layer(args[ind])
                    is_added.append(is_adde)
                    fuzzed_input.append(fuzzed_inpu)
                if True not in is_added:
                    return func(*args, **kwargs)
                if indexes == [0]:
                    args2 = args[1:]
                    if not args2:
                        result_fuzzed = func(fuzzed_input[0], **kwargs)
                    else:
                        # print("type", type(fuzzed_input), type(args[index]))
                        result_fuzzed = func(fuzzed_input[0], *args2, **kwargs)
                else:
                    newarg=()
                    #print(is_added)
                    for i in range(len(args)):
                        if i in indexes:
                            if is_added[indexes.index(i)]:
                                newarg+=(fuzzed_input[indexes.index(i)],)
                            else:
                                newarg +=(args[i],)
                        else:
                            newarg += (args[i],)
                    result_fuzzed = func(*newarg, **kwargs)
                result_original = func(*args, **kwargs)
                func_result = FuncResult(result_original, result_fuzzed)
                handler = CheckHandler()
                try:
                    e = set_e(result_original)
                except:
                    e = 1e-3
                result = handler.handle(args, kwargs, func_id, func_result, ptdir, e)
                return result
            except Exception as s:
                print(s)
                return func(*args, **kwargs)
        try:
            is_added, fuzzed_input = fuzz_layer(args[index])
            # print(func_id + " go out FuzzLayer!")
            if not is_added:
                return func(*args, **kwargs)
            if index == 0:
                args2 = args[1:]
                if not args2:
                    result_fuzzed = func(fuzzed_input, **kwargs)
                else:
                    # print("type", type(fuzzed_input), type(args[index]))
                    result_fuzzed = func(fuzzed_input, *args2, **kwargs)
            else:
                args2_front = args[:index]
                args2_rear = args[index + 1:]
                if not args2_rear:
                    result_fuzzed = func(*args2_front, fuzzed_input, **kwargs)
                else:
                    result_fuzzed = func(*args2_front, fuzzed_input, *args2_rear, **kwargs)
            result_original = func(*args, **kwargs)
            func_result = FuncResult(result_original, result_fuzzed)
            handler = CheckHandler()
        except Exception as s:
            print(s)
            return func(*args, **kwargs)
        # handler = FixHandler()
        try:
            e = set_e(result_original)
        except:
            e = 1e-3
        result = handler.handle(args, kwargs, func_id, func_result, ptdir, e)
        return result
    return fuzz_wrapper
  return decorator

'''


def set_e(inputs):
    if str(inputs.dtype) == "torch.float64":
        return 1e-8
    if str(inputs.dtype) == "torch.float32":
        return 1e-4
    if str(inputs.dtype) == "torch.float16":
        return 1e-3
    if str(inputs.dtype) == "torch.bfloat16":
        return 1e-2



