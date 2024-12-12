import math
import os

import torch
import torch.nn as nn
from typing import Optional, Iterable, Tuple, Any, Callable, List
from ..fuzz import utils
from torch._C import _VariableFunctions as tc



class FuncResult:
    result_original=None
    result_fuzzed=None
    def __init__(self,result_original,result_fuzzed):
        self.result_fuzzed=result_fuzzed
        self.result_original=result_original



class CheckLayer(nn.Module):
    def __init__(self,func_id,**kwargs):
        super(CheckLayer,self).__init__(**kwargs)
        self.func_id=func_id



    def forward(self,func_result):
        ans1= func_result.result_original
        ans2= func_result.result_fuzzed
        if torch.is_tensor(ans1):
            ratio= self.compute_ratio(func_result)
            return ratio
        elif isinstance(ans1, (Tuple, List)):
            for index, iter_v in enumerate(ans1):
                if torch.is_tensor(iter_v):
                    temp_result=FuncResult(ans1[index],ans2[index])
                    ratio = self.compute_ratio(temp_result)
                    break
            return ratio
        else:
            return [1,1]


    def compute_ratio(self,func_result):
        #ratio_one= self.ratio_one_calculate(func_result,func_id=self.func_id)
        #ratio_inf= self.ratio_inf_calculate(func_result,func_id=self.func_id)
        return [self.ratio_each_calculate(func_result,func_id=self.func_id)]



    def _get_abs_tol(self, inputs):
        if str(inputs.dtype) == "torch.float64":
            return 1e-8
        if str(inputs.dtype) == "torch.float32":
            return 1e-4
        if str(inputs.dtype) == "torch.float16":
            return 1e-3
        if str(inputs.dtype) == "torch.bfloat16":
            return 1e-2


    #R=|a-b|/|a+b|,
    def ratio_each_calculate(self,func_result,func_id):
        ans1 = func_result.result_original
        ans2 = func_result.result_fuzzed
        abs_tol = self._get_abs_tol(ans1)
        check = min(tc.max(tc.abs(ans1)).item(), tc.max(tc.abs(ans2)).item())
        if math.isclose(check, 0, abs_tol=abs_tol):
            return 0.0
        ratio_tensor = torch.where(tc.abs(tc.min(ans1,ans2)) > abs_tol, tc.abs(ans2-ans1) / tc.abs(ans2+ans1), 0.0)
        #ratio_tensor1 = torch.where(tc.abs(ans1) > abs_tol, ans2 / ans1, 1)
        #ratio_tensor2 = torch.where(tc.abs(ans2) > abs_tol, ans1 / ans2, 1)
        #norm1 = tc.min(ratio_tensor1).item()
        #norm2 = tc.min(ratio_tensor2).item()
        ratio = tc.max(ratio_tensor).item()
        return -ratio


    def ratio_one_calculate(self, func_result,func_id):
        ans1 = func_result.result_original
        ans2 = func_result.result_fuzzed
        abs_tol = self._get_abs_tol(ans1)
        check = min(tc.max(tc.abs(ans1)).item(),tc.max(tc.abs(ans2)).item())
        if math.isclose(check,0,abs_tol=abs_tol):
            return 1.0
        norm1=torch.norm(ans1,1).item()
        norm2=torch.norm(ans2,1).item()
        ratio=norm1/norm2
        return ratio


    def ratio_inf_calculate(self, func_result,func_id):
        ans1 = func_result.result_original
        ans2 = func_result.result_fuzzed
        abs_tol = self._get_abs_tol(ans1)
        check = min(tc.max(tc.abs(ans1)).item(),tc.max(tc.abs(ans2)).item())
        if math.isclose(check,0,abs_tol=abs_tol):
            return 1.0
        norm1=tc.max(tc.abs(ans1)).item()
        norm2=tc.max(tc.abs(ans2)).item()
        ratio=norm1/norm2
        return ratio



class CheckHandler:
    def handle(self,args,kwargs,func_id,func_result,ptdir,e=0.0001):
        check_layer=CheckLayer(func_id=func_id)
        ratio=check_layer(func_result)
        for r in ratio:
            if r>(1+e)/(1-e) or r<(1-e)/(1+e):
                self.save_result_original(func_id,func_result.result_original,r,ptdir)
                self.save_result_fuzzed(func_id,func_result.result_fuzzed,r,ptdir)
                self.save_args(func_id,args,r,ptdir)
                self.save_kwargs(func_id,kwargs,r,ptdir)
        return func_result.result_original


    def save_result_original(self,func_id,result_original,r,ptdir):
        if self._check_dir_files_number(ptdir):
            path=ptdir+"/result_original_"+str(r)+func_id+".pt"
            torch.save(result_original,path)



    def save_result_fuzzed(self,func_id,result_fuzzed,r,ptdir):
        if self._check_dir_files_number(ptdir):
            path = ptdir + "/result_fuzzed_" + str(r) + func_id + ".pt"
            torch.save(result_fuzzed, path)

    def save_args(self, func_id,args,r,ptdir):
        if self._check_dir_files_number(ptdir):
            for i ,v in enumerate(args):
                path = ptdir + "/agrs_" +str(i)+"_"+ str(r) + func_id + ".pt"
                torch.save(v, path)


    def save_kwargs(self, func_id,kwargs,r,ptdir):
        if self._check_dir_files_number(ptdir):
            path = ptdir + "/kwargs_" + str(r) + func_id + ".pt"
            torch.save(kwargs, path)



    def _check_dir_files_number(self,ptdir):
        count=0
        for root,dirs,files in os.walk(ptdir):
            count+=len(files)
        if count>1000:
            return False
        return True


