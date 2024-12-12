import torch
from ..fuzz.check_layer import CheckLayer
import os

class FixHandler:
    def handle(self,args,kwargs,func_id,func_result,ptdir,e=0.0001):
        check_layer=CheckLayer(func_id=func_id)
        ratio=check_layer(func_result)
        for r in ratio:
            if r>(1+e)/(1-e) or r<(1-e)/(1+e):
                newresult = func_result.result_fuzzed.to(dtype=func_result.result_original.dtype)
                return newresult
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


