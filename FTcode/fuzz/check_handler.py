import torch
from ..fuzz.check_layer import CheckLayer
import os
from ..statistics.api_count import check_api


#存储结果的字典，建为函数名，值为check的次数以及不合格的次数以及最小的ratio
import datetime

dicpath="/home/users/pzhang/0528/FuMiOutput/CheckedFucDic.json"
timestamp = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
dicpath = dicpath.replace('CheckedFucDic', f'CheckedFucDic_{timestamp}')
import pickle

def readpkl(dicpath):
    a=torch.load(dicpath)
    #with open(dicpath, 'rb') as f:
	#        a = pickle.load(f)
    return a

def writepkl(dicpath, a):
    #timestamp = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
    #new_dicpath = dicpath.replace('CheckedFucDic', f'CheckedFucDic_{timestamp}')
    torch.save(a, dicpath)
    print(f'Saved to {dicpath}')


#def writepkl(dicpath,a):
#    torch.save(a,dicpath)
    #with open(dicpath, 'wb') as f:
	#    pickle.dump(a, f)


#if not os.path.isfile(dicpath):
#    writepkl(dicpath, {})



class CheckHandler:
    def handle(self,args,kwargs,func_id,func_result,ptdir,e=0.001):
        check_layer=CheckLayer(func_id=func_id)
        ratio=check_layer(func_result)
        if func_id not in check_api.checked_api_dic:
            check_api.add_api_into_dic(func_id)
        else:
            check_api.update_count(func_id)
        check_api.update_ratio(func_id,ratio[0])
        '''
        CheckedFucDic = readpkl(dicpath)
        if func_id in CheckedFucDic:
            if ratio[0]<CheckedFucDic[func_id][2]:
                CheckedFucDic[func_id][2]=ratio[0]
                CheckedFucDic[func_id][0]+=1
                if ratio[0] > (1 + e) / (1 - e) or ratio[0] < (1 - e) / (1 + e):
                    self.save_result_original(func_id, func_result.result_original, ratio[0], ptdir)
                    self.save_result_fuzzed(func_id, func_result.result_fuzzed, ratio[0], ptdir)
                    self.save_args(func_id, args, ratio[0], ptdir)
                    self.save_kwargs(func_id, kwargs, ratio[0], ptdir)
        else:
            CheckedFucDic[func_id]=[1,0,1]
            if ratio[0]<CheckedFucDic[func_id][2]:
                CheckedFucDic[func_id][2]=ratio[0]
        print(CheckedFucDic)
        '''
        #print(func_id+" Check and get ratio:")
        #print(ratio)
        for r in ratio:
            if -r>e:
                #CheckedFucDic[func_id][1] += 1
                print(func_id+" Check and get ratio:")
                print(ratio)
                check_api.update_saved(func_id)
                #print(ptdir)
                #self.save_result_original(func_id,func_result.result_original,r,ptdir)
                #self.save_result_fuzzed(func_id,func_result.result_fuzzed,r,ptdir)
                #self.save_args(func_id,args,r,ptdir)
                #self.save_kwargs(func_id,kwargs,r,ptdir)
        #print(check_api.checked_api_dic)
        check_api.save_dic(dicpath)
        #writepkl(dicpath, CheckedFucDic)
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


