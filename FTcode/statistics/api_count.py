import os
import stat

#import torch
import json

class APIcount:
    def __init__(self):
        self.checked_api_dic={}




    def add_api_into_dic(self,fun_id):
        self.checked_api_dic[fun_id]=[1,0,1]


    def update_ratio(self,fun_id,ratio):
        if ratio<self.checked_api_dic[fun_id][2]:
            self.checked_api_dic[fun_id][2]=ratio

    def update_count(self,fun_id):
        self.checked_api_dic[fun_id][0] += 1


    def update_saved(self,fun_id):
        self.checked_api_dic[fun_id][1] += 1



    def save_dic(self,dicpath):
        with os.fdopen(os.open(dicpath,os.O_RDWR|os.O_CREAT,stat.S_IRUSR|stat.S_IWUSR),"a") as file:
            json.dump(self.checked_api_dic,file, indent=2)



check_api=APIcount()