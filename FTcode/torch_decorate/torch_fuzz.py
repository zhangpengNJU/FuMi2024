import os
import stat
import traceback
from functools import wraps
from typing import Callable, Tuple
import sys
sys.path.append("..")
#from fuzz_decorate import decorate_functuion
from pathlib import Path
import torch
import json
from charset_normalizer import detect
import shutil

#from const import fuzz_config

class PluggingTool:
    def __init__(self, perturbation_mode:str = "", model_name= "", output_dir= "", layer_type: str = ""):
        #//FTcode
        self.project_path = Path(__file__).parent.parent

        self.torch_root_dir =self._get_torch_root_dir()
        self.fuzz_plugging_dir = self.project_path
        self.data_dir=self.torch_root_dir.joinpath("FTcode").joinpath("data")

        self.perturbation_mode=perturbation_mode
        self.to_be_replaced_perturbation_mode = "add_noise"
        self.model_name =model_name
        self.to_be_replaced_model_name= "testedmodel"
        self.to_be_replaced_root_source ="."
        self.layer_type = layer_type
        self.to_be_replaced_layer_type ="check"
        self.output_dir=output_dir

        self.torch_positioning_file = self.data_dir.joinpath("torch_positioning.json")
        self.torch_version = self._get_torch_version()
        self.torch_init_file_name= "__init__.py"
        self.flags = os.O_RDWR |os.O_TRUNC|os.O_CREAT
        self.modes=stat.S_IRUSR |stat.S_IWUSR
        self.init_append_content = "from . import FTcode"


    def torch_plugging(self):
        self._move_fuzz_plugging_dir()
        self._change_torch_init_file()



    def remove_torch_plugging(self):
        init_original_length, init_original_final_str = self._get_torch_end_line_final_str()
        with open(self.torch_root_dir.joinpath(self.torch_init_file_name),"r") as file:
            init_original =file.readlines()
        new_content= init_original[:init_original_length]
        with os.fdopen(os.open(self.torch_root_dir.joinpath(self.torch_init_file_name), self.flags, self.modes),
                       "w") as file:
            file.writelines(new_content)


    def _get_torch_root_dir(self):
        torch_path_list = torch.__path__
        if torch_path_list:
            return Path(torch_path_list[0])
        else:
            raise Exception("no torch")


    def _change_torch_init_file(self):
        '''
        修改torch的init文件，补上init_append_content
        :return:
        '''
        init_original_length, init_original_final_str = self._get_torch_end_line_final_str()
        with open(self.torch_root_dir.joinpath(self.torch_init_file_name),"r") as file:
            init_original =file.readlines()
        content=['\n','\n']
        content.extend(self.init_append_content)
        if len(init_original) == init_original_length and init_original_final_str==init_original[-1]:
            flags = os.O_RDWR |os.O_CREAT
            with os.fdopen(os.open(self.torch_root_dir.joinpath(self.torch_init_file_name),flags,self.modes), "a+") as file:
                file.writelines(content)
        else:
            final_content=init_original[:init_original_length]
            final_content.extend(content)
            with os.fdopen(os.open(self.torch_root_dir.joinpath(self.torch_init_file_name),self.flags,self.modes), "w") as file:
                file.writelines(final_content)



    def _get_torch_end_line_final_str(self):
        if not self.torch_positioning_file.is_file():
            raise FileNotFoundError(self.torch_positioning_file)
        with open(self.torch_positioning_file,"r") as file:
            torch_position_dict = json.load(file)
        result=torch_position_dict.get(self.torch_version)
        if not result:
            raise Exception("no torch")
        end_line = result.get("end_line")
        final_str = result.get("final_str")
        return end_line,final_str


    def _move_fuzz_plugging_dir(self):
        torch_fuzz_plugging_dir = self.torch_root_dir.joinpath("FTcode")
        if os.path.exists(torch_fuzz_plugging_dir):
            shutil.rmtree(torch_fuzz_plugging_dir)
        shutil.copytree(self.fuzz_plugging_dir,torch_fuzz_plugging_dir)
        new_config_file = torch_fuzz_plugging_dir.joinpath("torch_decorate")
        new_config_file = new_config_file.joinpath("config.env")
        self._replace_str(file_path=new_config_file)


    def _replace_str(self,file_path):
        encoding = self._get_file_encoding(file_path=file_path.as_posix())
        with open(file_path,"r",encoding=encoding) as file:
            file_content = file.read()
        new_content=file_content.replace(self.to_be_replaced_perturbation_mode, self.perturbation_mode).replace(
            self.to_be_replaced_model_name, self.model_name).replace(
            self.to_be_replaced_root_source, self.output_dir).replace(
            self.to_be_replaced_layer_type,self.layer_type
        )
        with os.fdopen(os.open(file_path, self.flags, self.modes),
                  "w",encoding=encoding) as file:
            file.writelines(new_content)


    def _get_torch_version(self):
        version = torch.__version__
        return version


    def _get_file_encoding(self,file_path):
        with open(file_path,"rb") as file:
            msg=file.read()
        encoding = detect(msg).get("encoding")
        if encoding is None:
            raise Exception("no encoding")
        return encoding



def disable():
    tool = PluggingTool(perturbation_mode='add_noise',model_name="dd", output_dir="/mnt/tmp",layer_type="check")
    tool.remove_torch_plugging()


def enable(perturbation_mode='add_noise',model_name="dd", output_dir="/mnt/tmp",layer_type="check"):
    tool = PluggingTool(perturbation_mode=perturbation_mode,model_name=model_name, output_dir=output_dir,layer_type=layer_type)
    tool.torch_plugging()



#enable()

if __name__ == "__main__":
    disable()

