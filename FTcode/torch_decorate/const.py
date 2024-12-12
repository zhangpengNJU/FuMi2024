import os
import torch
from pathlib import Path
from typing import  Any
from pydantic import BaseSettings


class Config(BaseSettings):
    env_file = Path(__file__).parent.joinpath("config.env")
    validate_assignment =True

    ROOT_DIR: Path
    MODEL_NAME: str
    PERT_TYPE: str
    LAYER_TYPE: str

    TORCH_FLOAT_PREFIX='torch.'
    DEVICE ='device'
    META = 'meta'

    def __init__(self,**values):
        super().__init__(**values)
        if os.getenv("ROOT_DIR"):
            self.ROOT_DIR = Path(os.getenv("ROOT_DIR"))
        if os.getenv("PERT_TYPE"):
            self.ROOT_DIR = Path(os.getenv("PERT_TYPE"))
        if os.getenv("LAYER_TYPE"):
            self.ROOT_DIR = Path(os.getenv("LAYER_TYPE"))




fuzz_config=Config()