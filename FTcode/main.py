import os
from torch_decorate.torch_fuzz import PluggingTool

import click
from enum import Enum

def torch_plugging():
    tool=PluggingTool()
    tool.torch_plugging()


if __name__ == "__main__":
    torch_plugging()





