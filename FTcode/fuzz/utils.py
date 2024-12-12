

from typing import List, Set, Tuple
import torch
'''
按float32 1e-8
float64 1e-16
float16 1e-6
bfloat16:1e-4
增加扰动，
如果tensor中最大值小于p^0.5，则视为0，不增加扰动
'''
def is_add_noise(inputs):
    if not torch.is_floating_point(inputs):
        return False
    abs_tol = inputs.dtype



class CostomIterable:
    def __init__(self, iterable, func_id):
        self.iterable = iterable
        self.func_id = func_id
        self.container =self._create()

    def append(self, v):
        if self.container is None:
            return
        if isinstance(self.container, List):
           self.container.append(v)
        if isinstance(self.container,Set):
            self.container.add(v)
        else:
            return

    def get_container(self):
        if isinstance(self.iterable, Tuple):
            return tuple(self.container)
        return self.container


    def _create(self):
        if isinstance(self.iterable, (List,Tuple)):
            return list()
        else:
            raise Exception()