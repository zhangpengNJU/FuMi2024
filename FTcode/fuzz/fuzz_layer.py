import torch
import torch.nn as nn
from typing import Optional, Iterable, Tuple, Any, Callable, List
from ..fuzz import utils


class FuzzLayer(nn.Module):
    '''

    add_noise
    '''

    def __init__(self, func_id: str, **kwargs):
        super(FuzzLayer, self).__init__(**kwargs)
        self.func_id = func_id
        self.perturbation_value: Optional[float] = None
        self.head = 0
        self.tail = -1
        self.input_type = None

    def forward(self, inputs) -> Tuple[bool, Any]:
        '''

        :param inputs: 待扰动输入
        :return: 元组【是否被扰动，扰动后输入】
        '''
        if torch.is_tensor(inputs):
            # only float
            if not torch.is_floating_point(inputs):
                return False, inputs
            self._set_perturbation_value(inputs)
            noise = self._get_noise(inputs)
            # only non-zero
            result = torch.where(
                inputs > self.perturbation_value ** 0.5, noise + inputs, inputs)
            if inputs.dtype == torch.float16:
                result = result.half()
            return True, result
        elif isinstance(inputs, (Tuple, List)):
            return self.deal_with_iterable(inputs)
        else:
            return False, inputs

    def _get_noise(self, inputs):
        dtype = inputs.dtype
        if hasattr(inputs, 'device'):
            device = inputs.device
            if device == "meta":
                noise = torch.ones(inputs.shape, device="meta", dtype=dtype) * self.perturbation_value
            else:
                noise = torch.ones(inputs.shape, dtype=dtype).to(device) * self.perturbation_value
        else:
            noise = torch.ones(inputs.shape, dtype=dtype) * self.perturbation_value
        return noise

    def _set_perturbation_value(self, inputs):
        if str(inputs.dtype) == "torch.float64":
            self.perturbation_value = 1e-16
        if str(inputs.dtype) == "torch.float32":
            self.perturbation_value = 1e-8
        if str(inputs.dtype) == "torch.float16":
            self.perturbation_value = 1e-6
        if str(inputs.dtype) == "torch.bfloat16":
            self.perturbation_value = 1e-4

    def deal_with_iterable(self, inputs):
        costom_iterable = utils.CostomIterable(iterable=inputs, func_id=self.func_id)
        if costom_iterable.container is None:
            return False, inputs
        disturbances_conunt = 0
        is_added = False
        for _, v in enumerate(inputs):
            if disturbances_conunt == 0 and torch.is_tensor(v):
                if not torch.is_floating_point(v):
                    costom_iterable.append(v)
                    continue
                self._set_perturbation_value(v)
                noise = self._get_noise(v)
                # 只有对于非0位置，才加扰动
                result = torch.where(v > self.perturbation_value ** 0.5, noise + v, v)
                if v.dtype == torch.float16:
                    result = result.half()
                costom_iterable.append(result)
                disturbances_conunt += 1
                is_added = True
            else:
                costom_iterable.append(v)
        return is_added, costom_iterable.get_container()



class FuzzLayer2(FuzzLayer):
    '''

    扰动层
    '''

    def __init__(self, func_id: str, **kwargs):
        super(FuzzLayer, self).__init__(**kwargs)
        self.func_id = func_id
        self.perturbation_value: Optional[float] = None
        self.head = 0
        self.tail = -1
        self.input_type = None


    def forward(self, inputs) -> Tuple[bool, Any]:
        '''

        :param inputs: 待扰动输入
        :return: 元组【是否被扰动，扰动后输入】
        '''
        fuzzed_inputs=torch.clone(inputs)
        temp_first=torch.clone(fuzzed_inputs[self.head])
        temp_tail=torch.clone(fuzzed_inputs[self.tail])
        if isinstance(inputs,tuple):
            fuzzed_inputs_temp=list(fuzzed_inputs)
            fuzzed_inputs_temp[self.head]= temp_tail
            fuzzed_inputs_temp[self.tail]=temp_first
            return True, tuple(fuzzed_inputs_temp)
        elif isinstance(inputs, torch.Tensor):
            fuzzed_inputs[self.head]=temp_tail
            fuzzed_inputs[self.tail]=temp_first
            return True, fuzzed_inputs
        else:
            return False,inputs




class FuzzLayer3(FuzzLayer):
    '''

    扰动层
    '''

    def __init__(self, func_id: str, **kwargs):
        super(FuzzLayer, self).__init__(**kwargs)
        self.func_id = func_id
        self.perturbation_value: Optional[float] = None
        self.head = 0
        self.tail = -1
        self.input_type = None

    def forward(self, inputs) -> Tuple[bool, Any]:
        '''

        :param inputs: 待扰动输入
        :return: 元组【是否被扰动，扰动后输入】
        '''
        if torch.is_tensor(inputs):
            # 只扰动浮点数
            if not torch.is_floating_point(inputs):
                return False, inputs
            if inputs.dtype==torch.float64:
                return False, inputs
            self._set_perturbation_value(inputs)
            #精度变化：非float64精度则向上提升，float64不fuzz(float64->float32?)
            fuzzed_inputs = self._change_dtype(inputs)
            #非0位置变化
            #result = torch.where(inputs > self.perturbation_value ** 0.5, fuzzed_inputs, inputs)
            #if inputs.dtype == torch.float16:
            #    result = result.half()
            return True, fuzzed_inputs
        elif isinstance(inputs, (Tuple, List)):
            return False, inputs
        else:
            return False, inputs

    def _change_dtype(self, inputs):
        if hasattr(inputs, 'device'):
            device = inputs.device
            if device == "meta":
                fi = inputs.to(device="meta", dtype=self.perturbation_value)
            else:
                fi = inputs.to( dtype=self.perturbation_value).to(device)
        else:
            fi = inputs.to(dtype=self.perturbation_value)
        return fi


    def _set_perturbation_value(self, inputs):
        if str(inputs.dtype) == "torch.float32":
            self.perturbation_value = torch.float64
        if str(inputs.dtype) == "torch.float16":
            self.perturbation_value = torch.float32
        if str(inputs.dtype) == "torch.bfloat16":
            self.perturbation_value = torch.float32

'''
    def deal_with_iterable(self, inputs):
        costom_iterable = utils.CostomIterable(iterable=inputs, func_id=self.func_id)
        if costom_iterable.container is None:
            return False, inputs
        disturbances_conunt = 0
        is_added = False
        for _, v in enumerate(inputs):
            if disturbances_conunt == 0 and torch.is_tensor(v):
                if not torch.is_floating_point(v):
                    costom_iterable.append(v)
                    continue
                self._set_perturbation_value(v)
                noise = self._get_noise(v)
                # 只有对于非0位置，才加扰动
                result = torch.where(v > self.perturbation_value ** 0.5, noise + v, v)
                if v.dtype == torch.float16:
                    result = result.half()
                costom_iterable.append(result)
                disturbances_conunt += 1
                is_added = True
            else:
                costom_iterable.append(v)
        return is_added, costom_iterable.get_container()
'''




class FuzzLayer4(nn.Module):
    '''

    将函数换为一个自定义函数。#其实不需要这个东西，在fuzz_decorate里实现即可，故该类实际上没有实现，也没有使用
    '''

    def __init__(self, func_id: str, **kwargs):
        super(FuzzLayer, self).__init__(**kwargs)
        self.func_id = func_id
        self.perturbation_value: Optional[float] = None
        self.head = 0
        self.tail = -1
        self.input_type = None

    def forward(self, inputs) -> Tuple[bool, Any]:
        '''

        :param inputs: 待扰动输入
        :return: 元组【是否被扰动，扰动后输入】
        '''
        if torch.is_tensor(inputs):
            # only float
            if not torch.is_floating_point(inputs):
                return False, inputs
            self._set_perturbation_value(inputs)
            noise = self._get_noise(inputs)
            # only non-zero
            result = torch.where(
                inputs > self.perturbation_value ** 0.5, noise + inputs, inputs)
            if inputs.dtype == torch.float16:
                result = result.half()
            return True, result
        elif isinstance(inputs, (Tuple, List)):
            return self.deal_with_iterable(inputs)
        else:
            return False, inputs




