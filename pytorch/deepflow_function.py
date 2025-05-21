import sys
import torch
from itertools import product

class DeepFlowFunction:
    
    @staticmethod
    def check_var_dims(var_list, d, except_channels=False):
        for v in var_list:
            if len(v.shape) != d+2:
                raise Exception("Wrong tensor dimensionality. \n")
                return
        for v1,v2 in product(var_list,var_list):
            for i,dim in enumerate(zip(v1.shape,v2.shape)):
                if except_channels and i==1:
                    continue
                if dim[0] != dim[1]:
                    raise Exception("Tensor shapes must match")
                    return