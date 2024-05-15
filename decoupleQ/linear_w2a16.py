"""
Copyright (2024) Bytedance Ltd. and/or its affiliates
"""
import torch
from torch import nn 

from .decoupleQ_kernels import dQ_preprocess_weights_int2_for_weight_only, dQ_asymm_qw2_gemm


cnt = 0

class LinearW2A16(nn.Module):
    def __init__(self, in_features:int, out_features:int, bias:bool, group_size: int):
        super().__init__()
        self.k = in_features
        self.n = out_features
        self.with_bias = bias
        self.weight = None
        self.bias = torch.zeros((out_features), dtype=torch.float16).cuda()
        self.scale = None
        self.zp = None
        self.group_size = group_size
        self.weight_processed = False        

    def forward(self, input: torch.Tensor):
        if not self.weight_processed:
            assert self.weight != None, "LinearW2A16.forward: need assign weight first"
            if self.with_bias:
                assert self.bias != None, "LinearW2A16.forward: need assign bias if use_bias"

            self.weight = dQ_preprocess_weights_int2_for_weight_only(self.weight.cpu().contiguous())
            self.weight = self.weight.cuda()
            assert self.scale != None, "LinearW2A16.forward: need scale"
            self.weight_processed = True

        output = dQ_asymm_qw2_gemm(input, self.weight, self.scale, self.zp, self.bias, self.group_size)

        return output

class LinearA16(nn.Module):
    def __init__(self, in_features:int, out_features:int, bias:bool, group_size: int):
        super().__init__()
        self.k = in_features
        self.n = out_features
        self.with_bias = bias
        self.weight = None
        self.bias = torch.zeros((out_features), dtype=torch.bfloat16).cuda()
        self.scale = None
        self.zp = None
        self.group_size = group_size
        self.weight_processed = False        

    def forward(self, input: torch.Tensor):
        if not self.weight_processed:
            self.weight = self.weight.cuda()
            self.weight_processed = True
            
        output = torch.matmul(input, self.weight)
        return output


    
