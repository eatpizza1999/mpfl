from copy import deepcopy

from torch.nn import Module
import torch
from random import random


def no_attack(local_model: Module, global_model: Module):
    return deepcopy(local_model.state_dict())

def sign_flipping_attack(local_model: Module, global_model: Module):
    # local_model.cuda()
    # global_model.cuda()
    local_params = deepcopy(local_model.state_dict())
    global_params = deepcopy(global_model.state_dict())
    for k in local_params.keys():
        delta_param = global_params[k] - local_params[k]
        local_params[k] = global_params[k] + delta_param
    return local_params
    
    
def gaussian_attack(local_model: Module, global_model: Module):
    # local_model.cuda()
    # global_model.cuda()
    # local_params = deepcopy(local_model.state_dict())
    global_params = deepcopy(global_model.state_dict())
    for k in global_params.keys():
        # delta_param = global_params[k] - local_params[k]
        std = global_params[k].data.std()
        std *= 1
        global_params[k] += torch.randn(global_params[k].size()) * std
    return global_params


def gradient_scaling_attack(local_model: Module, global_model: Module):
    # local_model.cuda()
    # global_model.cuda()
    local_params = deepcopy(local_model.state_dict())
    global_params = deepcopy(global_model.state_dict())
    alpha = 2 * random() - 1    # ����(-1,1)�����ڵ������
    for k in local_params.keys():
        delta_param = global_params[k] - local_params[k]
        local_params[k] = global_params[k] - alpha * delta_param
    return local_params


def zero_gradient_attack(local_model: Module, global_model: Module):
    # global_model.cuda()
    global_params = deepcopy(global_model.state_dict())
    return global_params


def same_value_attack(local_model: Module, global_model: Module):
    # local_model.cuda()
    local_params = deepcopy(local_model.state_dict())
    for k in local_params.keys():
        local_params[k] += 0.1
    return local_params


