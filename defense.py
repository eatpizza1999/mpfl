import os
from copy import deepcopy
from typing import Dict, List
import torch
import random


def median(w_list: List[Dict]):
    # 将参数按维度堆叠
    stacked_w = {
        key: torch.stack([w[key] for w in w_list], dim=0)
        for key in w_list[0].keys()
    }
    # 计算每个维度的中位数
    median_w = {
        key: torch.median(params, dim=0).values
        for key, params in stacked_w.items()
    }
    return median_w
    
    
def trimmed_mean(w_list: List[Dict], beta: float):
    trimmed_num = int(len(w_list) * beta)
    # assert trimmed_num < (len(w_list) // 2)  # 如果等于会导致返回None
    if trimmed_num >= (len(w_list) // 2):   # 额外处理恶意比例>=50%的情况，否则会返回None
        trimmed_num = len(w_list) // 2 - 1
    if trimmed_num == 0:    # 额外处理恶意比例=0的情况，否则也会返回None
        trimmed_num = 1
    # 将参数按维度堆叠
    stacked_w = {
        key: torch.stack([w[key] for w in w_list], dim=0)
        for key in w_list[0].keys()
    }
    # 计算每个维度的截尾平均值
    median_w = {
        key: torch.mean(torch.sort(params, dim=0).values[trimmed_num:-trimmed_num], dim=0)
        for key, params in stacked_w.items()
    }
    return median_w
    
    
def krum(w_list: List[Dict], malicious_proportion: float):
    lowest_score = float('inf')
    lowest_w = None
    num_models = len(w_list)
    sum_num = int(num_models * (1 - malicious_proportion)) - 1  # 聚合数量-恶意数量-1
    assert sum_num > 0
    # 遍历每个模型
    for idx in range(num_models):
        current_state_dict = w_list[idx]
        
        distance_list = []
        
        # 遍历当前模型与其他模型之间的组合
        for other_idx in range(num_models):
            if other_idx == idx:
                continue
            
            other_state_dict = w_list[other_idx]
            
            distance = 0.0
            
            # 计算当前模型参数与其他模型参数之间的距离平方和
            for (current_key, current_param), (other_key, other_param) in zip(current_state_dict.items(),
                                                                              other_state_dict.items()):
                distance += torch.norm(current_param - other_param)
            
            distance_list.append(distance)
        # print(f"total_distance:{distance_list}")
        
        score = sum(sorted(distance_list)[:sum_num])
        # print(f"score:{score}")
        if score < lowest_score:
            lowest_score = score
            lowest_w = deepcopy(current_state_dict)
    assert lowest_w is not None
    return lowest_w
    

def krum_2(w_dict, malicious_proportion: float):
    w_list = list(w_dict.values())
    nid_list = list(w_dict.keys())
    lowest_nid = -1
    
    lowest_score = float('inf')
    lowest_w = None
    num_models = len(w_list)
    sum_num = int(num_models * (1 - malicious_proportion)) - 1  # 聚合数量-恶意数量-1
    print(f"sum_num: {sum_num}")
    assert sum_num > 0
    # 遍历每个模型
    for idx in range(num_models):
        current_state_dict = w_list[idx]
        
        distance_list = []
        
        # 遍历当前模型与其他模型之间的组合
        for other_idx in range(num_models):
            if other_idx == idx:
                continue
            
            other_state_dict = w_list[other_idx]
            
            distance = 0.0
            
            # 计算当前模型参数与其他模型参数之间的距离平方和
            for (current_key, current_param), (other_key, other_param) in zip(current_state_dict.items(),
                                                                              other_state_dict.items()):
                distance += torch.norm(current_param - other_param)
            
            distance_list.append(distance)
        # print(f"total_distance:{distance_list}")
        
        score = sum(sorted(distance_list)[:sum_num])
        print(f"节点{nid_list[idx]}的分数：{score}")
        # print(f"score:{score}")
        if score < lowest_score:
            lowest_score = score
            lowest_w = deepcopy(current_state_dict)
            lowest_nid = nid_list[idx]
    print(f"lowest_nid: {lowest_nid}")
    assert lowest_w is not None
    return lowest_w


def krum_3(w_list: List[Dict], malicious_proportion: float):
    num_models = len(w_list)
    score_list = []
    
    sum_num = int(num_models * (1 - malicious_proportion)) - 1  # 聚合数量-恶意数量-1
    # sum_num = num_models - 1
    sum_num = sum_num // 2 - 1
    assert sum_num > 0
    
    # 遍历每个模型
    for idx in range(num_models):
        current_state_dict = w_list[idx]
        
        distance_list = []
        
        # 遍历当前模型与其他模型之间的组合
        for other_idx in range(num_models):
            if other_idx == idx:
                continue
            
            other_state_dict = w_list[other_idx]
            
            distance = 0.0
            
            # 计算当前模型参数与其他模型参数之间的距离平方和
            for (current_key, current_param), (other_key, other_param) in zip(current_state_dict.items(),
                                                                              other_state_dict.items()):
                distance += torch.norm(current_param - other_param)
            
            distance_list.append(distance)
        # print(f"total_distance:{distance_list}")
        
        score = sum(sorted(distance_list)[:sum_num])
        # print(f"score:{score}")
        score_list.append(score)
        
    # print(score_list)
    # lowest_idx_list = sorted(range(len(score_list)), key=lambda k: score_list[k])[:num_models-sum_num]
    lowest_idx_list = sorted(range(len(score_list)), key=lambda k: score_list[k])[:sum_num]
    # print(num_models-sum_num)
    # print(lowest_idx_list)
    selected_index = random.sample(lowest_idx_list, 1)[0]
    return deepcopy(w_list[selected_index])
    
    
def multi_krum(w_list: List[Dict], malicious_proportion: float):
    num_models = len(w_list)
    score_list = []
    
    sum_num = int(num_models * (1 - malicious_proportion)) - 1  # 聚合数量-恶意数量-1
    assert sum_num > 0
    # sum_num = num_models - 1
    # sum_num = sum_num // 2 - 1
    
    # agg_num = num_models - int(num_models * malicious_proportion) * 2 - 1
    agg_num = sum_num
    # agg_num = 5
    # agg_num = 10
    # assert agg_num > 0
    # assert len(w_list) >= agg_num
    
    # 遍历每个模型
    for idx in range(num_models):
        current_state_dict = w_list[idx]
        
        distance_list = []
        
        # 遍历当前模型与其他模型之间的组合
        for other_idx in range(num_models):
            if other_idx == idx:
                continue
            
            other_state_dict = w_list[other_idx]
            
            distance = 0.0
            
            # 计算当前模型参数与其他模型参数之间的距离平方和
            for (current_key, current_param), (other_key, other_param) in zip(current_state_dict.items(),
                                                                              other_state_dict.items()):
                distance += torch.norm(current_param - other_param)
            
            distance_list.append(distance)
        # print(f"total_distance:{distance_list}")
        
        score = sum(sorted(distance_list)[:sum_num])
        # print(f"score:{score}")
        score_list.append(score)
        
    # print(score_list)
    lowest_idx_list = sorted(range(len(score_list)), key=lambda k: score_list[k])[:agg_num]
    # print(lowest_idx_list)
    
    avg_parameters = {}
    for k in w_list[0].keys():
        avg_parameters[k] = torch.zeros_like(w_list[0][k])
        for index in lowest_idx_list:
            # avg_parameters[k] += w_list[index][k] * local_datasets_ratio[index]
            avg_parameters[k] += w_list[index][k]
        avg_parameters[k] = torch.div(avg_parameters[k], len(lowest_idx_list))
        
    return avg_parameters
    
    
if __name__ == '__main__':
    os.chdir("..")
    import Models
    m = Models.CNN_FEMNIST.Model()
    w = m.state_dict()

    # for k in w.keys():
    #     wk = w[k]
    #     print(wk)
    w_list = []
    for i in range(5):
        temp_w = deepcopy(w)
        for k in temp_w.keys():
            temp_w[k] += i
        w_list.append(temp_w)

    # 获取所有模型的参数
    all_params = w_list
    print(w_list[1]['fc1.bias'])
    print(w_list[2]['fc1.bias'])
    print(w_list[3]['fc1.bias'])
    print(multi_krum(w_list, 2, 3)['fc1.bias'])
    # lst = [5, 10, 3, 8, 6, 1]
    # n = 3
    # sorted_indices = sorted(range(len(lst)), key=lambda k: lst[k])[:n]
    # print(sorted_indices)
    