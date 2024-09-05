import argparse
import random
import sys
from copy import deepcopy
from functools import partial

import torch
from transformers import logging
from Datasets import MNISTDataset, FEMNISTDataset, Sentiment140Dataset, ShakespeareDataset
from Models import SimpleCNN, CNN_FEMNIST, SimpleLSTM, LSTM_Shakespeare

from Datasets.split_dataset import split_dataset_non_iid, split_dataset_iid
from defense import median, trimmed_mean, krum, multi_krum, krum_2, krum_3
from util import MyLogger, get_current_time, setup_random_seed, save_experiment_result, \
    experiment_time
import attack
import time
from torch.multiprocessing import Barrier
import torch.multiprocessing as mp
mp.set_start_method('spawn', force=True) 

logging.set_verbosity_error()  # 关闭红色提示（模型权重不完全匹配）

def local_update_and_attack_multi(i, index, global_model, train_set_list, local_update, args, malicious_attack, malicious_index_set, model_index_list):
    # print(f"进程 {i} 开始，index：{index}")
    # 使用临时模型而不是全局模型列表，减少消耗的内存
    # global_model_temp = deepcopy(global_model)
    # 专门针对高斯攻击和冻结攻击的加速策略：不训练，直接返回恶意攻击后的全局模型参数
    if index in malicious_index_set:
        if args['malicious_attack'] == 'gaussian':  # 高斯攻击
            return (index, malicious_attack(None, global_model))
        elif args['malicious_attack'] == 'zero_gradient':   # 梯度冻结攻击
            return (index, malicious_attack(None, global_model))
    
    local_model = deepcopy(global_model)
    # 这个变量会导致子线程异常退出
    # transmission_unit += 2  # 从中心服务器下载模型以及训练后上传模型，发生了两次网络传输，为了方便这里就写在一起了
    
    # 获取第index个模型对应的训练数据集
    train_dataset = train_set_list[index]
    
    # 本地训练
    avg_loss = local_update(local_model, train_dataset, args['learning_rate'], args['batch_size'], args['epoch_num'])

    local_model.to('cpu')
    global_model.to('cpu')
    
    # 本地添加均匀噪声（系数为该层参数的标准差）
    local_params = local_model.state_dict()
    for k in local_params.keys():
        std = local_params[k].data.std()
        # noise_range = args['noise_intensity'] * std / 2
        # noise = torch.zeros(local_params[k].size()).cuda()
        # noise = torch.nn.init.uniform_(noise, -noise_range, noise_range)
        # 这里需要改进，只用一个torch函数
        noise = (torch.rand(local_params[k].size()) - 0.5) * std * args['noise_intensity']
        local_params[k] += noise

    if (i+1) % int(1 if len(model_index_list) // 3 < 1 else len(model_index_list) // 3) == 0:
        print(f"客户端 {i+1} / {len(model_index_list)} 完成本地训练，loss = {avg_loss:.4f}")

    # 恶意攻击
    if index in malicious_index_set:
        return (index, malicious_attack(local_model, global_model))
    else:
        return (index, deepcopy(local_model.state_dict()))

if __name__ == '__main__':
    start_time = get_current_time()     # 实验开始时间
    parser = argparse.ArgumentParser(prog='Vanilla Federated Learning',
                                     description='Vanilla Federated Learning Experiment')
    # 架构相关
    parser.add_argument('-client', '--client_num', type=int, required=True,
                        help='number of clients')
    parser.add_argument('-round', '--comm_round_num', type=int, required=True,
                        help='number of communication rounds')
    parser.add_argument('-activate', '--activate_proportion', type=float, required=True,
                        help='proportion of activated clients')
    parser.add_argument('-agg', '--agg_proportion', type=float, required=True,
                        help='proportion of aggregated clients(which are activated)')
    
    parser.add_argument('-seed', '--random_seed', type=int, default=0,
                        help='random seed')
    
    parser.add_argument('-interval', '--test_interval', type=int, default=1,
                        help='test global model after each interval round')
    parser.add_argument('-log', '--save_log', type=int, default=1, choices=[0, 1],
                        help='if save running log to local file(0 means no, 1 means yes)')
    parser.add_argument('-log_name', '--log_name', type=str, default=start_time,
                        help='log file name')
    parser.add_argument('-multi_process', '--multi_process', type=int, default=0,
                        help='size of multi process pool(0 means sequential, >=1 means parallel)')
    
    # 训练相关
    parser.add_argument('-model', '--model', type=str, required=True,
                        help='model')
    parser.add_argument('-dataset', '--dataset', type=str, required=True,
                        help='dataset')
    parser.add_argument('-lr', '--learning_rate', type=float, default=1e-2,
                        help='learning rate in local training')
    parser.add_argument('-bs', '--batch_size', type=int, default=100,
                        help='batch size in local training')
    parser.add_argument('-epoch', '--epoch_num', type=int, default=1,
                        help='epoch num in local training')
    
    # 数据集IID划分相关
    parser.add_argument('-iid', '--iid', type=float, default=0,
                        help='dataset split(0 means non-iid, >0 means dirichlet alpha)')
    
    # 恶意攻击相关
    parser.add_argument('-malicious', '--malicious_proportion', type=float, default=0,
                        help='proportion of malicious clients')
    parser.add_argument('-attack', '--malicious_attack', type=str, default='no_attack',
                        choices=['no_attack', 'same_value', 'zero_gradient', 'gradient_scaling', 'sign_flipping',
                                 'gaussian'],
                        help='attack strategy of malicious clients(need malicious_proportion > 0)')
    parser.add_argument('-noise_intensity', '--noise_intensity', type=float, default=1,
                        help='intensity of local noise')
    # 聚合策略相关（防御恶意攻击）
    parser.add_argument('-defense', '--defense_algorithm', type=str, default='fed_avg',
                        choices=['fed_avg', 'krum', 'multi_krum', 'median', 'trimmed_mean'],
                        help='Byzantine tolerant algorithms')
    
    args = parser.parse_args()
    args = args.__dict__
    
    # 决定是否把print内容保存到本地
    if args['save_log']:
        sys.stdout = MyLogger(sys.stdout, args['log_name'])  # 同步print内容
        sys.stderr = MyLogger(sys.stderr, args['log_name'])  # 同步error内容
    
    print(f"实验开始时间：{start_time}")
    print("*******************************")
    print("框架：原始FL")
    for hyper in args.items():
        print(f"{hyper[0]}: {hyper[1]}")
    print("*******************************")

    # FED_PROX = False  # 使用FedProx的近端项loss来训练本地模型
    # if FED_PROX:
    #     LOCAL_UPDATE_FUNCTION = MODEL_PATH.local_update_FedProx
    # else:
    #     LOCAL_UPDATE_FUNCTION = MODEL_PATH.local_update
    
    # 固定随机数种子
    setup_random_seed(args['random_seed'])
    
    # 初始化模型
    if args['model'] == 'simple_cnn':
        model = SimpleCNN.Model
        local_update = SimpleCNN.local_update
        local_evaluate = SimpleCNN.local_evaluate
    elif args['model'] == 'cnn_femnist':
        model = CNN_FEMNIST.Model
        local_update = CNN_FEMNIST.local_update
        local_evaluate = CNN_FEMNIST.local_evaluate
    elif args['model'] == 'lstm':
        word2idx = SimpleLSTM.get_word2idx()
        model = SimpleLSTM.Model
        local_update_word2idx = SimpleLSTM.local_update
        local_update = partial(local_update_word2idx, word2idx=word2idx)
        local_evaluate_word2idx = SimpleLSTM.local_evaluate
        local_evaluate = partial(local_evaluate_word2idx, word2idx=word2idx)
    elif args['model'] == 'lstm_shakespeare':
        word2idx = LSTM_Shakespeare.get_word2idx()
        model = LSTM_Shakespeare.Model
        local_update_word2idx = LSTM_Shakespeare.local_update
        local_update = partial(local_update_word2idx, word2idx=word2idx)
        local_evaluate_word2idx = LSTM_Shakespeare.local_evaluate
        local_evaluate = partial(local_evaluate_word2idx, word2idx=word2idx)
    else:
        print("model not exist!")
        exit()
    
    # 初始化数据集
    if args['dataset'] == 'mnist':
        train_set, test_set = MNISTDataset.get_dataset()
    elif args['dataset'] == 'femnist':
        train_set, test_set = FEMNISTDataset.get_dataset()
    elif args['dataset'] == 'sentiment140':
        train_set, test_set = Sentiment140Dataset.get_dataset()
    elif args['dataset'] == 'shakespeare':
        train_set, test_set = ShakespeareDataset.get_dataset()
    else:
        print("dataset not exist!")
        exit()
    
    # 划分数据集
    if args['iid'] > 0:
        train_set_list = split_dataset_non_iid(train_set, args['client_num'], args['iid'])
    else:
        train_set_list = split_dataset_iid(train_set, args['client_num'])
    del train_set
    # print(f"节点数量为{args['client_num']}，划分的数据集大小分别为{[len(train_set) for train_set in train_set_list]}")
    print(f"每个节点的数据集大小：{len(train_set_list[0])}")
    
    # 随机生成恶意节点（这里只生成恶意节点的下标列表）
    index_list = list(range(args['client_num']))
    random.shuffle(index_list)
    malicious_index_set = set(index_list[0: int(args['client_num'] * args['malicious_proportion'])])
    # print(f"恶意节点数量为{len(malicious_index_set)}")

    # 初始化恶意攻击函数
    if args['malicious_attack'] == 'sign_flipping':  # 符号取反攻击
        malicious_attack = attack.sign_flipping_attack
    elif args['malicious_attack'] == 'gaussian':  # 高斯攻击
        malicious_attack = attack.gaussian_attack
    elif args['malicious_attack'] == 'zero_gradient':   # 梯度冻结攻击
        malicious_attack = attack.zero_gradient_attack
    elif args['malicious_attack'] == 'same_value':  # 同值攻击
        malicious_attack = attack.same_value_attack
    elif args['malicious_attack'] == 'gradient_scaling':    # 梯度缩放攻击
        malicious_attack = attack.gradient_scaling_attack
    elif args['malicious_attack'] == 'no_attack':    # 无攻击
        malicious_attack = attack.no_attack
    else:
        print("malicious_attack not exist!")
        exit()
    
    global_model = model()  # 全局模型（每轮结束后测试聚合性能用，顺便存储当前轮的聚合后参数）
    
    # 用于保存实验过程的变量
    experiment_result = []  # 每轮的表现保存为一个dict，多个dict组成该list
    best_fitness = 0.0  # 历史最佳表现
    transmission_unit = 0   # 网络传输数据量（基本单位为一个模型）
    running_time = 0.0

    # synchronizer = Barrier(int(args['client_num'] * args['activate_proportion'] * args['agg_proportion']))
    
    
    for comm_round in range(args['comm_round_num']):
        print(f"\n————第 {comm_round + 1} / {args['comm_round_num']} 轮通信————")
        running_start_time = time.time()    # 当前轮开始计时
        w_dict = {}     # 仅保存当前轮中训练出的所有本地模型
        
        # 选择一定比例的客户端节点
        model_index_list = list(range(args['client_num']))
        random.shuffle(model_index_list)
        model_index_list = model_index_list[0: int(args['client_num'] * args['activate_proportion'] * args['agg_proportion'])]
        # print(f"随机选择 {len(model_index_list)} 个客户端节点 {model_index_list}")
        
        test_start = time.time()
        # 多进程完成本地操作（包括本地训练、本地添加噪声、本地恶意攻击三个步骤）
        if args['multi_process'] > 0:
            # 创建进程池
            pool = mp.Pool(processes=args['multi_process'])
            res_list = []
            for i, index in enumerate(model_index_list):
                res = pool.apply_async(local_update_and_attack_multi, (i, index, global_model, train_set_list, local_update, args, malicious_attack, malicious_index_set, model_index_list))
                res_list.append(res)
            pool.close()
            pool.join()
            # 获取多进程的运行结果
            for res in res_list:
                (index, local_w) = res.get()
                w_dict[index] = local_w
        # 单进程训练
        else:
            for i, index in enumerate(model_index_list):
                (index, local_w) = local_update_and_attack_multi(i, index, global_model, train_set_list, local_update, args, malicious_attack, malicious_index_set, model_index_list)
                w_dict[index] = local_w
        test_end = time.time()
        # print(f"w_dict: {len(w_dict)}")
        # print(f"args['multi_process']={args['multi_process']}，本地更新环节花了 {test_end - test_start} 秒")

        # print(f"\n开始聚合随机选择的 {len(model_index_list)} 个客户端节点")
        agg_parameters = None
        if args['defense_algorithm'] == 'fed_avg':
            # FedAvg算法
            # 统计每个节点的本地数据集大小和全体数据集总和
            local_datasets_size = {}
            datasets_size_sum = 0
            for index in model_index_list:
                local_dataset = train_set_list[index]
                local_datasets_size[index] = len(local_dataset)
                datasets_size_sum += len(local_dataset)
                # if index in malicious_index_set:    # fedavg的额外攻击
                #     k = 10
                #     local_datasets_size[index] += len(local_dataset) * k
                #     datasets_size_sum += len(local_dataset) * k
            
            # 计算每个节点的本地数据集所占比例
            local_datasets_ratio = {}
            for k in local_datasets_size.keys():
                local_datasets_ratio[k] = local_datasets_size[k] / float(datasets_size_sum)
            # print(local_datasets_ratio)
            
            # 以本地数据集大小为权重，计算选中节点的模型参数的平均值
            avg_parameters = {}
            for k in w_dict[model_index_list[0]].keys():
                avg_parameters[k] = torch.zeros_like(w_dict[model_index_list[0]][k])
                for index in model_index_list:
                    avg_parameters[k] += w_dict[index][k] * local_datasets_ratio[index]
                    # avg_parameters[k] += w_dict[index][k]
                # avg_parameters[k] = torch.div(avg_parameters[k], len(model_index_list))
            agg_parameters = avg_parameters
        elif args['defense_algorithm'] == 'median':
            agg_parameters = median(list(w_dict.values()))
        elif args['defense_algorithm'] == 'multi_krum':
            agg_parameters = multi_krum(list(w_dict.values()), args['malicious_proportion'])
        elif args['defense_algorithm'] == 'krum':
            agg_parameters = krum(list(w_dict.values()), args['malicious_proportion'])
            # agg_parameters = krum_2(w_dict, args['malicious_proportion'])
            # agg_parameters = krum_3(list(w_dict.values()), args['malicious_proportion'])
        elif args['defense_algorithm'] == 'trimmed_mean':
            agg_parameters = trimmed_mean(list(w_dict.values()), args['malicious_proportion'])
        global_model.load_state_dict(agg_parameters)
        
        running_end_time = time.time()  # 每轮计时结束
        running_time += running_end_time - running_start_time
        
        if (comm_round+1) % args['test_interval'] == 0:
            print("\n参数聚合完毕，正在验证聚合后的模型性能")
            fitness, average_loss = local_evaluate(global_model, test_set)
            global_model.to('cpu')
            print(f"第 {comm_round + 1} 轮通信结束，fitness = {fitness:.4f}，loss = {average_loss:.4f}")
            
            if fitness > best_fitness:
                best_fitness = fitness
                print(f'第 {comm_round + 1} 轮的模型在测试集上诞生了目前最好的表现！')
            
            experiment_result.append({
                'round': comm_round + 1,
                'fitness': fitness,
                'loss': average_loss,
                'best_fitness': best_fitness,
                'transmission_unit': transmission_unit,
                'running_time': str(int(running_time))
            })
    
    print("********* 达到设定的最大通信轮数，训练终止 *********")
    
    save_experiment_result(experiment_result, args['log_name'])
    # print("实验过程保存完毕，相关数据：")
    # for row in experiment_result:
    #     print(row)
    
    end_time = get_current_time()
    minutes, seconds = experiment_time(start_time, end_time)
    print(f"\n本次实验正式结束\n持续时间：{start_time} - {end_time}，共花费{minutes}分{seconds}秒")
    

