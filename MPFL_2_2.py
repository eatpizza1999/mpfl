# coding=utf-8
import argparse
import os
import random
import sys
import numpy as np
import torch
from copy import deepcopy
import attack
from Datasets import MNISTDataset, FEMNISTDataset, Sentiment140Dataset, ShakespeareDataset
from Models import SimpleCNN, CNN_FEMNIST, SimpleLSTM, LSTM_Shakespeare
from Datasets.split_dataset import split_dataset_non_iid, split_dataset_iid
from Node import Node, ROLE_POPULATION, ROLE_INDIVIDUAL
from Network import Network
from transformers import logging
from util import MyLogger, get_current_time, save_experiment_result, experiment_time
from functools import partial
import time
import torch.multiprocessing as mp
mp.set_start_method('spawn', force=True) 

logging.set_verbosity_error()  # 关闭红色提示（模型权重不完全匹配）


def setup_random_seed(seed=0):
    """
    固定与模型训练有关的所有随机数，防止随机波动

    :param seed: 随机数种子
    :return: None
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)  # 禁止hash随机化
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def selection_tournament(fitness_dict, selection_num):
    """
    二元锦标赛选择算法
    :param fitness_dict:
    :param selection_num:
    :return:
    """
    assert len(fitness_dict) >= selection_num
    
    selection_set = set()
    while len(selection_set) < selection_num:
        individual_1, individual_2 = random.sample(list(fitness_dict.items()), 2)
        if individual_1[1] >= individual_2[1]:
            selection_set.add(individual_1[0])
        else:
            selection_set.add(individual_2[0])
    return selection_set


def selection(fitness_dict, selection_num):
    """
    选择算法（轮盘赌选择）
    这个算法有点不太适配，后续可以换成锦标赛选择之类的其他算法
    这个算法有BUG，部分特殊情况会出现无限死循环

    :param fitness_dict: 适应度组成的字典，key为id，value为适应度
    :param selection_num: 选择的数量，必须满足该参数指定的数量，算法才会结束
    :return: 被选中的列表下标index组成的集合（注意是集合）
    """
    
    assert len(fitness_dict) >= selection_num
    
    # 为了防止出现适应度为0的个体永远无法选中（其所占面积为0）的BUG，添加一个微小的值
    epsilon = 1e-5
    for k in fitness_dict.keys():
        fitness_dict[k] += epsilon
    
    # 获得分别有序的id和适应度组成的列表
    id_list = list(fitness_dict.keys())
    fitness_list = []
    for i in id_list:
        fitness_list.append(fitness_dict[i])
    
    fitness_dict.values()
    total_fitness = sum(fitness_list)
    accu_fitness = 0
    accu_list = []
    
    # 计算累计概率
    for i in range(len(fitness_list)):
        accu_fitness += fitness_list[i] / total_fitness
        accu_list.append(accu_fitness)
    
    # 转轮盘抽取
    selection_set = set()
    while len(selection_set) < selection_num:
        r = random.uniform(0, 1)
        idx = 0
        while r > accu_list[idx]:
            idx += 1
        selection_set.add(id_list[idx])
    return selection_set


def selection_topk(fitness_dict, selection_num):
    """
    选择算法（TOP_k选择）

    :param fitness_dict: 适应度组成的字典，key为id，value为适应度
    :param selection_num: 选择的数量，必须满足该参数指定的数量，算法才会结束
    :return: 被选中的列表下标index组成的集合（注意是集合）
    """
    
    assert len(fitness_dict) >= selection_num
    
    # 获得分别有序的id和适应度组成的列表
    top_f_list = sorted(fitness_dict.items(), key=lambda d: d[1], reverse=True)[: selection_num]
    
    # 转轮盘抽取
    selection_set = set()
    for top_nid, _ in top_f_list:
        selection_set.add(top_nid)

    return selection_set


def FedAvg(w_dict, weight_dict):
    assert len(w_dict) == len(weight_dict)
    # 以本地数据集大小为权重，计算选中节点的模型参数的平均值
    temp_w = list(w_dict.values())[0]
    avg_parameters = {}
    for k in temp_w.keys():
        avg_parameters[k] = torch.zeros_like(temp_w[k])
        for nid in w_dict.keys():
            avg_parameters[k] += w_dict[nid][k] * weight_dict[nid]
        #     avg_parameters[k] += w_dict[nid][k]
        # avg_parameters[k] = torch.div(avg_parameters[k], len(w_dict))
    return avg_parameters


class Block:
    """
    区块链里的单个区块，目前属性不完整，但足够用于模拟
    """
    
    def __init__(self, block_id, time, content):
        self.bid = block_id  # 区块id
        self.time = time  # 创建时间
        self.content = content  # 交易内容
        # 之后再加一个前一个区块id


class BlockChain:
    def __init__(self):
        self.block_list = []
    
    def get_latest_block(self):
        return self.block_list[-1]
    
    def get_latest_model_parameters(self):
        return self.get_latest_block().content
    
    def add_latest_block(self, block):
        self.block_list.append(block)
    
    def add_block(self, content):
        new_block = Block(self.get_latest_block().bid + 1, get_current_time(), content)
        self.block_list.append(new_block)
        # 测试加速用，只保留最新的N个区块，控制区块链长度避免内存爆炸
        latest_block_num = 3
        if len(self.block_list) > latest_block_num:
            self.block_list = self.block_list[-latest_block_num:]
    
    def get_len(self):
        return len(self.block_list)


if __name__ == '__main__':
    start_time = get_current_time()     # 实验开始时间
    parser = argparse.ArgumentParser(prog='Multi-Population Federated Learning',
                                     description='Multi-Population Federated Learning Experiment')
    
    # 我的MPFL框架的额外超参数
    parser.add_argument('-population', '--population_num', type=int, required=True,
                        help='number of populations')
    # parser.add_argument('-select', '--select_num', type=int, required=True,
    #                     help='number of clients selected by each populations to aggregate')
    # parser.add_argument('-dp', '--differential_privacy', type=int, default=0, choices=[0, 1],
    #                     help='differential privacy(0 means no 1 means yes)')
    parser.add_argument('-imm_interval', '--immigration_interval', type=int, default=1,
                        help='clients immigrate after each interval round')
    parser.add_argument('-imm_num', '--immigration_num', type=int, default=1,
                        help='number of immigrated clients after each interval round')
    parser.add_argument('-ele_interval', '--election_interval', type=int, default=1,
                        help='clients elect new committee nodes after each interval round')
    # parser.add_argument('-double_agg', '--double_aggregation', type=int, default=1, choices=[0, 1],
    #                     help='if double aggregation(0 means no 1 means yes')
    parser.add_argument('-competition', '--competition_threshold', type=float, required=True,
                        help='compare ones own client and others fitness, if touch this then vote')
    # parser.add_argument('-consensus', '--consensus_threshold', type=float, required=True,
    #                     help='minimum proportion of agreed populations to reach a consensus')
    parser.add_argument('-ablation_selection', '--ablation_selection', type=int, default=0,
                        help='ablation study of selection')
    parser.add_argument('-ablation_crossover', '--ablation_crossover', type=int, default=0,
                        help='ablation study of crossover')
    
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
                        # choices=['simple_cnn', 'cnn_femnist', 'lstm'],
                        help='model')
    parser.add_argument('-dataset', '--dataset', type=str, required=True,
                        # choices=['mnist', 'femnist', 'sentiment140'],
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
    args = parser.parse_args()
    args = args.__dict__
    
    # 决定是否把print内容保存到本地
    if args['save_log']:
        sys.stdout = MyLogger(sys.stdout, args['log_name'])  # 同步print内容
        sys.stderr = MyLogger(sys.stderr, args['log_name'])  # 同步error内容
    
    print(f"实验开始时间：{start_time}")
    print("*******************************")
    print("框架：二次聚合MPFL（改进版）")
    for hyper in args.items():
        print(f"{hyper[0]}: {hyper[1]}")
    print("*******************************")
    
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
    # 接下来使用的模型（具体使用时仅加载参数）
    global_model = model()
    local_model = deepcopy(global_model)
    
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
        node_train_set_list = split_dataset_non_iid(train_set, args['client_num'], args['iid'])
    else:
        node_train_set_list = split_dataset_iid(train_set, args['client_num'])
    del train_set
    # print(f"每个节点的数据集大小分别为{[len(train_set) for train_set in node_train_set_list]}")
    print(f"每个种群内的节点数：{args['client_num'] // args['population_num']}")
    print(f"每个节点的数据集大小：{len(node_train_set_list[0])}")
    
    # 随机分配节点身份和所属种群
    node_num_per_population = args['client_num'] // args['population_num']  # 每个种群的节点数量（包括个体节点和种群节点）
    role_list = []
    belong_list = []
    for i in range(args['population_num']):
        belong_list += [f'P{i}'] * node_num_per_population
        role_list += [ROLE_POPULATION] + [ROLE_INDIVIDUAL] * (node_num_per_population - 1)
    # role_list = [ROLE_INDIVIDUAL] * args['client_num']  # 开局全员个体
    # print(role_list)
    # print(belong_list)
    
    # 初始化区块链（创世区块里塞个空白模型）
    # 注意，本来区块链应该作为节点的属性，每个节点都保存一条区块链，但是这么做在模拟的时候消耗太大，我这里暂时将它独立出来，也就是所有节点共用一条区块链
    # 已修改，目前有种群数量+1条区块链
    # print("正在初始化区块链……")
    global_params = global_model.state_dict()
    # 全局精英链
    genesis_block = Block(
        block_id=0,
        time=get_current_time(),
        content=deepcopy(global_params)
    )
    global_block_chain = BlockChain()
    global_block_chain.add_latest_block(genesis_block)
    # # 种内精英链（为了节省空间，暂时只使用全局区块链）
    # local_block_chain_list = []
    # for i in range(args['population_num']):
    #     genesis_block = Block(
    #         block_id=0,
    #         time=get_current_time(),
    #         content=deepcopy(global_params)
    #     )
    #     local_block_chain = BlockChain()
    #     local_block_chain.add_latest_block(genesis_block)
    #     local_block_chain_list.append(local_block_chain)
    
    # 初始化节点网络
    # print("正在初始化节点网络……")
    node_index_list = []    # 节点id列表
    for i in range(args['client_num']):
        node_index_list.append('N' + str(i))
    node_network = Network()
    for i in range(args['client_num']):
        # 创建一个新的节点
        new_node = Node(
            node_id=node_index_list[i],
            role=role_list[i],
            belong=belong_list[i],
            train_set=node_train_set_list[i],
            # test_set=test_set,  # 每个节点都使用完整的测试集
            test_set=node_train_set_list[i],
            # model=None,
            # local_update_function=local_update,
            # local_evaluate_function=local_evaluate,
            # local_block_chain=local_block_chain_list[i // node_num_per_population], # 为了节省空间，暂时只使用全局区块链
            local_block_chain=None,
            global_block_chain=global_block_chain,
        )
        # 将该节点加入到节点网络
        node_network.add_node(new_node)
    # print("当前节点网络：" + node_network.print_status())
    
    # 随机生成恶意节点（这里只生成恶意节点的下标列表）
    random.shuffle(node_index_list)
    malicious_index_set = set(node_index_list[0: int(args['client_num'] * args['malicious_proportion'])])

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
    
    """ 训练正式开始 """
    experiment_result = []
    best_fitness = 0.0
    transmission_unit = 0
    running_time = 0.0
    
    for current_conn_num in range(args['comm_round_num']):
        print(f"\n————第 {current_conn_num + 1} / {args['comm_round_num']} 轮通信————")
        running_start_time = time.time()    # 每轮计时开始
        """相比于原版框架新添加的加速部分：每轮通信开始时先激活节点，接下来只对激活节点进行操作"""
        # 随机激活指定比例/数量的节点（必须均匀分布在每个种群中）
        active_dict = {}
        sample_num = int(args['client_num'] * args['activate_proportion']) // args['population_num']  # 每个种群出几个节点
        population_dict = node_network.get_populations_expect_committee()
        # print("各种群大小：", {b: len(n) for b, n in population_dict.items()})
        for belong, node_list in population_dict.items():
            active_dict[belong] = random.sample(node_list, sample_num)
        
        # 遍历每个种群
        population_fitness_dict = {}
        population_params_dict = {}
        exchange_pool = {}
        for population_index, (belong, node_list) in enumerate(active_dict.items()):
            print(f"种群 {belong} ({population_index + 1} / {len(active_dict)})")
            w_dict = {}  # 临时保存当前种群内的所有模型参数,key为nid，value为对应参数
            # 首先每个种群内的节点进行本地训练
            for node_index, node in enumerate(node_list):
                # Step 1: 初始化种群（个体节点在本地训练模型）
                # 当前节点加载种内精英链上的最新模型参数（为了节省内存，目前只使用一条全局区块链）
                # latest_model_parameters = node.local_block_chain.get_latest_model_parameters()
                # 专门针对高斯攻击和冻结攻击的加速策略：不训练，直接返回恶意攻击后的全局模型参数
                if node.nid in malicious_index_set:
                    if args['malicious_attack'] == 'gaussian':  # 高斯攻击
                        w_dict[node.nid] = malicious_attack(None, global_model)
                        continue
                    elif args['malicious_attack'] == 'zero_gradient':   # 梯度冻结攻击
                        w_dict[node.nid] = malicious_attack(None, global_model)
                        continue
                latest_model_parameters = node.global_block_chain.get_latest_model_parameters()
                local_model.load_state_dict(latest_model_parameters)
                # transmission_unit += 1  # 从区块链下载模型，发生了网络传输
                # 当前节点使用本地训练集训练模型
                loss = local_update(local_model, node.train_set, args['learning_rate'], args['batch_size'],
                                    args['epoch_num'])
                
                local_model.to('cpu')
                
                # Step 2：变异（个体节点对训练好的模型参数添加均匀噪声）
                local_params = local_model.state_dict()
                for k in local_params.keys():
                    std = local_params[k].data.std()
                    noise = (torch.rand(local_params[k].size()) - 0.5) * std * args['noise_intensity']
                    local_params[k] += noise
                
                # 恶意节点行动
                if node.nid in malicious_index_set:
                    # 为了节省空间，暂时只使用一条全局区块链（这里不使用也行）
                    # global_model.load_state_dict(node.local_block_chain.get_latest_model_parameters())
                    # global_model.load_state_dict(node.global_block_chain.get_latest_model_parameters())
                    w_dict[node.nid] = malicious_attack(local_model, global_model)
                else:
                    w_dict[node.nid] = deepcopy(local_model.state_dict())
                
                # if (node_index + 1) % (1 if len(node_list) // 3 == 0 else len(node_list) // 3) == 0:
                #     print(f"    节点 {node.nid} ({node_index + 1} / {len(node_list)}) 完成训练, loss = {loss:.4f}")
                if node_index == 0:
                    print(f"    节点完成训练，loss = {loss:.4f}")
            
            # # 如果达到选举间隔，重新选举种群节点
            # """ 该步骤暂时省略，直接采用开局随机分配的身份 """
            # """后续修改成使用上一轮表现最好的个体节点"""
            #
            # population_node = random.sample(node_list, 1)[0]
            # population_node.role = ROLE_POPULATION
            population_node = node_network.get_population_node_by_belong(belong)
            
            # 种群节点计算该种群内所有个体的适应度，并选择一定数量的个体进行参数聚合
            # Step 3：计算个体适应度（种群节点验证个体节点训练好的模型）
            fitness_dict = {}  # 保存个体适应度的字典，key为个体节点的id，value为适应度/模型性能
            if not args['ablation_selection']:
                # 再遍历每个种群节点拥有的个体节点
                # print(f"\n种群节点{population_node.nid}正在验证个体节点的模型性能")
                for nid, w in w_dict.items():
                    # 获取个体节点的模型，计算个体节点的适应度（验证模型性能）
                    local_model.load_state_dict(w)
                    fitness, _ = local_evaluate(local_model, population_node.test_set)
                    fitness_dict[nid] = fitness
                    transmission_unit += 1  # 种群节点从个体节点接收模型，发生了网络传输
            else:
                for nid, w in w_dict.items():
                    fitness_dict[nid] = 1
                    transmission_unit += 1  # 种群节点从个体节点接收模型，发生了网络传输
            # print(f"种群节点{population_node.nid}的个体适应度列表：{fitness_dict}")
            population_fitness_dict[belong] = fitness_dict
            
            # 将表现最差的节点添加进交换池
            sorted_fitness_dict = sorted(fitness_dict.items(), key=lambda d: d[1], reverse=False)
            # if sorted_fitness_dict[0][0] == population_node.nid:    # 防止种群节点把自己加进交换池导致BUG
            #     worst_nid = sorted_fitness_dict[1][0]
            # else:
            #     worst_nid = sorted_fitness_dict[0][0]
            worst_nid = sorted_fitness_dict[0][0]
            exchange_pool[worst_nid] = w_dict[worst_nid]
            
            # Step 4：选择（种群节点选择指定比例的优秀个体节点）
            selected_nodes_id_set = selection(fitness_dict, int(sample_num * args['agg_proportion']))
            # selected_nodes_id_set = selection_topk(fitness_dict, int(sample_num * args['agg_proportion']))
            
            # print(f"种群节点{population_node.nid}聚合选中的{len(selected_nodes_id_set)}个体节点")
            selected_w_dict = {}
            for nid in selected_nodes_id_set:
                selected_w_dict[nid] = w_dict[nid]
            
            # Step 5: 交叉（种群节点聚合被选中的个体节点的模型参数）
            ''' 改版FedAvg实现 '''
            # 计算每个节点的本地数据集所占比例
            local_datasets_size = {}
            datasets_size_sum = 0
            for nid in selected_w_dict.keys():
                local_dataset = node_network.get_node_by_id(nid).train_set
                local_datasets_size[nid] = len(local_dataset)
                datasets_size_sum += len(local_dataset)
            local_datasets_ratio = {}
            for k in local_datasets_size.keys():
                local_datasets_ratio[k] = local_datasets_size[k] / float(datasets_size_sum)
            
            agg_parameters = FedAvg(selected_w_dict, local_datasets_ratio)
            
            # # 恶意节点行动（在模型参数聚合后再度恶意攻击）
            # malicious_committee = False
            # if malicious_committee:
            #     if population_node.nid in malicious_index_set:
            #         global_model.load_state_dict(population_node.local_block_chain.get_latest_model_parameters())
            #         local_model.to('cpu')
            #         local_model.load_state_dict(agg_parameters)
            #         agg_parameters = malicious_attack(local_model, global_model)
            
            # 种群节点将聚合后的模型参数保存起来
            population_params_dict[belong] = agg_parameters
            # print(f"种群节点{population_node.nid}的参数聚合完毕")
        
        # Step 6: 人工选择/精英保留
        if not args['ablation_crossover']:
            """改造：通过委员会共识算法达成共识，进化节点（可选操作：选择其他种群的种内精英个体，再进行一次 FedAvg）将精英个体上传至精英链"""
            # print("\n开始选举进化节点")
            eval_dict = {}
            # 每个种群节点互相验证聚合后的模型
            # print("\n种群节点互相打分……")
            for p_node in node_network.get_nodes_by_role(ROLE_POPULATION):
                for belong, w in population_params_dict.items():
                    local_model.load_state_dict(w)
                    curr_fitness, _ = local_evaluate(local_model, p_node.test_set)
                    transmission_unit += 1  # 种群节点互相传输模型，发生了网络传输
                    if belong not in eval_dict.keys():
                        eval_dict[belong] = {}
                    eval_dict[belong][p_node.belong] = curr_fitness
            # print(eval_dict)
            # 计算每个种群节点的总分，选择总分最高的节点作为优胜节点
            score_dict = {}
            for belong in eval_dict.keys():
                score_dict[belong] = 0
                for p_belong in eval_dict[belong].keys():
                    score_dict[belong] += eval_dict[belong][p_belong]
            # print(f"总分：{score_dict}")
            score_list = sorted(score_dict.items(), key=lambda d: d[1], reverse=True)
            max_belong, max_score = score_list[0]
            inter_best_node = node_network.get_population_node_by_belong(max_belong)
            # print(f"得分最高的种群为{inter_best_node.belong}")
            
            # 优胜节点聚合指定数量的种群节点（优先聚合总分更高的）
            second_aggregated_params_dict = {}
            for belong, score in score_list:
                if score >= max_score * args['competition_threshold']:
                    second_aggregated_params_dict[node_network.get_population_node_by_belong(belong).nid] = population_params_dict[belong]
            print(f"优胜节点选择聚合{len(second_aggregated_params_dict)}个种群")
            
            local_datasets_size = {}
            datasets_size_sum = 0
            for nid in second_aggregated_params_dict.keys():
                local_dataset = node_network.get_node_by_id(nid).train_set
                local_datasets_size[nid] = len(local_dataset)
                datasets_size_sum += len(local_dataset)
            
            # 计算每个节点的本地数据集所占比例
            local_datasets_ratio = {}
            for k in local_datasets_size.keys():
                local_datasets_ratio[k] = local_datasets_size[k] / float(datasets_size_sum)
            
            second_avg_params = FedAvg(second_aggregated_params_dict, local_datasets_ratio)
        else:
            # 消融实验：不进行二次聚合，随机挑选一个种群模型的参数作为二次聚合后的参数
            eval_dict = {}
            for p_node in node_network.get_nodes_by_role(ROLE_POPULATION):
                for belong, w in population_params_dict.items():
                    transmission_unit += 1  # 种群节点互相传输模型，发生了网络传输
                    if belong not in eval_dict.keys():
                        eval_dict[belong] = {}
                    eval_dict[belong][p_node.belong] = 1
            # print(eval_dict)
            # 计算每个种群节点的总分，选择总分最高的节点作为优胜节点
            score_dict = {}
            for belong in eval_dict.keys():
                score_dict[belong] = 0
                for p_belong in eval_dict[belong].keys():
                    score_dict[belong] += eval_dict[belong][p_belong]
            # print(f"总分：{score_dict}")
            score_list = sorted(score_dict.items(), key=lambda d: d[1], reverse=True)
            max_belong, max_score = score_list[0]
            inter_best_node = node_network.get_population_node_by_belong(max_belong)
            second_avg_params = random.sample(list(population_params_dict.values()), 1)[0]

        
        # Step 7：共识（最优种群节点上传模型到区块链中）
        # 暂时省略最优种群节点对最优种内个体的再度聚合
        # 再加一条与当前区块链顶端模型f1的对比（或许可以
        
        # print("\nStep 7：共识（进化节点上传模型到全局链中，其他种群节点更新本地链）")
        # print(f"\n种群节点达成共识，由节点{inter_best_node.nid}上传模型至全局链！")
        inter_best_node.global_block_chain.add_block(deepcopy(second_avg_params))   # 把最新的二次聚合参数上传至区块链
        global_model.load_state_dict(global_block_chain.get_latest_model_parameters())  # 更新global_model的参数
        transmission_unit += 1  # 上传模型给区块链，发生了网络传输
        # 遍历种群节点，同步更新本地链
        for p_node in node_network.get_nodes_by_role(ROLE_POPULATION):
            # 为了节省内存，暂时只使用一个全局区块链
            # p_node.local_block_chain.add_block(deepcopy(second_avg_params))
            transmission_unit += 1  # 上传模型给区块链，发生了网络传输
        '''这里应该模拟PBFT增加一些网络传输，暂时搁置'''
        # 重新选举委员会节点
        if (current_conn_num + 1) % args['election_interval'] == 0:
            # print("达到选举间隔轮数，重新选举委员会节点")
            # 获取本轮中总分最低的种群（比例暂定为50%）
            reselection_num = len(score_list) // 2
            worst_belong_list = [x[0] for x in score_list[-reselection_num:]]
            # 获取这些种群对应的种群节点
            for worst_belong in worst_belong_list:
                p_node = node_network.get_population_node_by_belong(worst_belong)
                # 重置身份为个体节点
                p_node.role = ROLE_INDIVIDUAL
                # 在整个种群中重新选择种群节点（暂定为随机选择）
                new_p_node = random.sample(population_dict[worst_belong], 1)[0]
                # 多写这两行是为了防止出现移民池中节点被选为种群节点，随后在移民环节中被交换到其他种群的情况
                while new_p_node.nid in exchange_pool.keys():
                    new_p_node = random.sample(population_dict[worst_belong], 1)[0]
                new_p_node.role = ROLE_POPULATION

            # # 如果达到选举间隔，重新选举种群节点
            # """ 该步骤暂时省略，直接采用开局随机分配的身份 """
            # """后续修改成使用上一轮表现最好的个体节点"""
            #
            # population_node = random.sample(node_list, 1)[0]
            # population_node.role = ROLE_POPULATION
        
        # Step 8：移民（种群之间交换个体节点）
        """（每隔固定轮数的通信）交换不同种群内的个体节点（可选：随机挑选或者挑选出上一轮表现最好或最差的节点）"""
        if (current_conn_num + 1) % args['immigration_interval'] == 0:
            # print("达到移民间隔轮数，开始移民")
            # 移民
            # print(f"交换池：{exchange_pool.keys()}")
            # node_network.immigration(args['immigration_num'], exchange_pool)
            p_nodes = node_network.get_nodes_by_role(ROLE_POPULATION)
            # 种群节点依次挑选交换池中的节点
            for p_node in p_nodes:
                exchange_fitness_dict = {}
                for nid, w in exchange_pool.items():
                    local_model.load_state_dict(w)
                    fitness, _ = local_evaluate(local_model, p_node.test_set)
                    exchange_fitness_dict[nid] = fitness
                exchange_fitness_list = sorted(exchange_fitness_dict.items(), key=lambda d: d[1], reverse=True)
                best_nodes = []
                for item in exchange_fitness_list[:args['immigration_num']]:
                    nid = item[0]
                    best_nodes.append(node_network.get_node_by_id(nid))
                    exchange_pool.pop(nid)  # 将该节点pop出交换池

                for best_node in best_nodes:
                    best_node.belong = p_node.belong  # 修改种群
                    # best_node.local_block_chain = p_node.local_block_chain  # 修改本地链
        
        # print(f"——第 {current_conn_num + 1} / {args['comm_round_num']} 轮训练结束——")
        running_end_time = time.time()  # 每轮计时结束
        running_time += running_end_time - running_start_time
        # print(f"当前全局精英链长度为：{global_block_chain.get_len()}")
        if (current_conn_num + 1) % args['test_interval'] == 0:
            print("使用完整测试集测试全局模型性能……")
            # global_model.load_state_dict(global_block_chain.get_latest_model_parameters())
            fitness, loss = local_evaluate(global_model, test_set)
            global_model.to('cpu')
            print(f"第 {current_conn_num + 1} 轮的模型 fitness = {fitness:.4f}, loss = {loss:.4f}")
            if fitness > best_fitness:
                best_fitness = fitness
                print(f'第 {current_conn_num + 1} 轮的模型在测试集上诞生了目前最好的表现！')
                
            experiment_result.append({
                'round': current_conn_num + 1,
                'fitness': fitness,
                'loss': loss,
                'best_fitness': best_fitness,
                'transmission_unit': transmission_unit,
                'running_time': str(int(running_time))
            })
    
    print("********* 达到设定的最大通信轮数，程序终止 *********")
    
    save_experiment_result(experiment_result, args['log_name'])
    # print("实验过程保存完毕，相关数据：")
    # for row in experiment_result:
    #     print(row)
    
    end_time = get_current_time()
    minutes, seconds = experiment_time(start_time, end_time)
    print(f"\n本次实验结束，持续时间：{start_time} 至 {end_time}，共花费{minutes}分{seconds}秒")

