# 超参数
ROLE_INDIVIDUAL = 'I'   # 代表个体节点的身份
ROLE_POPULATION = 'P'   # 代表种群节点的身份


class Node:
    """
    节点类
    """
    
    def __init__(self, node_id, role, belong, train_set, test_set, local_block_chain, global_block_chain):
        # 节点自身的属性
        self.nid = node_id      # 节点的id
        self.role = role    # 节点的身份（分为个体节点I和种群节点P）
        self.belong = belong    # 节点所属种群的id（注意，并不是该种群的当前种群节点的id）
        
        # 与模型训练相关的属性
        self.train_set = train_set  # 节点本地的训练集
        self.test_set = test_set    # 节点本地的测试集
        # self.model = model  # 节点所使用的模型
        # self.local_update_function = local_update_function   # 节点的训练函数（为了适用于不同模型所以在模型类中实现）
        # self.local_evaluate_function = local_evaluate_function  # 节点的验证函数（为了适用于不同模型所以在模型类中实现）
        
        # 与区块链相关的属性
        self.local_block_chain = local_block_chain    # 种内精英链
        self.global_block_chain = global_block_chain  # 全局精英链
        # self.malicious = malicious      # 恶意节点（有具体类型）
    
    def print_status(self):
        return f"节点id：{self.nid}，种群{self.belong}，身份{self.role}"
    
    # def local_update(self):
    #     """
    #     节点使用本地训练集，训练一个epoch
    #     本地训练使用的相关超参数已经经过测试，故直接写死
    #     :return:
    #     """
    #
    #     self.local_update_function(self.model, self.train_set)
    #
    #
    # def differential_privacy(self):
    #     """
    #     对模型参数使用差分隐私
    #     :return: None
    #     """
    #     self.model.load_state_dict(add_gaussian_noise(self.model.state_dict()))
    #
    # def load_model_parameters(self, parameters):
    #     self.model.load_state_dict(parameters)
    #
    # def evaluate_model(self, model):
    #     # print("正在验证模型")
    #     fitness = self.local_evaluate_function(model, self.test_set)
    #     return fitness
    

def add_gaussian_noise(parameters, sigma=0.1):
    """
    差分隐私（高斯噪声）
    :param parameters: 输入的模型参数（通常为一层的参数）
    :param sigma: 噪声幅度（隐私预算）
    :return: 加噪后的模型参数
    """
    
    skip_list = {'bert.embeddings.position_ids'}  # 这一层参数的类型是Long，跟Float类型不兼容
    # linear_parameter_names = {'linear.weight', 'linear.bias'}     # 线性层的参数
    for k in parameters.keys():
        # 过滤特定参数层
        if k in skip_list:
            continue
        
        # 获取该层模型参数，计算方差
        # 之后改成差分隐私（先通过实验找到最佳梯度剪裁阈值C）
        p = parameters[k]
        std = sigma * p.data.std()
        
        # 生成高斯噪声并添加至模型特定的参数层
        noise = torch.cuda.FloatTensor(p.shape).normal_(0, std)
        parameters[k] = parameters[k].add_(noise)
    
    return parameters
