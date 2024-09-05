from copy import deepcopy

from Node import Node, ROLE_INDIVIDUAL, ROLE_POPULATION
from typing import Set, List, Dict, Any
import random


class Network:
    """
    节点组成的网络，主要用于实现一些发生在节点之间的操作，如获取特定节点、节点通信等
    """
    
    def __init__(self):
        # 节点网络的基本数据类型是节点组成的字典，key为节点id，value为节点
        self.node_dict = {}
    
    def get_populations(self) -> Dict[Any, List[Node]]:
        """
        返回根据种群划分的节点组成的字典，key为种群id，value为节点组成的列表
        :return:
        """
        population_dict = {}
        for node in self.node_dict.values():
            if node.belong in population_dict:
                population_dict[node.belong].append(node)
            else:
                population_dict[node.belong] = [node]
        return population_dict
    
    def get_populations_expect_committee(self) -> Dict[Any, List[Node]]:
        """
        返回根据种群划分的节点组成的字典（刨除了种群节点避免BUG），key为种群id，value为节点组成的列表
        :return:
        """
        population_dict = {}
        for node in self.node_dict.values():
            if node.role == ROLE_POPULATION:
                continue
            if node.belong in population_dict:
                population_dict[node.belong].append(node)
            else:
                population_dict[node.belong] = [node]
        return population_dict
    
    def search_population_node(self, nodes) -> Node:
        """
        从输入的节点中找出种群节点并返回，若不存在种群节点则返回None
        :param nodes:
        :return:
        """
        for node in nodes:
            if node.role == ROLE_POPULATION:
                return node

    def add_node(self, node: Node) -> None:
        """
        向网络中添加节点
        :param node: 需要添加的节点
        :return: None
        """
        self.node_dict[node.nid] = node
    
    def get_node_by_id(self, node_id) -> Node:
        """
        根据节点id获取节点本身
        :param node_id: 节点id
        :return: 节点
        """
        return self.node_dict[node_id]
    
    def get_nodes_by_ids(self, ids) -> Set[Node]:
        """
        根据多个节点id获取节点集合
        :param ids: 节点id组成的列表或集合
        :return: 节点组成的集合
        """
        node_set = set()
        for node in self.node_dict.values():
            if node.nid in ids:
                node_set.add(node)
        return node_set
    
    def get_nodes_by_role(self, role: str) -> Set[Node]:
        """
        根据节点身份获取对应身份的节点组成的集合
        :param role: 节点身份
        :return: 节点组成的集合
        """
        node_set = set()
        for node in self.node_dict.values():
            if node.role == role:
                node_set.add(node)
        return node_set
    
    def print_status(self) -> str:
        """
        打印目前的节点状况
        :return: 节点状况
        """
        status = ""
        for node in self.node_dict.values():
            status += f'节点{node.nid}({node.belong}, {node.role}) | '
        return status
    
    def get_individual_node_by_population_node(self, p_node: Node):
        """
        根据种群节点的id获取它对应的个体节点
        :return:
        """
        node_set = set()
        pid = p_node.belong
        for node in self.node_dict.values():
            if node.role == ROLE_INDIVIDUAL and node.belong == pid:
                node_set.add(node)
        return node_set
    
    def get_nodes_by_belong(self, belong):
        """
        根据种群id获取该种群对应的所有节点（包括个体节点和种群节点）
        :param belong:
        :return:
        """
        node_set = set()
        for node in self.node_dict.values():
            if node.belong == belong:
                node_set.add(node)
        return node_set
    
    def get_population_node_by_belong(self, belong):
        for node in self.node_dict.values():
            if node.role == ROLE_POPULATION and node.belong == belong:
                return node
        print("未能搜索到对应种群的种群节点！检查种群选举算法是否出现了BUG！")
        exit()
    
    def immigration(self, immigrant_node_num, exchange_pool: dict):
        """
        移民操作（当前为随机选择）
        :return:
        """
        
        # """注意，该算法还不完整，目前只支持每个种群交换1个节点，超过1个节点需要重新设计算法"""
        # # print("！！！移民算法被调用！！！")
        # # print(f"population_fitness_dict:{population_fitness_dict}")
        # exchange_pool = []  # 交换池，节点组成的列表
        # # 获取每个种群内本轮表现最差的节点
        # p_nodes = self.get_nodes_by_role(ROLE_POPULATION)
        # for p_node in p_nodes:
        #     # print(f"p_node:{p_node.nid}")
        #     worst_nodes_ids = []
        #     fitness_dict = population_fitness_dict[p_node.belong]
        #     # print(f"fitness_dict:{fitness_dict}")
        #     worst_nodes_list = sorted(fitness_dict.items(), key=lambda d: d[1], reverse=False)
        #     # print(f"worst_fitness_dict_list:{worst_fitness_dict_list}")
        #     index = 0
        #     while len(worst_nodes_ids) < immigrant_node_num:
        #         worst_nid = worst_nodes_list[index][0]
        #         index += 1
        #         # 额外判断是否是种群节点，如果是种群节点则跳过
        #         if worst_nid == p_node.nid:
        #             continue
        #         else:
        #             worst_nodes_ids.append(worst_nid)
        #
        #     exchange_pool += self.get_nodes_by_ids(worst_nodes_ids)
        
        # print(f"exchange_pool:{[no.nid for no in exchange_pool]}")

        p_nodes = self.get_nodes_by_role(ROLE_POPULATION)
        # 种群节点依次挑选交换池中的节点
        for p_node in p_nodes:
            for _ in immigrant_node_num:
                nid = random.sample(list(exchange_pool.keys()), 1)[0]   # 随机选择交换池中的一个节点
                w = exchange_pool.pop(nid)  # 将该节点pop出交换池，并获取其模型参数
                
            # 由于没法验证性能，暂定随机选择（exchange_pool已经shuffle过，这里按顺序选取没有问题）
            best_nodes = exchange_pool[:immigrant_node_num]
            
            for best_node in best_nodes:
                exchange_pool.remove(best_node)     # 从交换池中删掉被选中的节点
                best_node.belong = p_node.belong    # 修改种群
                best_node.local_block_chain = p_node.local_block_chain  # 修改本地链
            
    def select_population_node(self):
        """
        重新选举每个种群内的种群节点（当前为随机选择）
        :return:
        """
        
        # 先获取不同种群的节点组成的列表组成的字典
        population_dict = {}
        for node in self.node_dict.values():
            if node.belong in population_dict:
                population_dict[node.belong].append(node)
            else:
                population_dict[node.belong] = [node]
            # 把所有节点的身份全部重置成个体节点
            node.role = ROLE_INDIVIDUAL
        
        # 在每个种群中随机选择一个节点作为种群节点
        for population in population_dict.values():
            selected_node = random.sample(population, 1)[0]
            selected_node.role = ROLE_POPULATION
        
    def select_evolution_nodes(self) -> List[Node]:
        """
        从种群节点中选择一个作为进化节点（当前为随机选择）
        
        :return: 被选中的种群节点列表（如果当前种群结果没通过表决，那么就继续找序列上的下一个种群节点作为进化节点）
        """
        
        population_nodes = self.get_nodes_by_role(ROLE_POPULATION)
        evolution_nodes = random.sample(population_nodes, len(population_nodes))
        # evolution_node.role = ROLE_EVOLUTION    # 暂时不加第三种身份
        return evolution_nodes
    