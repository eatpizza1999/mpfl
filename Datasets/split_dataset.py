import random
from random import shuffle

import numpy as np
from matplotlib import pyplot as plt
from torch.utils.data import Dataset, Subset


def split_dataset_iid(dataset, split_num: int):
    node_train_set_list = []  # 保存每个节点的本地数据集的列表
    train_set_size = len(dataset)  # 原始数据集的大小
    split_ratio = 1 / split_num  # 每个节点占原始数据集的比例
    split_dataset_size = int(train_set_size * split_ratio)  # 每个节点的数据集大小
    index_list = list(range(train_set_size))  # 原始数据集的下标列表
    for _ in range(split_num):
        # 根据下表列表从原始数据集中划分出子数据集
        node_train_set_list.append(Subset(dataset, random.sample(index_list, split_dataset_size)))
    return node_train_set_list


def split_dataset_non_iid(dataset, split_num, dirichlet_alpha):
    dataset_list = []  # 保存每个节点的本地数据集的列表
    train_labels = dataset.targets  # 所有的label，是个一维向量
    train_labels = train_labels.int()   # FEMNIST读出来的targets是默认float类型，如果这里不改一下后面就会报错
    split_ids = dirichlet_split(train_labels, dirichlet_alpha, split_num)
    for indices in split_ids:
        # 额外加个shuffle打乱每个数据集的有序排列（不然都是从0开始往后排）
        index_list = list(indices)
        shuffle(index_list)
        dataset_list.append(Subset(dataset, index_list))
    return dataset_list


def dirichlet_split(labels, alpha, client_num):
    '''
    按照参数为alpha的Dirichlet分布将样本索引集合划分为n_clients个子集
    '''
    
    class_num = labels.max() + 1
    # (K, N) 类别标签分布矩阵X，记录每个类别划分到每个client去的比例
    label_distribution = np.random.dirichlet([alpha] * client_num, class_num)
    # (K, ...) 记录K个类别对应的样本索引集合
    class_ids = [np.argwhere(labels == y).flatten() for y in range(class_num)]
    
    # 记录N个client分别对应的样本索引集合
    client_ids = [[] for _ in range(client_num)]
    for k_ids, distribution in zip(class_ids, label_distribution):
        # np.split按照比例fracs将类别为k的样本索引k_ids划分为了N个子集
        # i表示第i个client，ids表示其对应的样本索引集合ids
        for i, ids in enumerate(np.split(k_ids, (np.cumsum(distribution)[:-1] * len(k_ids)).astype(int))):
            client_ids[i] += [ids]
    
    client_ids = [np.concatenate(ids) for ids in client_ids]
    
    # plt.figure(figsize=(12, 8))
    # plt.hist([labels[idc] for idc in client_ids], stacked=True,
    #          bins=np.arange(min(labels) - 0.5, max(labels) + 1.5, 1),
    #          label=["Client {}".format(i) for i in range(client_num)],
    #          rwidth=0.5)
    # plt.xticks(np.arange(class_num))
    # plt.xlabel("Label type")
    # plt.ylabel("Number of samples")
    # plt.legend(loc="upper right")
    # plt.title("Display Label Distribution on Different Clients")
    # plt.show()
    return client_ids


def pathological_split(dataset, class_num, client_num, class_num_per_client):
    def iid_divide(list_to_divide, group_num):
        """
        将列表`l`分为`g`个独立同分布的group（其实就是直接划分）
        每个group都有 `int(len(l)/g)` 或者 `int(len(l)/g)+1` 个元素
        返回由不同的groups组成的列表
        """
        element_num = len(list_to_divide)
        group_size = int(len(list_to_divide) / group_num)
        num_big_groups = element_num - group_num * group_size
        num_small_groups = group_num - num_big_groups
        glist = []
        for i in range(num_small_groups):
            glist.append(list_to_divide[group_size * i: group_size * (i + 1)])
        bi = group_size * num_small_groups
        group_size += 1
        for i in range(num_big_groups):
            glist.append(list_to_divide[bi + group_size * i:bi + group_size * (i + 1)])
        return glist

    data_ids = list(range(len(dataset)))
    label2index = {k: [] for k in range(class_num)}
    for index in data_ids:
        _, label = dataset[index]
        label2index[label].append(index)

    sorted_ids = []
    for label in label2index:
        sorted_ids += label2index[label]
    shard_num = client_num * class_num_per_client
    # 一共分成n_shards个独立同分布的shards
    shards = iid_divide(sorted_ids, shard_num)
    np.random.shuffle(shards)
    # 然后再将n_shards拆分为n_client份
    tasks_shards = iid_divide(shards, client_num)

    client_ids = [[] for _ in range(client_num)]
    for client_id in range(client_num):
        for shard in tasks_shards[client_id]:
            # 这里shard是一个shard的数据索引(一个列表)
            # += shard 实质上是在列表里并入列表
            client_ids[client_id] += shard
    return client_ids


def split_dataset_non_iid_2(dataset, split_num, n_classes_per_client=2):
    # 第二种非IID采样的实现：病态划分，即每个节点只能持有两个或三个标签
    dataset_list = []  # 保存每个节点的本地数据集的列表
    split_ids = pathological_split(dataset, 10, split_num, n_classes_per_client)
    for indices in split_ids:
        # 额外加个shuffle打乱每个数据集的有序排列（不然都是从0开始往后排）
        index_list = list(indices)
        shuffle(index_list)
        dataset_list.append(Subset(dataset, index_list))
    return dataset_list

