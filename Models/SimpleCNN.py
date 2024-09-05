from copy import deepcopy

import numpy as np
import torch
from torch import nn
from PIL import Image
import matplotlib.pyplot as plt
import os

from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils


import torch.nn.functional as F

import torch.optim as optim


# class Model(nn.Module):
#     def __init__(self):
#         super(Model, self).__init__()
#         self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
#         self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
#         self.conv2_drop = nn.Dropout2d()
#         self.fc1 = nn.Linear(320, 50)
#         self.fc2 = nn.Linear(50, 10)
#
#     def forward(self, x):
#         x = F.relu(F.max_pool2d(self.conv1(x), 2))
#         x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
#         x = x.view(-1, x.shape[1]*x.shape[2]*x.shape[3])
#         x = F.relu(self.fc1(x))
#         x = F.dropout(x, training=self.training)
#         x = self.fc2(x)
#         # return F.log_softmax(x, dim=1)
#         return x


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, x.shape[1]*x.shape[2]*x.shape[3])
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return x


# class Model(nn.Module):
#     def __init__(self):
#         super(Model, self).__init__()
#         self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
#         self.pool = nn.MaxPool2d(2, 2)
#         self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
#         self.fc1 = nn.Linear(64*7*7, 1024)  # 两个池化，所以是7*7而不是14*14
#         self.fc2 = nn.Linear(1024, 512)
#         self.fc3 = nn.Linear(512, 10)
#
#     def forward(self, x):
#         x = self.pool(F.relu(self.conv1(x)))
#         x = self.pool(F.relu(self.conv2(x)))
#
#         x = x.view(-1, 64 * 7 * 7)  # 将数据平整为一维的
#         x = F.relu(self.fc1(x))
#
#         x = F.relu(self.fc2(x))
#         x = self.fc3(x)
# #         x = F.log_softmax(x,dim=1) NLLLoss()才需要，交叉熵不需要
#         return x


def local_update(model, train_set, learning_rate, batch_size, epoch_num):
    """
    节点使用本地训练集，训练一个epoch
    本地训练使用的相关超参数已经经过测试，故直接写死
    :return: 训练好的模型参数
    """
    # 固定好的一些超参数（目前不打算再改，后续有需要可以作为该函数的参数，作为超参数）
    device = torch.device("cuda")  # 使用GPU训练
    # learning_rate = 5e-2  # 学习率
    # batch_size = 64  # 批次大小
    # epoch_num = 1  # 迭代轮数
    
    # 获取model（之后要改成从文件路径读取）
    model.to(device)
    
    # 初始化optimizer（目前固定为SGD）
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    # optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    
    # 初始化dataloader
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=False)
    # print(f"DataLoader构造完毕，数据集大小为{len(train_loader)}")
    
    # 初始化loss函数
    loss_function = nn.CrossEntropyLoss()
    
    total_loss = 0.0
    
    # 训练一个epoch
    for epoch in range(epoch_num):
        model.train()
        model.zero_grad()
        for step, batch in enumerate(train_loader):
            # 正向传播
            optimizer.zero_grad()
            
            batch_inputs, batch_labels = batch
            batch_inputs = batch_inputs.to(device)
            batch_labels = batch_labels.to(device)
            outputs = model.forward(batch_inputs)
            
            # 计算交叉熵
            loss = loss_function(outputs, batch_labels)  # 求该batch内所有样本的平均交叉熵
            
            total_loss += loss.item()
            
            # 反向传播
            loss.backward()
            # 参数更新
            optimizer.step()
            
            # # 打印训练过程
            # if step > 0 and step % 100 == 0:
            #     print("epoch: %d / %d, step: %d / %d" % (epoch+1, epoch_num, step, len(train_loader)))
        
        # local_evaluate(model, test_data)
    
    # 打印本地训练的结果
    average_training_loss = total_loss / (len(train_loader) * epoch_num)
    # print("训练完成, Average training loss: %.4f" % average_training_loss)
    return average_training_loss


def local_update_FedProx(model, train_set, global_model, mu):
    """
    节点使用本地训练集，训练一个epoch
    本地训练使用的相关超参数已经经过测试，故直接写死
    :return: 训练好的模型参数
    """
    # 固定好的一些超参数（目前不打算再改，后续有需要可以作为该函数的参数，作为超参数）
    device = torch.device("cuda")  # 使用GPU训练
    learning_rate = 1e-3  # 学习率
    batch_size = 64  # 批次大小
    epoch_num = 1  # 迭代轮数
    
    # 获取model（之后要改成从文件路径读取）
    model.to(device)
    global_model.to(device)
    
    # 初始化optimizer（目前固定为SGD）
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    
    # 初始化dataloader
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=False)
    # print(f"DataLoader构造完毕，数据集大小为{len(train_loader)}")
    
    # 初始化loss函数
    loss_function = nn.CrossEntropyLoss()
    
    # 训练一个epoch
    for epoch in range(epoch_num):
        model.train()
        model.zero_grad()
        
        epoch_total_loss = 0
        
        for step, batch in enumerate(train_loader):
            # 正向传播
            optimizer.zero_grad()
            
            batch_inputs, batch_labels = batch
            batch_inputs = batch_inputs.to(device)
            batch_labels = batch_labels.to(device)
            outputs = model.forward(batch_inputs)
            
            # 计算交叉熵
            loss = loss_function(outputs, batch_labels)  # 求该batch内所有样本的平均交叉熵
            
            # 计算近端项
            proximal_term = 0.0
            for w, w_t in zip(model.parameters(), global_model.parameters()):
                proximal_term += (w - w_t).norm(2)
            
            # 计算总loss
            loss += (mu / 2) * proximal_term
            
            epoch_total_loss += loss.item()
            
            # 反向传播
            loss.backward()
            # 参数更新
            optimizer.step()
            
            # 打印训练过程
            if step > 0 and step % 100 == 0:
                print("epoch: %d / %d, step: %d / %d, average loss: %.4f" % (
                epoch + 1, epoch_num, step, len(train_loader), (epoch_total_loss / step)))
        
        # local_evaluate(model, test_data)
        # 打印本地训练的结果
        print("训练完成, Average training loss: %.4f" % (epoch_total_loss / len(train_loader)))
    

class MyScaffoldOptimizer(Optimizer):
    def __init__(self, params, lr, weight_decay, c, ci):
        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        defaults = dict(lr=lr, weight_decay=weight_decay)
        self.c = c
        self.ci = ci
        super(MyScaffoldOptimizer, self).__init__(params, defaults)
    
    def step(self, closure=None):
        for param_group in self.param_groups:
            for p, c, ci in zip(param_group['params'], self.c.values(), self.ci.values()):
                if p.grad is None:
                    continue
                # print(f"c:{c}, ci:{ci}")
                # print(f"2c:{c.data}, ci:{ci.data}")
                dp = p.grad.data + c.data - ci.data
                p.data = p.data - dp.data * param_group['lr']
                

class MyScaffoldOptimizer2(optim.SGD):
    def __init__(self, params, lr, c, ci):
        self.c = c
        self.ci = ci
        super(MyScaffoldOptimizer2, self).__init__(params, lr=lr)
    
    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()
        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']
            
            for p, c, ci in zip(group['params'], self.c.values(), self.ci.values()):
                # print(type(p))
                # print(p)
                if p.grad is None:
                    continue
                d_p = p.grad.data + c.data - ci.data
                p.data = p.data - d_p.data * group['lr']
                
        # for group in self.param_groups:
        #     weight_decay = group['weight_decay']
        #     momentum = group['momentum']
        #     dampening = group['dampening']
        #     nesterov = group['nesterov']
        #
        #     for p, c, ci in zip(group['params'], self.c.values(), self.ci.values()):
        #         # print(type(p))
        #         # print(p)
        #         if p.grad is None:
        #             continue
        #         d_p = p.grad.data
        #         if weight_decay != 0:
        #             d_p.add_(weight_decay, p.data)
        #         if momentum != 0:
        #             param_state = self.state[p]
        #             if 'momentum_buffer' not in param_state:
        #                 buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
        #             else:
        #                 buf = param_state['momentum_buffer']
        #                 buf.mul_(momentum).add_(1 - dampening, d_p)
        #             if nesterov:
        #                 d_p = d_p.add(momentum, buf)
        #             else:
        #                 d_p = buf
        #         # d_p.add_(torch.Tensor(list(self.c.values())))
        #         # d_p.add_(-torch.Tensor(list(self.ci.values())))
        #         d_p.add_(c.data)
        #         d_p.add_(-ci.data)
        #         p.data.add_(-group['lr'], d_p)
        
        return loss

        
def local_update_SCAFFOLD(model, train_set, c, ci):
    """
    节点使用本地训练集，训练一个epoch
    本地训练使用的相关超参数已经经过测试，故直接写死
    :return: 训练好的模型参数
    """
    
    # 固定好的一些超参数（目前不打算再改，后续有需要可以作为该函数的参数，作为超参数）
    device = torch.device("cuda")  # 使用GPU训练
    learning_rate = 0.01  # 学习率
    batch_size = 64  # 批次大小
    epoch_num = 1  # 迭代轮数
    
    # 获取model（之后要改成从文件路径读取）
    model.to(device)
    
    x = deepcopy(model.state_dict())
    
    # 初始化optimizer（目前固定为SGD）
    # optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    optimizer = MyScaffoldOptimizer2(model.parameters(), lr=learning_rate, c=c, ci=ci)
    
    # 初始化dataloader
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=False)
    # print(f"DataLoader构造完毕，数据集大小为{len(train_loader)}")
    
    # 初始化loss函数
    loss_function = nn.CrossEntropyLoss()

    # 训练一个epoch
    for epoch in range(epoch_num):
        model.train()
        model.zero_grad()
        
        epoch_total_loss = 0
        
        for step, batch in enumerate(train_loader):
            # 正向传播
            optimizer.zero_grad()
            
            batch_inputs, batch_labels = batch
            batch_inputs = batch_inputs.to(device)
            batch_labels = batch_labels.to(device)
            outputs = model.forward(batch_inputs)
            
            # 计算交叉熵
            # loss = cross_entropy(outputs, batch_labels, reduction='sum')  # 求该batch内所有样本的交叉熵之和
            loss = loss_function(outputs, batch_labels)  # 求该batch内所有样本的平均交叉熵
            
            epoch_total_loss += loss.item()
            
            # 反向传播
            loss.backward()
            # 参数更新
            optimizer.step()
            
            # 打印训练过程
            if step > 0 and step % 100 == 0:
                print("epoch: %d / %d, step: %d / %d, average loss: %.4f" % (
                    epoch + 1, epoch_num, step, len(train_loader), (epoch_total_loss / step)))
            
        # 打印本地训练的结果
        print("训练完成, Average training loss: %.4f" % (epoch_total_loss / len(train_loader)))
    
    # update c
    # c+ <- ci - c + 1/(steps * lr) * (x-yi)
    yi = {}
    for k, v in model.named_parameters():
        yi[k] = v.data.clone()
    
    c_plus = {}

    local_steps = epoch_num * len(train_loader)
    for k in x.keys():
        c_plus[k] = ci[k] - c[k] + (x[k] - yi[k]) / (local_steps * learning_rate)

    for k in c_plus.keys():
        ci[k] = deepcopy(c_plus[k])


def local_evaluate(model, eval_set):
    # 固定好的一些超参数（目前不打算再改，后续有需要可以挪到init函数里，作为可修改参数）
    device = torch.device("cuda")  # 使用GPU训练
    batch_size = 64  # 批次大小
    # class_num = eval_set.class_num  # 标签分类数量，解决Subset丢失class_num问题之后再改回来，现在这么写会报错
    class_num = 10  # 标签分类数量
    
    # 初始化验证数据集的加载器（直接使用该节点被分配到的test_set属性）
    valid_loader = DataLoader(eval_set, batch_size=batch_size, shuffle=False)
    
    # 初始化loss函数
    loss_function = nn.CrossEntropyLoss()
    
    model.to(device)
    model.eval()
    total_eval_loss = 0
    TP = [0] * class_num
    FP = [0] * class_num
    FN = [0] * class_num
    for step, batch in enumerate(valid_loader):
        with torch.no_grad():
            # 正向传播
            batch_inputs, batch_labels = batch
            batch_inputs = batch_inputs.to(device)
            labels = batch_labels.to(device)
            
            outputs = model.forward(batch_inputs)
            # print(batch_inputs)
            # print(labels)
            # print(outputs)
        
        # if step > 0 and step % 100 == 0:
        #     print("[evaluate] step: %d / %d" % (step, len(valid_loader)))
        
        # 计算loss
        # loss = cross_entropy(outputs, labels.long(), reduction='sum')
        loss = loss_function(outputs, labels)
        total_eval_loss += loss.item()
        
        _, predict_labels = torch.max(outputs.data, 1)
        # print(predict_labels)
        # predict_labels = torch.argmax(outputs, -1)
        # labels = torch.argmax(labels, -1)
        # print(labels)
        # print(type(labels))
        for i in range(len(labels)):
            if predict_labels[i] == labels[i]:
                TP[predict_labels[i]] += 1
            else:
                FP[predict_labels[i]] += 1
                FN[labels[i]] += 1
    
    TP_sum = sum(TP)
    FP_sum = sum(FP)
    FN_sum = sum(FN)
    if TP_sum + FP_sum == 0:
        precision_micro = 0
    else:
        precision_micro = float(TP_sum) / float(TP_sum + FP_sum)
    if TP_sum + FN_sum == 0:
        recall_micro = 0
    else:
        recall_micro = float(TP_sum) / float(TP_sum + FN_sum)
    if precision_micro + recall_micro == 0:
        f1_micro = 0
    else:
        f1_micro = float(2 * precision_micro * recall_micro) / float(precision_micro + recall_micro)
    
    precision = [0.0] * class_num
    recall = [0.0] * class_num
    for i in range(class_num):
        if TP[i] + FP[i] != 0:
            precision[i] = float(TP[i]) / float(TP[i] + FP[i])
        if TP[i] + FN[i] != 0:
            recall[i] = float(TP[i]) / float(TP[i] + FN[i])
    precision_macro = sum(precision) / float(class_num)
    recall_macro = sum(recall) / float(class_num)
    if precision_macro + recall_macro == 0:
        f1_macro = 0
    else:
        f1_macro = float(2 * precision_macro * recall_macro) / float(precision_macro + recall_macro)
    
    # print("验证结果：micro_f1: %.2f, macro_f1: %.2f, average loss: %.4f" % (
    #     f1_micro, f1_macro, (total_eval_loss / len(valid_loader))))
    average_loss = total_eval_loss / len(valid_loader)
    return f1_micro, average_loss


def global_test(model, test_set):
    # print("正在使用完整测试集验证模型性能……")
    # 固定好的一些超参数（目前不打算再改，后续有需要可以挪到init函数里，作为可修改参数）
    device = torch.device("cuda")  # 使用GPU训练
    batch_size = 64  # 批次大小
    label_num = 10  # 标签分类数量
    
    # # 之后要彻底解耦，换成对应数据集的传入参数
    # transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.5], std=[0.5])])
    # test_set = datasets.MNIST(
    #     root="D:/PycharmProjects/BlockchainBasedFedaratedLearning/MPFL/Datasets/",  # 数据的路径
    #     train=False,  # 不使用训练数据集
    #     transform=transform,  # 将数据转化为torch使用的张量，范围为［0，1］
    #     download=False  # 不下载数据
    # )
    
    # 初始化验证数据集的加载器（直接使用该节点被分配到的test_set属性）
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    
    # 初始化loss函数
    loss_function = nn.CrossEntropyLoss()
    
    model.to(device)
    model.eval()
    total_eval_loss = 0
    TP = [0] * label_num
    FP = [0] * label_num
    FN = [0] * label_num
    for step, batch in enumerate(test_loader):
        with torch.no_grad():
            # 正向传播
            batch_inputs, batch_labels = batch
            batch_inputs = batch_inputs.to(device)
            labels = batch_labels.to(device)
            
            outputs = model.forward(batch_inputs)
        
        # 计算loss
        loss = loss_function(outputs, labels)
        total_eval_loss += loss.item()
        
        _, predict_labels = torch.max(outputs.data, 1)
        # print(predict_labels)
        # predict_labels = torch.argmax(outputs, -1)
        # labels = torch.argmax(labels, -1)
        # print(labels)
        # print(type(labels))
        for i in range(len(labels)):
            if predict_labels[i] == labels[i]:
                TP[predict_labels[i]] += 1
            else:
                FP[predict_labels[i]] += 1
                FN[labels[i]] += 1
    
    TP_sum = sum(TP)
    FP_sum = sum(FP)
    FN_sum = sum(FN)
    if TP_sum + FP_sum == 0:
        precision_micro = 0
    else:
        precision_micro = float(TP_sum) / float(TP_sum + FP_sum)
    if TP_sum + FN_sum == 0:
        recall_micro = 0
    else:
        recall_micro = float(TP_sum) / float(TP_sum + FN_sum)
    if precision_micro + recall_micro == 0:
        f1_micro = 0
    else:
        f1_micro = float(2 * precision_micro * recall_micro) / float(precision_micro + recall_micro)
    
    precision = [0.0] * label_num
    recall = [0.0] * label_num
    for i in range(label_num):
        if TP[i] + FP[i] != 0:
            precision[i] = float(TP[i]) / float(TP[i] + FP[i])
        if TP[i] + FN[i] != 0:
            recall[i] = float(TP[i]) / float(TP[i] + FN[i])
    precision_macro = sum(precision) / float(label_num)
    recall_macro = sum(recall) / float(label_num)
    if precision_macro + recall_macro == 0:
        f1_macro = 0
    else:
        f1_macro = float(2 * precision_macro * recall_macro) / float(precision_macro + recall_macro)
    
    average_loss = total_eval_loss / len(test_loader)
    # print("验证结果：micro_f1: %.2f, macro_f1: %.2f, average loss: %.4f" % (
    #     f1_micro, f1_macro, average_loss))
    return f1_micro, average_loss


if __name__ == '__main__':
    model = Model()
    print(model.state_dict())
