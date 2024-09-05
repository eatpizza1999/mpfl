import os

import torch.nn as nn
import torch
from torch import optim
from torch.utils.data import DataLoader
import torch.nn.functional as F


# 来自MNIST用的简易CNN，表现排第二，仅次于L
# class Model(nn.Module):
#     def __init__(self):
#         super(Model, self).__init__()
#         self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
#         self.pool = nn.MaxPool2d(2, 2)
#         self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
#         self.fc1 = nn.Linear(64*7*7, 1024)  # 两个池化，所以是7*7而不是14*14
#         self.fc2 = nn.Linear(1024, 512)
#         self.fc3 = nn.Linear(512, 62)
#
#     def forward(self, x):
#         x = self.pool(F.relu(self.conv1(x)))
#         x = self.pool(F.relu(self.conv2(x)))
#         x = x.view(-1, 64 * 7 * 7)
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = self.fc3(x)
#         return x

# # 来自FedLab中专用于FEMNIST的模型结构，收敛最快
# class Model(nn.Module):
#     def __init__(self):
#         super(Model, self).__init__()
#         self.conv2d_1 = nn.Conv2d(1, 32, kernel_size=3)
#         self.max_pooling = nn.MaxPool2d(2, stride=2)
#         self.conv2d_2 = nn.Conv2d(32, 64, kernel_size=3)
#         self.dropout_1 = nn.Dropout(0.25)
#         self.flatten = nn.Flatten()
#         self.linear_1 = nn.Linear(9216, 128)
#         self.dropout_2 = nn.Dropout(0.5)
#         self.linear_2 = nn.Linear(128, 62)
#         self.relu = nn.ReLU()
#
#     def forward(self, x):
#         x = self.conv2d_1(x)
#         x = self.relu(x)
#         x = self.conv2d_2(x)
#         x = self.relu(x)
#         x = self.max_pooling(x)
#         x = self.dropout_1(x)
#         x = self.flatten(x)
#         x = self.linear_1(x)
#         x = self.relu(x)
#         x = self.dropout_2(x)
#         x = self.linear_2(x)
#         return x


# 结构最简单，训练速度最快，但不容易收敛
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=3)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        # self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 128)
        self.fc2 = nn.Linear(128, 62)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        # x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = x.view(-1, x.shape[1]*x.shape[2]*x.shape[3])
        x = F.relu(self.fc1(x))
        # x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return x


def local_update(model, train_set, learning_rate, batch_size, epoch_num):
    """
    节点使用本地训练集，训练一个epoch
    本地训练使用的相关超参数已经经过测试，故直接写死
    :return: 训练好的模型参数
    """
    # 固定好的一些超参数（目前不打算再改，后续有需要可以作为该函数的参数，作为超参数）
    device = torch.device("cuda")  # 使用GPU训练
    # learning_rate = 5e-2  # 学习率
    # # learning_rate = 0.1  # 学习率
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
            # if step > 0 and step % 1000 == 0:
            #     print("epoch: %d / %d, step: %d / %d" % (epoch+1, epoch_num, step, len(train_loader)))
            #     print("Average Training Loss: %.2f" % (total_loss / step))
        
        # local_evaluate(model, test_data)
    
    # 打印本地训练的结果
    average_training_loss = total_loss / (len(train_loader) * epoch_num)
    # print("训练完成, Average training loss: %.4f" % average_training_loss)
    return average_training_loss


def local_update_FedProx(model, train_set, global_model, mu):
    pass


def local_evaluate(model, eval_set):
    # 固定好的一些超参数（目前不打算再改，后续有需要可以挪到init函数里，作为可修改参数）
    device = torch.device("cuda")  # 使用GPU训练
    batch_size = 100  # 批次大小
    class_num = 62  # 标签分类数量
    
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
    # label_num = 10  # 标签分类数量
    label_num = 62
    
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
        # print(labels)
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
    # os.chdir("..")
    # from MPFL.Datasets.FEMNISTDataset import get_dataset
    
    # m = Model()
    # train_data, test_data = get_dataset()
    # local_update(m, test_data, 0.1, 100, 1)
    model = Model()
    total_params = sum(p.numel() for p in model.parameters())
    print(f'{total_params:,} total parameters.')
    total_trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad)
    print(f'{total_trainable_params:,} training parameters.')