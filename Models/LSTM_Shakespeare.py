import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
from torch.utils.data import DataLoader


# # 创建LSTM模型
# class Model(nn.Module):
#     def __init__(self):
#         super(Model, self).__init__()
        
#         # words = []
#         embeddings = []
#         with open('./Models/GloVe/glove.6B/glove.6B.50d.txt', encoding='utf-8') as f:
#             for line in f:
#                 values = line.split()
#                 # word = values[0]
#                 # words.append(word)
#                 embedding = np.asarray(values[1:], dtype='float32')
#                 embeddings.append(embedding)
#         # words = ['<pad>'] + ['<unk>'] + words
#         embeddings = [np.zeros_like(embeddings[0], dtype='float32')] + [
#             np.zeros_like(embeddings[0], dtype='float32')] + embeddings
#         # self.word2idx = {o: i for i, o in enumerate(words)}
#         # idx2word = {i: o for i, o in enumerate(words)}
        
#         # 构建词索引
#         embedding_matrix = np.stack(embeddings)
        
#         self.embedding = nn.Embedding.from_pretrained(torch.FloatTensor(embedding_matrix))
#         self.lstm = nn.LSTM(50, 128, 1, batch_first=True)
#         self.fc = nn.Linear(128, 79)

#     def forward(self, x):
#         # x = tokenize(x, self.word2idx)
#         embedded = self.embedding(x)
#         out, _ = self.lstm(embedded)
#         out = self.fc(out[:, -1, :])  # 使用最后一个时间步的输出
#         return out

# 创建LSTM模型
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.embed = nn.Embedding(80, 8)
        self.lstm = nn.LSTM(8, 256, 2, batch_first=True)
        self.drop = nn.Dropout()
        self.out = nn.Linear(256, 80)

    def forward(self, x):
        x = self.embed(x)
        x, hidden = self.lstm(x)
        x = self.drop(x)
        return self.out(x[:, -1, :])
    
    
def pad_input(sentences, seq_len):
    """
    将句子长度固定为`seq_len`，超出长度的从后面截断，长度不足的在后面补0
    """
    features = np.zeros((len(sentences), seq_len), dtype=int)
    for ii, review in enumerate(sentences):
        if len(review) != 0:
            # features[ii, :len(review)] = np.array(review)[:seq_len]
            features[ii, -len(review):] = np.array(review)[:seq_len]
    # print(features)
    return features

    
def tokenize(batch_sentences, word2idx):
    LETTER_LIST = "\n !\"&'(),-.0123456789:;>?ABCDEFGHIJKLMNOPQRSTUVWXYZ[]abcdefghijklmnopqrstuvwxyz}"
    word2idx = dict(zip(LETTER_LIST, list(range(len(LETTER_LIST)))))
    s = []
    for sentence in batch_sentences:
        # 分词
        indexes = []
        for c in sentence:
            indexes.append(word2idx[c])
        s.append(indexes)
    # print(f"s: {s}")
    # # 填充（固定80长度，不用填充）
    # s = pad_input(s, 200)
    # 转类型
    s = torch.LongTensor(s)
    # print(s)
    return s


def local_update(model, train_set, learning_rate, batch_size, epoch_num, word2idx):
    # 定义损失函数和优化器
    device = torch.device("cuda")
    loss_function = nn.CrossEntropyLoss()
    # optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    # word2idx = get_word2idx()
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    # print("数据集大小：", len(train_loader))
    
    model.to(device)
    
    # 训练一个epoch
    total_loss = 0
    for epoch in range(epoch_num):
        epoch_loss = 0.0
        model.train()
        model.zero_grad()
        for step, batch in enumerate(train_loader):
            # 正向传播
            optimizer.zero_grad()
            
            batch_inputs, batch_labels = batch
            batch_inputs = tokenize(batch_inputs, word2idx)
            batch_inputs = batch_inputs.to(device)
            batch_labels = batch_labels.to(device)
            outputs = model.forward(batch_inputs)
            # print(outputs)
            # print(batch_labels)
            # 计算交叉熵
            loss = loss_function(outputs, batch_labels)  # 求该batch内所有样本的平均交叉熵
            
            total_loss += loss.item()
            epoch_loss += loss.item()
            
            # 反向传播
            loss.backward()
            # 参数更新
            optimizer.step()
            
        #     # 打印训练过程
        #     if (step + 1) % 100 == 0:
        #         print("epoch: %d / %d, step: %d / %d, avg loss: %f" % (
        #         epoch + 1, epoch_num, step + 1, len(train_loader), epoch_loss / step))
        # print(f"epoch_avg_loss: {epoch_loss / len(train_loader)}")
    
    # 打印本地训练的结果
    average_training_loss = total_loss / (len(train_loader) * epoch_num)
    # print(f"average_training_loss: {average_training_loss}")
    return average_training_loss


def local_evaluate(model, eval_set, word2idx):
    # 固定好的一些超参数（目前不打算再改，后续有需要可以挪到init函数里，作为可修改参数）
    device = torch.device("cuda")  # 使用GPU训练
    batch_size = 100  # 批次大小
    class_num = 80  # 标签分类数量
    # word2idx = get_word2idx()
    
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
            
            batch_inputs = tokenize(batch_inputs, word2idx)
            batch_inputs = batch_inputs.to(device)
            labels = batch_labels.to(device)
            
            outputs = model.forward(batch_inputs)
            # print(batch_inputs)
            # print(labels)
            # print(outputs)
        
        # if step > 0 and step % 500 == 0:
            # print("[evaluate] step: %d / %d" % (step, len(valid_loader)))
        
        # 计算loss
        # loss = cross_entropy(outputs, labels.long(), reduction='sum')
        loss = loss_function(outputs, labels)
        total_eval_loss += loss.item()
        
        _, predict_labels = torch.max(outputs.data, 1)
        # print(predict_labels)
        # print(labels)
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
    average_loss = total_eval_loss / len(valid_loader)
    # print("验证结果：micro_f1: %.2f, macro_f1: %.2f, average loss: %.4f" % (
    #     f1_micro, f1_macro, (total_eval_loss / len(valid_loader))))
    return f1_micro, average_loss


def get_word2idx():
    words = []
    # embeddings = []
    with open('./Models/GloVe/glove.6B/glove.6B.50d.txt', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            words.append(word)
            # embedding = np.asarray(values[1:], dtype='float32')
            # embeddings.append(embedding)
    words = ['<pad>'] + ['<unk>'] + words
    # embeddings = [np.zeros_like(embeddings[0], dtype='float32')] + [
    #     np.zeros_like(embeddings[0], dtype='float32')] + embeddings
    
    word2idx = {o: i for i, o in enumerate(words)}
    # idx2word = {i: o for i, o in enumerate(words)}
    # embedding_matrix = np.stack(embeddings)
    return word2idx
    

if __name__ == '__main__':
    # os.chdir("..")
    # from MPFL.Datasets.Sentiment140Dataset import get_dataset
    # from MPFL.Datasets.split_dataset import split_dataset_iid
    # # 初始化模型
    # print("正在初始化模型……")
    # m = Model()
    # print(m)
    # train_set, test_data = get_dataset()
    # # train_set_list = split_dataset_iid(train_set, 500)
    # # train_set = train_set_list[0]
    
    # stoi = get_word2idx()
    # local_update(m, train_set, 0.01, 100, 1, stoi)
    
    # local_evaluate(m, test_data, stoi)
    model = Model()
    total_params = sum(p.numel() for p in model.parameters())
    print(f'{total_params:,} total parameters.')
    total_trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad)
    print(f'{total_trainable_params:,} training parameters.')
    