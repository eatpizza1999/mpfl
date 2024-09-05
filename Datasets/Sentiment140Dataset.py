import re
import os
import torch
from torch.utils.data import Dataset, random_split, DataLoader
import pandas as pd


class Sentiment140(Dataset):
    """
    情感分类数据集，共有16000条数据和2种标签
    """
    
    def __init__(self, is_train):
        dataset_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Sentiment140', 'training_nltk.csv')
        # df = pd.read_csv(dataset_path, header=None, sep=',',
        #                  names=['label', 'id', 'time', 'query', 'user', 'content'],
        #                  encoding='ISO-8859-1', dtype=str)
        
        # df = df.loc[0: 99]     # 取前N条数据，快速测试用
        
        # # 将标签映射成onehot向量
        # label_dict = {'0': 0, '4': 1}
        # label_array = df['label'].map(lambda x: label_dict[x])

        # df['content'] = df['content'].str.lower()     # 转小写
        # df['content'] = df['content'].map(lambda x: ' '.join(nltk.word_tokenize(x)))  # 分词
        # print(df['content'])
        # df.to_csv("./Datasets/Sentiment140/training_nltk.csv", columns=['label', 'content'])  # 保存

        # self.labels = label_array
        # self.texts = df['content']
        
        df = pd.read_csv(dataset_path, index_col=0, sep=',', dtype={'content': str})

        # my_array = np.array(df['content'])
        # my_tensor = torch.tensor(my_array)
        # 统计每句话里的词语数量
        # len_list = df['content'].str.count(' ').tolist()
        # counter = Counter(len_list)
        # print(counter)
        
        # df = df.loc[0: 99]     # 取前N条数据，快速测试用
        data_size = len(df)
        train_size = int(data_size * 0.9)
        # test_size = data_size - train_size
        if is_train:
            df = df.loc[0: train_size-1]
            # print(f"train df:{df}")
        else:
            df = df.loc[train_size: data_size]
            df = df.reset_index(drop=True)
            # print(f"test df:{df}")
            
        self.labels = df['label']
        # print(type(self.labels))
        # print(self.labels)
        self.texts = df['content']
        self.targets = torch.tensor(self.labels)

    def __len__(self):
        return self.labels.shape[0]
    
    def __getitem__(self, index):
        return self.texts[index], self.labels[index]


def get_dataset():
    # dataset = Sentiment140()
    # data_size = len(dataset)
    # train_size = int(data_size * 0.9)
    # test_size = data_size - train_size
    # return random_split(dataset, [train_size, test_size], generator=torch.Generator().manual_seed(0))
    train_set = Sentiment140(is_train=True)
    test_set = Sentiment140(is_train=False)
    return train_set, test_set


if __name__ == '__main__':
    os.chdir("..")
    from split_dataset import split_dataset_non_iid

    train_set, te = get_dataset()
    train_set_list = split_dataset_non_iid(train_set, 500, 1)
    print([len(t) for t in train_set_list])
    # print(train_set[0][0])
    # print(te[0][0])
    # for i in range(50):
    #     print(train_set[i][0])

    # train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=False)
    # for step, batch in enumerate(train_loader):
    #     texts, labels = batch
    #     print(labels)
    #     print(type(labels))
    #     print(type(labels[0]))

