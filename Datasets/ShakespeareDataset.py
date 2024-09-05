import json
import os
import torch
from torch.utils.data import Dataset, random_split, DataLoader
import pandas as pd


class Shakespeare(Dataset):
    """
    莎士比亚数据集，共有16000条数据和2种标签
    """
    
    def __init__(self, is_train):
        # 数据集里的文本和标签所涉及的所有字符，以及这些字符对应的id
        LETTER_LIST = "\n !\"&'(),-.0123456789:;>?ABCDEFGHIJKLMNOPQRSTUVWXYZ[]abcdefghijklmnopqrstuvwxyz}"
        LETTER_DICT = dict(zip(LETTER_LIST, list(range(len(LETTER_LIST)))))
        # self.LETTER_DICT = dict(zip(LETTER_LIST, [i%2 for i in range(len(LETTER_LIST))]))  # 随机映射成2分类

        train_dataset_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Shakespeare', 'all_data_iid_0_0_keep_0_train_9.json')
        test_dataset_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Shakespeare', 'all_data_iid_0_0_keep_0_test_9.json')

        if is_train:
            dataset_path = train_dataset_path
        else:
            dataset_path = test_dataset_path

        # 读取原始json数据，解析获取其中的输入文本和标签
        with open(dataset_path, 'r') as f:
            data = json.load(f)
        texts = []
        labels = []
        for data_dict in data['user_data'].values():
            texts += data_dict['x']
            labels += data_dict['y']
        
        data = {'label': labels, 'text': texts}
        df = pd.DataFrame(data)
        # df = df.loc[0: 10000]     # 取前N条数据，快速测试用

        self.labels = pd.Series(df['label'].map(lambda x: LETTER_DICT[x]).values)
        self.texts = df['text']
        self.targets = torch.tensor(self.labels)

        # df_len_text = df['text'].str.len()
        # print(df_len_text.value_counts())

    def __len__(self):
        return self.labels.shape[0]
    
    def __getitem__(self, index):
        return self.texts[index], self.labels[index]
        # sentence, target = self.texts[index], self.labels[index]
        # indices = [self.LETTER_DICT[c] for c in sentence]
        # return indices, target

def get_dataset_test():
    from collections import Counter

    test_dataset_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Shakespeare', 'all_data.json')
    # test_dataset_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Shakespeare', 'all_data_iid_0_0.json')
    # test_dataset_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Shakespeare', 'all_data_iid_01_0_keep_0_test_9.json')
    # test_dataset_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Shakespeare', 'all_data_iid_0_0_keep_0_test_9.json')
    # test_dataset_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Shakespeare', 'all_data_iid_0_0_keep_0.json')
    
    with open(test_dataset_path, 'r') as f:
        data = json.load(f)
    
    texts = []
    labels = []
    for data_dict in data['user_data'].values():
        texts += data_dict['x']
        labels += data_dict['y']
    print(len(texts))
    print(len(labels))
    print(len(set(labels)))
    print(set(labels))
    print(Counter(labels))

    # print(type(data))
    # print(data.keys())
    # print(type(data['users']))
    # print(type(data['num_samples']))
    # print(type(data['user_data']))
    # print(data['users'][0])
    # print(data['num_samples'], len(data['num_samples']))
    # # print(data['hierarchies'])
    # # print(data['user_data'].keys())
    # # print(data['user_data'][data['users'][0]])

def get_dataset():
    train_set = Shakespeare(is_train=True)
    test_set = Shakespeare(is_train=False)
    return train_set, test_set

if __name__ == '__main__':
    # get_dataset_test()
    tr, te = get_dataset()
    print(tr[0])
    
