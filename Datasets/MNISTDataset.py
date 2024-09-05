import os
import random
import numpy as np
import torch
from torchvision import datasets, transforms


def get_dataset():
    # transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.5], std=[0.5])])
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    path = "./Datasets"
    train_set = datasets.MNIST(root=path, train=True, transform=transform, download=False)
    test_set = datasets.MNIST(root=path, train=False, transform=transform, download=False)
    return train_set, test_set


if __name__ == '__main__':
    os.chdir("..")
    from split_dataset import split_dataset_non_iid
    seed = 0
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)  # 禁止hash随机化
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    tr, te = get_dataset()
    dl = split_dataset_non_iid(tr, 20, 10)
    # print([len(t) for t in dl])
    # print(tr.classes)
    # print(list("123"))

    
