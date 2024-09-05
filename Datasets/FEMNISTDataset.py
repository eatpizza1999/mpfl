from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
from PIL import Image
import torch
import os


class FEMNIST(MNIST):
    def __init__(self, dataset_path, train=True, transform=None):
        root = dataset_path
        super(MNIST, self).__init__(root, transform=transform)
        
        training_file = f'{root}/FEMNIST/processed/femnist_train.pt'
        test_file = f'{root}/FEMNIST/processed/femnist_test.pt'
        
        self.train = train
        self.root = root
        self.transform = transform

        if train:
            data_file = training_file
        else:
            data_file = test_file

        data_and_targets = torch.load(data_file)
        self.data, self.targets = data_and_targets[0], data_and_targets[1]

    def __getitem__(self, index):
        img, target = self.data[index], int(self.targets[index])
        img = Image.fromarray(img.numpy(), mode='F')
        img = self.transform(img)
        return img, target


def get_dataset():
    path = "./Datasets"
    # path = "/home/ding-wen-qi/projects/fl/Datasets"
    transform = transforms.Compose([
        transforms.ToTensor()  # 仅对数据做转换为 tensor 格式操作
    ])
    return FEMNIST(path, True, transform), FEMNIST(path, False, transform)
    

if __name__ == '__main__':
    os.chdir("..")
    from split_dataset import split_dataset_non_iid
    print(os.getcwd())
    # 打印图片测试
    tr, te = get_dataset()
    print(tr.targets)
    print(type(tr.targets))
    # for i in tr.targets.int():
    #     print(i)
    # print(tr.targets.int())

    # sp_tr = split_dataset_non_iid(tr, 500, 0.1)
    # print([len(t) for t in sp_tr])

    # dataloader = DataLoader(te, batch_size=1)
    # for batch_idx, data in enumerate(dataloader, 0):
    #     if (batch_idx+1) % 1000 == 0:
    #         print(batch_idx, '/', len(dataloader))
    #     inputs, targets = data
    #
    #     inputs = inputs.numpy()
    #     targets = targets.numpy()
    #
    #     inputs = inputs.reshape(28, 28)
    #     targets = targets[0]
    #
    #     plt.imshow(inputs, cmap='gray')
    #     plt.axis('off')
    #     plt.savefig('./Datasets/test_images/' + str(batch_idx) + '.jpg', bbox_inches='tight', pad_inches=0.0)

    