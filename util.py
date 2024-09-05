import os
import random
from datetime import datetime
import pandas as pd
import numpy as np
import torch
from matplotlib import pyplot as plt


class MyLogger(object):
    """
    用于将print内容同步保存至本地txt文件的小技巧
    """
    
    def __init__(self, stream, filename):
        self.terminal = stream
        if not os.path.exists('Logger'):
            os.mkdir('Logger')
        log_path = os.path.join('Logger', f'{filename}.txt')
        self.log = open(log_path, 'a')
    
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
    
    def flush(self):
        pass


def save_experiment_result(experiment_result, file_name):
    df = pd.DataFrame(experiment_result)
    result_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Result')
    if not os.path.exists(result_dir):
        os.mkdir(result_dir)
    result_path = os.path.join(result_dir, f'{file_name}.csv')
    df.to_csv(result_path)


def experiment_time(start_time, end_time):
    time_1_struct = datetime.strptime(start_time, "%Y.%m.%d_%H.%M.%S")
    time_2_struct = datetime.strptime(end_time, "%Y.%m.%d_%H.%M.%S")
    seconds = (time_2_struct - time_1_struct).seconds
    return seconds // 60, seconds % 60


def save_loss(loss_list, file_name):
    f = open(f"./Result/{file_name}.txt", "w")
    for loss in loss_list:
        f.write(str(loss)+'\n')
    f.close()
    

def draw_loss(file_name):
    f = open(f"./Result/{file_name}.txt", "r")
    loss_list_str = f.readlines()
    loss_list = []
    for loss in loss_list_str:
        loss_list.append(float(loss))

    x = list(range(1, len(loss_list) + 1))
    
    plt.xlabel('Rounds')
    plt.ylabel('Training Loss')
    plt.title('C=20')
    
    plt.plot(x, loss_list)
    
    # plt.legend(('ep_0', 'ep_1', 'ep_2'), loc='upper right')
    plt.show()


def compare_loss(file_list, name_list):
    fig, axes = plt.subplots(1, 2)

    for file_name in file_list:
        df = pd.read_csv(f"./Result/{file_name}")
        df = df.to_dict()
        x = list(df['round'].values())
        y1 = list(df['loss'].values())
        axes[0].plot(x, y1)
        y2 = list(df['fitness'].values())
        axes[1].plot(x, y2)
    
    axes[0].set_xlabel('Round', fontdict={'weight': 'normal', 'size': 18})
    axes[0].set_ylabel('Loss', fontdict={'weight': 'normal', 'size': 18})
    axes[1].set_xlabel('Round', fontdict={'weight': 'normal', 'size': 18})
    axes[1].set_ylabel('ACC', fontdict={'weight': 'normal', 'size': 18})
    # plt.title('T')

    # for i in range(len(loss_lists)):
    #     print(loss_lists[i])
    #     x = list(range(1, len(loss_lists[i]) + 1))
    #     plt.plot(x, loss_lists[i])

    # axes[0].set_ylim((0.5, 4.5))

    axes[0].legend(
        name_list,
        loc='upper right')
    axes[1].legend(
        name_list,
        loc='lower right')
    plt.show()


def setup_random_seed(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)  # 禁止hash随机化
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    

def get_current_time():
    return datetime.now().strftime("%Y.%m.%d_%H.%M.%S")
