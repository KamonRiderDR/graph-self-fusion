'''
Description: 
Author: Rui Dong
Date: 2023-10-19 12:25:58
LastEditors: Please set LastEditors
LastEditTime: 2023-11-09 17:32:42
'''

import torch
import random
import numpy as np
from sklearn.model_selection import StratifiedKFold
import time


def decrease_to_max_value(x, max_value):
    x[x > max_value] = max_value
    return x


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def k_fold(dataset, folds):
    skf = StratifiedKFold(folds, shuffle=True, random_state=114514)

    test_indices, train_indices = [], []
    for _, idx in skf.split(torch.zeros(len(dataset)), dataset.data.y):
        test_indices.append(torch.from_numpy(idx).to(torch.long))

    val_indices = [test_indices[i - 1] for i in range(folds)]

    for i in range(folds):
        train_mask = torch.ones(len(dataset), dtype=torch.bool)
        train_mask[test_indices[i]] = 0
        train_mask[val_indices[i]] = 0
        train_indices.append(train_mask.nonzero().view(-1))

    return train_indices, test_indices, val_indices

def print_time(begin):
    fin_time = time.time()
    current_time = time.strftime("%m/%d %H:%M", time.localtime(fin_time))
    duration = time.strftime("%dd %H:%M:%S", time.gmtime(fin_time - begin))
    return "[{:s} +{:s}]".format(current_time, duration)