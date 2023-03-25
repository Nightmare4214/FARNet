import os
import random

import numpy as np
import torch


def get_markposion_fromtxt(point_num, path):
    flag = 0
    x_pos = []
    y_pos = []
    with open(path) as note:
        for line in note:
            if flag >= point_num:
                break
            else:
                flag += 1
                x, y = [float(i) for i in line.split(',')]
                x_pos.append(x)
                y_pos.append(y)
        x_pos = np.array(x_pos)
        y_pos = np.array(y_pos)
    return x_pos, y_pos


def get_prepoint_from_htmp(heatmaps, scal_ratio_w, scal_ratio_h):
    pred = np.zeros((19, 2))
    for i in range(19):
        heatmap = heatmaps[i]
        pre_y, pre_x = np.where(heatmap == np.max(heatmap))
        pred[i][1] = pre_y[0] * scal_ratio_h
        pred[i][0] = pre_x[0] * scal_ratio_w
    return pred


def setup_seed(seed):
    """
    set random seed

    :param seed: seed num
    """
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"  # LSTM(cuda>10.2)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.use_deterministic_algorithms(True, warn_only=True)
    # torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
