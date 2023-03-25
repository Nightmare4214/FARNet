import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from config import Config
from data import medical_dataset
from model import Farnet
from test import get_errors
from train import train_model
from utils import setup_seed, seed_worker
from packaging import version

setup_seed(42)
g = torch.Generator()
g.manual_seed(0)

if __name__ == '__main__':
    model = Farnet()
    if version.parse(torch.__version__) >= version.parse('2.0.0'):
        model = torch.compile(model)
    model.cuda(Config.GPU)

    train_set = medical_dataset(Config.img_dir, Config.gt_dir, Config.resize_h, Config.resize_w, Config.point_num,
                                Config.sigma)
    train_loader = DataLoader(dataset=train_set, batch_size=1, shuffle=True, num_workers=1,
                              pin_memory=False,
                              # worker_init_fn=seed_worker, generator=g
                              )
    test_set1 = medical_dataset(Config.test_img_dir1, Config.test_gt_dir1, Config.resize_h, Config.resize_w,
                                Config.point_num, Config.sigma)
    test1_loader = DataLoader(dataset=test_set1, batch_size=1, shuffle=False, num_workers=1,
                              pin_memory=False,
                              # worker_init_fn=seed_worker, generator=g
                              )
    if Config.test_img_dir2 != '' and Config.test_gt_dir2 != '':
        result_dir, result_name = os.path.split(Config.save_results_path)
        save_result1 = os.path.join(result_dir, 'test1', result_name)
        save_result2 = os.path.join(result_dir, 'test2', result_name)
        test_set2 = medical_dataset(Config.test_img_dir2, Config.test_gt_dir2, Config.resize_h, Config.resize_w,
                                    Config.point_num, Config.sigma)
        test2_loader = DataLoader(dataset=test_set2, batch_size=1, shuffle=False, num_workers=1,
                                  pin_memory=False,
                                  # worker_init_fn=seed_worker, generator=g
                                  )
    else:
        save_result1 = Config.save_results_path
        save_result2 = ''
        test2_loader = None

    criterion = nn.MSELoss(reduction='none')
    criterion = criterion.cuda(Config.GPU)
    optimizer_ft = optim.Adam(model.parameters(), lr=Config.lr)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer_ft, [200], gamma=0.1, last_epoch=-1)
    model_ft = train_model(model, criterion, optimizer_ft, scheduler, train_loader, Config.num_epochs)
    os.makedirs(os.path.dirname(Config.save_model_path), exist_ok=True)
    torch.save(model_ft.state_dict(), Config.save_model_path)
    if save_result1 != '':
        os.makedirs(os.path.dirname(save_result1), exist_ok=True)
        get_errors(model, test1_loader, Config.test_gt_dir1, save_result1)

    if save_result2 != '':
        os.makedirs(os.path.dirname(save_result2), exist_ok=True)
        get_errors(model, test2_loader, Config.test_gt_dir2, save_result2)
