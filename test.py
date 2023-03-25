import os

from cv2 import cv2
import numpy as np
import torch
import xlwt
from packaging import version
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import Config
from data import medical_dataset
from model import Farnet
from utils import get_markposion_fromtxt, get_prepoint_from_htmp


# Config = Config()


def get_errors(model, test_loader, note_gt_dir, save_path):
    loss = np.zeros(19)
    num_err_below_20 = np.zeros(19)
    num_err_below_25 = np.zeros(19)
    num_err_below_30 = np.zeros(19)
    num_err_below_40 = np.zeros(19)
    img_num = 0
    with tqdm(enumerate(test_loader), total=len(test_loader)) as pbar:
        for img_num, (img, heatmaps, _, img_name, _, _) in pbar:
            # print('图片', img_name[0])
            img = img.cuda(Config.GPU)
            outputs, _ = model(img)
            outputs = outputs[0].cpu().detach().numpy()
            pred = get_prepoint_from_htmp(outputs, Config.scal_w, Config.scal_h)
            note_gt_road = os.path.join(note_gt_dir, os.path.splitext(img_name[0])[0] + '.txt')
            gt_x, gt_y = get_markposion_fromtxt(19, note_gt_road)
            gt_x = np.trunc(np.reshape(gt_x, (19, 1)))
            gt_y = np.trunc(np.reshape(gt_y, (19, 1)))
            gt = np.concatenate((gt_x, gt_y), 1)
            for j in range(19):
                error = np.sqrt((gt[j][0] - pred[j][0]) ** 2 + (gt[j][1] - pred[j][1]) ** 2)
                loss[j] += error
                if error <= 20:
                    num_err_below_20[j] += 1
                elif error <= 25:
                    num_err_below_25[j] += 1
                elif error <= 30:
                    num_err_below_30[j] += 1
                elif error <= 40:
                    num_err_below_40[j] += 1

    loss = loss / (img_num + 1)
    num_err_below_25 = num_err_below_25 + num_err_below_20
    num_err_below_30 = num_err_below_30 + num_err_below_25
    num_err_below_40 = num_err_below_40 + num_err_below_30

    row0 = ['NO', '<=20', '<=25', '<=30', '<=40', 'mean_err']
    f = xlwt.Workbook()
    sheet1 = f.add_sheet('sheet1', cell_overwrite_ok=True)
    for i in range(0, len(row0)):
        sheet1.write(0, i, row0[i])
    for i in range(0, 19):
        sheet1.write(i + 1, 0, i + 1)
        sheet1.write(i + 1, 1, num_err_below_20[i] / (img_num + 1))
        sheet1.write(i + 1, 2, num_err_below_25[i] / (img_num + 1))
        sheet1.write(i + 1, 3, num_err_below_30[i] / (img_num + 1))
        sheet1.write(i + 1, 4, num_err_below_40[i] / (img_num + 1))
        sheet1.write(i + 1, 5, loss[i])
    f.save(save_path)


def predict(model, img_path):
    img = cv2.imread(img_path)
    img_h, img_w, _ = img.shape
    img_resize = cv2.resize(img, (Config.resize_w, Config.resize_h))
    img_data = np.transpose(img_resize, (2, 0, 1))
    img_data = np.reshape(img_data, (1, 3, Config.resize_h, Config.resize_w))
    img_data = torch.from_numpy(img_data).float()
    scal_ratio_w = img_w / Config.resize_w
    scal_ratio_h = img_h / Config.resize_h
    img_data = img_data.cuda(Config.GPU)
    outputs = model(img_data)
    outputs = outputs[0].cpu().detach().numpy()
    pred = get_prepoint_from_htmp(outputs, scal_ratio_w, scal_ratio_h)
    return pred


if __name__ == '__main__':
    model = Farnet()
    if version.parse(torch.__version__) >= version.parse('2.0.0'):
        model = torch.compile(model)
    model.load_state_dict(torch.load(Config.save_model_path))
    model.cuda(Config.GPU)

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

    if save_result1 != '':
        os.makedirs(os.path.dirname(save_result1), exist_ok=True)
        get_errors(model, test1_loader, Config.test_gt_dir1, save_result1)

    if save_result2 != '':
        os.makedirs(os.path.dirname(save_result2), exist_ok=True)
        get_errors(model, test2_loader, Config.test_gt_dir2, save_result2)
