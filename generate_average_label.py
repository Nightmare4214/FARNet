#!/usr/bin/env python
# _*_ coding:utf-8 _*_
import os
from glob import glob

import numpy as np

HEIGHT = 512
WIDTH = 416

junior_path = '/mnt/data/datasets/ISBI2015_ceph/400_junior'
senior_path = '/mnt/data/datasets/ISBI2015_ceph/400_senior'
average_path = '/mnt/data/datasets/ISBI2015_ceph/400_average'
os.makedirs(average_path, exist_ok=True)

if __name__ == '__main__':
    junior_paths = list(glob(os.path.join(junior_path, '*')))
    junior_paths.sort()
    senior_paths = list(glob(os.path.join(senior_path, '*')))
    senior_paths.sort()

    for junior, senior in zip(junior_paths, senior_paths):
        junior_label = np.loadtxt(junior, delimiter=',', max_rows=19)
        senior_label = np.loadtxt(senior, delimiter=',', max_rows=19)
        name = os.path.basename(junior)

        annotation = (junior_label + senior_label) / 2
        np.savetxt(os.path.join(average_path, name), annotation, fmt='%.1f', delimiter=',')
