# -*- coding: UTF-8 -*-
__author__ = 'Jiaxu Zhu'

import os
import imagehash
from PIL import Image
import cv2
import numpy as np


main_dir = '/home/jiaxuzhu/data/tieba_1000/faces_x/'
sub_dir = {
    'noisy_data': 'images',
    'clean_val': 'clean_val',
    'clean_train': 'clean_train',
    'mxnet_lsts': 'mxnet_lists',
    'mxnet_recs': 'mxnet_recs',
    'mxnet_models': 'mxnet_models',
    'mxnet_logs': 'mxnet_logs',
    'mxnet_features': 'mxnet_features',
    'noisy_images_train': 'noisy_train'
}

hash_dict = dict()

# im1 = cv2.imread('/home/jiaxuzhu/data/tieba_1000/faces_x/noisy_train/乔振宇/乔振宇_00000320_1.jpg')
# im2 = cv2.imread('/home/jiaxuzhu/data/tieba_1000/faces_x/clean_val/乔振宇/2.jpg')
# im1 = im1.astype('float')
# im2 = im2.astype('float')
#
# print np.linalg.norm(im1 - im2)
# print im1 - im2

names = os.listdir(os.path.join(main_dir, sub_dir['clean_val']))
for name in names:
    images = os.listdir(os.path.join(main_dir, sub_dir['clean_val'], name))
    hash_dict[name] = []
    for image in images:
        im = os.path.join(main_dir, sub_dir['clean_val'], name, image)
        hash_dict[name].append(str(imagehash.phash(Image.open(im), hash_size=8)))

count = 0
for name in names:
    images = os.listdir(os.path.join(main_dir, sub_dir['noisy_images_train'], name))
    for image in images:
        im = os.path.join(main_dir, sub_dir['noisy_images_train'], name, image)
        hash = str(imagehash.phash(Image.open(im), hash_size=8))
        if hash in hash_dict[name]:
            count += 1
            cmd = 'rm %s' % im
            os.system(cmd)
print count