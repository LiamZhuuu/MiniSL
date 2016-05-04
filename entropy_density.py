import sys
sys.path.insert(0, '/home/jiaxuzhu/developer/mxnet/python')
import os
import json
from ft_lib import *
import mxnet as mx
from simpleNet import simpleNet
import logging
import imagehash
from PIL import Image
import time
import math
from utils import parse_data
from exp_scripts import noise_density
import codecs
import cv2


def t_entropy(prob, key):
    t = np.zeros(prob.shape)
    z = np.zeros(prob.shape)
    py = np.argmax(prob, axis=1)
    for i in range(len(py)):
        z[i, py[i]] = 1
    t[:, int(key)] = 1
    beta = 0.1
    u = beta * z + (1 - beta) * t
    u = np.sum(np.multiply(u, np.log(prob)), axis=1)
    return -u

main_dir = '/home/jiaxuzhu/data/tieba_1000/faces_x/'
sub_dir = {
    'clean_val': 'clean_val',
    'clean_train': 'clean_train',
    'mxnet_lsts': 'mxnet_lists',
    'mxnet_recs': 'mxnet_recs',
    'mxnet_models': 'mxnet_models',
    'mxnet_logs': 'mxnet_logs',
    'mxnet_features': 'mxnet_features',
    'noisy_train': 'noisy_train',
    'noisy_train_disturb': 'noisy_train_disturb',
    'clean_train_disturb': 'clean_train_disturb',
}
for key, value in sub_dir.items():
    sub_dir[key] = os.path.join(main_dir, value)

test = True
b_size = 128
ctx = [mx.gpu(0), mx.gpu(1), mx.gpu(2), mx.gpu(3)]


lst_file = os.path.join(sub_dir['mxnet_lsts'], 'noisy_train_disturb.lst')
feature_file = os.path.join(sub_dir['mxnet_features'], 'noisy_train_disturb_feature.npy')
prob_file = os.path.join(sub_dir['mxnet_features'], 'noisy_train_disturb_prob.npy')
data = parse_data(lst_file, feature_file, prob_file)

m = 10
count = np.zeros(m)
total = 0
count = 0
for key, item in data.items():
    features = item['feature']
    truth = np.asarray(item['truth'])
    d = cdist(features, features)
    d = np.mean(d, axis=1)
    # u = get_entropy(item['prob'], key)
    py = np.argmax(item['prob'], axis=1)
    max_p = np.max(item['prob'], axis=1)
    for i in range(features.shape[0]):
        if truth[i] == 0 and (py[i] != int(key) or max_p[i] < 0.5):
            count += 1
    total += np.sum(truth == 0)
print float(count) / total
