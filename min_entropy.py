import os
from utils import *
import numpy as np
import json

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


lst_file = os.path.join(sub_dir['mxnet_lsts'], 'noisy_train_disturb.lst')
feature_file = os.path.join(sub_dir['mxnet_features'], 'noisy_train_disturb_feature.npy')
prob_file = os.path.join(sub_dir['mxnet_features'], 'noisy_train_disturb_prob.npy')
data = parse_data(lst_file, feature_file, prob_file)

n_bin = 10
total_count = np.zeros(n_bin)
for key, item in data.items():
	u = get_entropy(item['prob'])
	truth = np.asarray(item['truth'])
	n_noise = np.sum(truth == 0)
	u_idx = np.argsort(-u)
	n_data = len(u)
	bin_size = int(n_data / n_bin)
	count = np.zeros(n_bin)
	for i in range(n_bin):
		count[i] = np.sum(truth[u_idx[(i)*bin_size:(i+1)*bin_size]] == 0)
		count[i] /= bin_size
	total_count += count

total_count /= len(data.items())
for i in range(n_bin):
	print total_count[i]
