import os
from utils import *
from sklearn.neighbors import KernelDensity
from sklearn.decomposition import PCA
import numpy as np


data_dir = '/Users/jiaxuzhu/Dropbox/projects/faces_x'

lst_file = os.path.join(data_dir, 'noisy_train_disturb.lst')
feature_file = os.path.join(data_dir, 'noisy_train_disturb_feature.npy')
prob_file = os.path.join(data_dir, 'noisy_train_disturb_prob.npy')
data = parse_data(lst_file, feature_file, prob_file)

total_feature = np.load(feature_file)
pca = PCA(n_components=30)
pca.fit(total_feature)

for key, item in data.items():
    print key
    feature = pca.transform(item['feature'])
    print np.max(feature), np.min(feature)
    d = cdist(feature, feature)
    print np.mean(d)
    truth = np.asarray(item['truth'])
    n_data = feature.shape[0]

    n_noise = np.sum(truth == 0)
    n_idx = np.where(truth == 0)[0]
    p_idx = np.where(truth == 1)[0]
    n_feature = feature[n_idx, :]
    p_feature = feature[p_idx, :]
    kde = KernelDensity(bandwidth=1)
    kde.fit(n_feature)
    density = kde.score_samples(feature)

    print np.var(density)
    debug = 1
