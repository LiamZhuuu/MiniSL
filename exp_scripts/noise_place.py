__author__ = 'Jiaxu Zhu'

import os
import numpy as np
import codecs
from utils import *
from sklearn.neighbors import KernelDensity
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import sys

def update_d(c, feature):
    d = cdist(feature, feature[c, :])
    d = np.square(np.min(d, axis=1))
    d = d / sum(d)
    return d

def pick_kde(pick, feature, u, n_idx, p_idx, method, theta):
    pick = list(pick)
    n_data = feature.shape[0]
    kde = KernelDensity(bandwidth=1)
    kde.fit(feature[pick, :])

    if method == 'density':
        p = kde.score_samples(feature[n_idx, :])
        # p = np.where(p > theta, p, 0)
        return np.sum(p)
        # return  np.sum(q[pick])
        # return np.sum(np.max(S[:, pick], axis=1))


def sim_matrix(feature, k):
    d = cdist(feature, feature)
    nn = np.sort(d, axis=1)
    tuning = nn[:, k/2:k/2+1]
    tuning = np.dot(tuning, tuning.T)
    sim = np.exp(-np.divide(np.square(d), tuning))
    return sim

# def script(sub_dir=, image_dir, ratio, method, files):
def script(ratio, method, files):
    prefix = '%s_placement_%.1f' % (method, ratio)
    print prefix
    lines = []
    total = 0

    c_count = 0
    data = parse_data(**files)

    total_feature = np.load(files['feature_file'])
    pca = PCA(n_components=30)
    pca.fit(total_feature)


    for key, item in data.items():
        # if key != '115':
        #     continue
        c_count += 1

        feature = item['feature']
        prob = item['prob']
        feature = pca.fit_transform(feature)

        kde = KernelDensity(bandwidth=5)
        kde.fit(feature)
        p = kde.score_samples(feature)
        theta = np.mean(p)

        truth = np.asarray(item['truth'])
        n_data = feature.shape[0]

        print key, c_count, n_data
        n_noise = np.sum(truth == 0)
        n_sample = int(round(n_noise * ratio))
        n_idx = np.where(truth == 0)[0]
        p_idx = np.where(truth == 1)[0]
        n_feature = feature[n_idx, :]
        p_feature = feature[p_idx, :]
        W = sim_matrix(feature, 12)
        # np.fill_diagonal(W, 0)
        # D = np.diag(1.0 / np.sqrt(np.sum(W, axis=1)))
        # S = np.dot(D, np.dot(W, D))
        u = get_entropy(prob)

        start = n_idx[np.argmax(u[n_idx])]
        # print u[start]

        pick_idx = set([start])
        rest = set(n_idx)
        rest.remove(start)

        # print n_data, n_sample
        for i in range(n_sample):
            f_value = -np.ones(n_data) * sys.maxint
            # f_base = pick_score(pick_idx, W, u, n_idx, p_idx, method)
            for j in rest:
                pick_idx.add(j)
                f_value[j] = pick_kde(pick_idx, feature, u, n_idx, p_idx, method, theta)
                pick_idx.remove(j)
            max_idx = np.argmax(f_value)
            # print u[max_idx]
            pick_idx.add(max_idx)
            rest.remove(max_idx)
        pick_idx = list(pick_idx)

        draw_graph(truth, prob[:, int(key)], W, k=12, A=pick_idx)
            # print n_density
        #
        #
        # pick_idx = np.concatenate((pick_idx, p_idx))
        # for i in pick_idx:
        #     line = '%d\t%s\t%s\n' % (total, key, item['image'][i])
        #     lines.append(line)
        #     total += 1
        # break

    # if method == 'random':
    #     gen_recs(sub_dir, image_dir, prefix, lines, refresh=True)
    # else:
    #     gen_recs(sub_dir, image_dir, prefix, lines, refresh=True)
    return prefix

if __name__ == '__main__':
    data_dir = '/Users/jiaxuzhu/Dropbox/projects/faces_x'
    files = dict(
        lst_file=os.path.join(data_dir, 'noisy_train_disturb.lst'),
        feature_file = os.path.join(data_dir, 'noisy_train_disturb_feature.npy'),
        prob_file = os.path.join(data_dir, 'noisy_train_disturb_prob.npy')
    )

    prefix = script(ratio=0.1, method='density', files=files)
