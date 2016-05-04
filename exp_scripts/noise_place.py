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

def pick_score(pick, S, u, n_idx, p_idx, method, beta):
    pick = list(pick)
    n_data = S.shape[0]
    t = np.zeros((n_data, 1))
    t[pick] = 1
    # q = np.dot(S, t)
    q = np.max(S[:, pick], axis=1)
    # q = np.log(q)

    if method == 'nDensity':
        return np.sum(q[pick])
        # return np.sum(np.max(S[:, pick], axis=1))

    if method == 'combine':
        return beta * np.min(q[pick]) + (1-beta) * (np.min(q))

    if method == 'comboEnt':
        q = np.multiply(q, u)
        beta = 0.5
        return  beta * np.min(q[pick]) + (1-beta) * (np.min(q))

    if method == 'nEntropy':
        tmp = np.multiply(q[pick], u[pick])
        return np.sum(tmp)

def pick_kde(pick, feature, u, n_idx, p_idx, method):
    pick = list(pick)
    n_data = feature.shape[0]
    kde = KernelDensity(bandwidth=20)
    kde.fit(feature[pick, :])

    if method == 'combine':
        beta = 0.5
        scores = kde.score_samples(feature)
        density = np.sum(scores[pick])
        sparsity = np.sum(scores[n_idx])
        return beta * density + (1 - beta) * sparsity
        # return  np.sum(q[pick])
        # return np.sum(np.max(S[:, pick], axis=1))

    if method == 'pDensity':
        return np.sum(q[p_idx])

    if method == 'allDensity':
        return np.sum(q)

    if method == 'nEntropy':
        tmp = np.multiply(q, u)
        return np.sum(tmp[n_idx])



def sim_matrix(feature, k):
    d = cdist(feature, feature)
    nn = np.sort(d, axis=1)
    tuning = nn[:, k/2:k/2+1]
    tuning = np.dot(tuning, tuning.T)
    sim = np.exp(-np.divide(np.square(d), tuning))
    return sim

def script(sub_dir, image_dir, data, ratio, method):
    prefix = '%s_placement_%.1f' % (method, ratio)
    print prefix
    lines = []
    total = 0

    c_count = 0

    for key, item in data.items():
        if key != '115':
            continue
        c_count += 1

        feature = item['feature']
        prob = item['prob']
        # pca = PCA(n_components=2)
        pca = TSNE(n_components=2)
        # feature = pca.fit_transform(feature)
        # print feature

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
        np.fill_diagonal(W, 0)
        D = np.diag(1.0 / np.sqrt(np.sum(W, axis=1)))
        S = np.dot(D, np.dot(W, D))
        u = get_entropy(prob)

        start = n_idx[np.argmax(u[n_idx])]
        # print u[start]

        for beta in np.arange(0, 1.1, 0.1):
            pick_idx = set([start])
            rest = set(n_idx)
            rest.remove(start)

            # print n_data, n_sample
            for i in range(n_sample):
                f_value = -np.ones(n_data) * sys.maxint
                # f_base = pick_score(pick_idx, W, u, n_idx, p_idx, method)
                for j in rest:
                    pick_idx.add(j)
                    f_value[j] = pick_score(pick_idx, W, u, n_idx, p_idx, method, beta=beta)
                    # f_value[j] = pick_kde(pick_idx, feature, u, n_idx, p_idx, method)
                    pick_idx.remove(j)
                max_idx = np.argmax(f_value)
                # print u[max_idx]
                pick_idx.add(max_idx)
                rest.remove(max_idx)
            pick_idx = list(pick_idx)

            draw_graph(truth, prob[:, int(key)], W, k=12, A=pick_idx, file_name='pick_%f.png' % beta)
            # print n_density
        #
        #
        # pick_idx = np.concatenate((pick_idx, p_idx))
        # for i in pick_idx:
        #     line = '%d\t%s\t%s\n' % (total, key, item['image'][i])
        #     lines.append(line)
        #     total += 1
        # break

   

    if method == 'random':
        gen_recs(sub_dir, image_dir, prefix, lines, refresh=True)
    else:
        gen_recs(sub_dir, image_dir, prefix, lines, refresh=True)
    return prefix
