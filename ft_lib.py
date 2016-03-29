# -*- coding: UTF-8 -*-

import os
import random
import codecs
import sys
import json
import mxnet as mx
import numpy as np
import shutil
from sklearn.cluster import *
from scipy.spatial.distance import cdist
from scipy.stats import norm
from sklearn.decomposition import PCA
import math
import time


def d_matrix(f, c):
    d = cdist(f, c, 'euclidean')
    d = np.min(d, axis=1)
    d = np.square(d)
    return d

def softmax(z):
    return np.exp(z) / np.sum(np.exp(z))

def parse_list(lst_file):
    labels = []
    files = []
    with open(lst_file, 'r') as lstFile:
        for line in lstFile.readlines():
            tmp = line.strip('\n').split('\t')
            if len(tmp) < 3:
                break
            labels.append(int(tmp[1]))
            files.append(tmp[-1])
    return np.array(labels), files

def gen_list(image_dir, lst_file, pick_range, ratio=1.0, rec_file=None, shuffle=True):
    if not os.path.exists(lst_file):
        total = 0
        lst_content = []
        for name, label in pick_range.items():
            images = os.listdir(os.path.join(image_dir, name))
            images.sort()
            random.shuffle(images)
            images = images[:int(len(images)*ratio)]
            for i in range(len(images)):
                line = '%d\t%d\t%s\n' % (total, int(label), os.path.join(name, images[i]))
                lst_content.append(line)
                total += 1
        print '%s generated with %d total' % (lst_file, total)
        if shuffle:
            random.shuffle(lst_content)

        with codecs.open(lst_file, 'w', 'utf-8') as lstFile:
            lstFile.writelines(lst_content)

        cmd = '/home/jiaxuzhu/developer/mxnet/bin/im2rec %s %s/ %s -resize=256' % (lst_file, image_dir, rec_file)
        os.system(cmd)
        print 'rec file generated'


def get_entropy(p):
    np.hstack((p, 1-p))
    x = np.ma.log(p)
    x.filled(0.0)
    entropy = np.multiply(p, x)
    entropy = -np.sum(entropy, axis=1)
    return entropy


def cluster_feature(features, prob, truth=None):
    c_feature = []
    # y
    if truth is None:
        y = 0
    else:
        y = float(np.sum(truth == 1)) / len(truth)

    # entropy features
    entropy = np.multiply(prob, np.log(prob))
    entropy = - np.sum(entropy, axis=1)
    # print np.max(entropy)
    # 1. entropy mean
    c_feature.append(np.mean(entropy))
    # 2. entropy std
    c_feature.append(np.std(entropy))
    # 3. entropy min
    c_feature.append(np.min(entropy))
    # 4. entropy max
    c_feature.append(np.max(entropy))
    # 4. entropy max
    c_feature.append(np.sum(entropy))
    # 4. entropy max
    c_feature.append(len(entropy))

    # distance to center
    center = np.mean(features, axis=0).reshape(1, features.shape[1])
    d = cdist(features, center, 'euclidean')
    # 5. distance mean
    c_feature.append(np.mean(d))
    # 6. distance std
    c_feature.append(np.std(d))
    # 7. distance min
    c_feature.append(np.min(d))
    # 8. distance max
    c_feature.append(np.max(d))

    # pair-wise distance
    d = cdist(features, features, 'euclidean')
    # 9. distance max
    c_feature.append(np.mean(np.mean(d, axis=1)))
    # 10. distance max
    c_feature.append(np.mean(np.max(d, axis=1)))


    # low-d features
    # pca = PCA(n_components=2)
    # tmp = pca.fit_transform(features)
    # # 12. cov
    #
    # cov = pca.get_covariance()
    # cov = np.diagonal(cov)
    # cov = np.sort(cov)
    # c_feature.append((cov[-1] + cov[-2]) / np.sum(cov))
    #
    # c_feature = np.array(c_feature)
    # c_feature.reshape(1, len(c_feature))

    return y, np.array(c_feature).reshape(1, len(c_feature))

def zscore(train_x, val_x):
    mu = np.mean(train_x, axis=0)
    sigma = np.std(train_x, axis=0)
    train_x = np.divide(np.subtract(train_x, mu), sigma)
    val_x = np.divide(np.subtract(val_x, mu), sigma)
    return train_x, val_x

def cluster_rank(pc, n_cluster, features, prob, truth=None):
    c_x = None
    c_y = []
    c_set = np.unique(pc)
    truth = np.asarray(truth)
    for c in c_set:
        c_idx = np.where(pc == c)[0]
        c_features = features[c_idx, :]
        c_prob = prob[c_idx, :]
        c_truth = truth[c_idx]
        y, c_feature = cluster_feature(c_features, c_prob, c_truth)
        c_y.append(y)
        if c_x is None:
            c_x = c_feature
        else:
            c_x = np.vstack((c_x, c_feature))
    # print c_x.shape
    rank = np.argsort(-c_x[:, 4])
    return rank, c_x, c_y, c_set


def kmeans_propose(main_dir, pick_range, lst_file, method, net, image_label, image_lst, image_truth, n_round,
                   r_label=1.0, r_cluster=1, n_reduce=1, rec_file=None):
    print 'Method:\t' + str(method)
    start = time.time()

    print n_reduce, r_cluster, r_label

    lst_content = []
    total = 0
    label_count = 0
    total_t = 0
    c_y = []
    c_x = None
    select = {}
    data_iter = mx.io.ImageRecordIter(
            path_imgrec=os.path.join(main_dir, 'mxnet_recs', 'noisy_test.rec'),
            batch_size=500,
            data_shape=(3, 224, 224),
            mean_img='/home/jiaxuzhu/data/cub_noisy/mxnet_models/inception_bn/mean_224.nd',
            shuffle=False
    )
    feature_file = os.path.join(main_dir, 'mxnet_features', 'noisy_features_%d.npy' % n_round)
    if not os.path.exists(feature_file):
        extractor = net.feature_extractor()
        all_features = extractor.predict(data_iter)
        np.save(feature_file, all_features)
    else:
        all_features = np.load(feature_file)

    prob_file = os.path.join(main_dir, 'mxnet_features', 'noisy_probs_%d.npy' % n_round)
    if not os.path.exists(prob_file):
        data_iter.reset()
        all_prob = net.net.predict(data_iter)
        np.save(prob_file, all_prob)
    else:
        all_prob = np.load(prob_file)

    if n_reduce != 1:
        pca = PCA(n_components=n_reduce)
        all_features = pca.fit_transform(all_features)

    # print time.time() - start
    step = 0
    for name, label in pick_range.items():
        print name, label_count
        select[name] = []
        label_count += 1
        features = all_features[step:step+len(image_lst[name]), :]
        prob = all_prob[step:step+len(image_lst[name]), :]

        step += len(image_lst[name])
        n_sample = int(round(len(prob) * r_label))
        n_cluster = int(round(len(prob) * r_cluster))

        if method == 'agglo':
            c_method = AgglomerativeClustering(n_clusters=n_cluster)
        elif method == 'mini':
            c_method = MiniBatchKMeans(n_clusters=n_cluster, batch_size=n_cluster*3)
        clusters = c_method.fit_predict(features)
        if len(np.unique(clusters)) < n_cluster:
            print name, len(np.unique(clusters)), n_cluster
        rank, tmp_x, tmp_y, c_set = cluster_rank(clusters, n_cluster, features, prob, image_truth[name])

        c_y.extend(tmp_y)
        if c_x is None:
            c_x = tmp_x
        else:
            c_x = np.vstack((c_x, tmp_x))
        n_label = 0
        # if name == '林书豪'.decode('utf8'):
        #     if os.path.exists(os.path.join(main_dir, 'sample_cluster')):
        #         shutil.rmtree(os.path.join(main_dir, 'sample_cluster'))
        #     os.mkdir(os.path.join(main_dir, 'sample_cluster'))
        #     for i in range(len(c_set)):
        #         c_idx = np.where(clusters == c_set[i])[0]
        #         os.mkdir(os.path.join(main_dir, 'sample_cluster', '%.2f_%d' % (tmp_x[i, 4], c_set[i])))
        #         for j in c_idx:
        #             src = os.path.join(main_dir, 'noisy_train', name, image_lst[name][j])
        #             dst = os.path.join(main_dir, 'sample_cluster', '%.2f_%d' % (tmp_x[i, 4], c_set[i]))
        #             shutil.copy2(src, dst)
	print n_cluster, n_sample
        for i in range(len(c_set)):

            print tmp_x[rank[i], 4]
            c_label = c_set[rank[i]]
            c_idx = np.where(clusters == c_label)[0]
            pick_bit = True
            if n_label < n_sample:
                need_label = True
                for j in c_idx:
                    if image_label[name][j] == 1:
                        if image_truth[name][j] == 0:
                            pick_bit = False
                        need_label = False
			print 'already labeled'

                if need_label:
                    pick = np.random.randint(len(c_idx))
                    image_label[name][c_idx[pick]] = 1
                    if image_truth[name][c_idx[pick]] == 0:
                        pick_bit = False
                    n_label += 1
            if not pick_bit:
		print 'discarded'
            if pick_bit:
                for j in range(len(c_idx)):
                    if image_truth[name][c_idx[j]] == 1:
                        total_t += 1
                    line = '%d\t%d\t%s\n' % (total, label, os.path.join(name, image_lst[name][c_idx[j]]))
                    lst_content.append(line)
                    total += 1

        random.shuffle(lst_content)
        with codecs.open(lst_file, 'w', 'utf-8') as lstFile:
            lstFile.writelines(lst_content)

    c_y = np.array(c_y).reshape(len(c_y), 1)
    np.save(os.path.join(main_dir, 'mxnet_features', '%s_%.2f_%.2f_%.1f.npy' % (method, r_label, r_cluster, n_reduce)),
            np.hstack((c_y, c_x)))
    tmp = np.hstack((c_y, 1-c_y))
    print np.mean(np.min(tmp, axis=1))
    print float(np.sum(c_y > 0.6897)) / len(c_y)
    print np.mean(c_x[:, 0])

    print "%f percent %d clean data" % (total_t / float(len(lst_content)), len(lst_content))

    cmd = '/home/jiaxuzhu/developer/mxnet/bin/im2rec %s %s/ %s' % \
          (lst_file, os.path.join(main_dir, 'noisy_train'), rec_file)
    os.system(cmd)
    print time.time() - start
    return image_label
