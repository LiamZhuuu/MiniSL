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
            for root, dirs, files in os.walk(os.path.join(image_dir, name)):
                images = files
                break
            images.sort()
            random.shuffle(images)
            images = images[:int(len(images)*ratio)]
            for i in range(len(images)):
                line = '%d\t%d\t%s\n' % (total, label, os.path.join(name, images[i]))
                lst_content.append(line)
                total += 1
        print '%s generated with %d total' % (lst_file, total)
        if shuffle:
            random.shuffle(lst_content)

        with codecs.open(lst_file, 'w', 'utf-8') as lstFile:
            lstFile.writelines(lst_content)
    if rec_file is not None and not os.path.exists(rec_file):
        cmd = '/home/jiaxuzhu/im2rec %s %s/ %s' % (lst_file, image_dir, rec_file)
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
    for c in c_set:
        c_idx = np.where(pc == c)[0]
        c_features = features[c_idx, :]
        c_prob = prob[c_idx, :]
        if truth is not None:
            c_truth = truth[c_idx]
        else:
            c_truth = None
        y, c_feature = cluster_feature(c_features, c_prob, c_truth)
        c_y.append(y)
        if c_x is None:
            c_x = c_feature
        else:
            c_x = np.vstack((c_x, c_feature))
    # print c_x.shape
    rank = np.argsort(c_x[:, 0])
    return rank, c_x, c_y

def cal_label(method, entropy, n_sample):
    n_cluster = len(entropy)
    if method == 'uniform':
        if n_sample == n_cluster:
            return np.ones(n_cluster).astype(int)
        elif n_sample < n_cluster:
            tmp = np.zeros(n_cluster).astype(int)
            idx = int(float(n_cluster - n_sample) / 2)
            idx = int(idx * 1.3)
            rank = np.argsort(entropy)
            # print entropy[rank[idx]]
            for i in range(idx, idx+n_sample):
                tmp[rank[i]] = 1
            return tmp
    elif method == 'combine':
        tmp = np.zeros(n_cluster).astype(int)
        rank = np.argsort(entropy)
        for idx in range(len(entropy)):
            if entropy[rank[idx]] > 2:
                break
        for i in range(idx, idx + n_sample):
            # print entropy[rank[i]]
            tmp[rank[i]] = 1
        return tmp

def kmeans_propose(main_dir, pick_range, lst_file, method, r_label=1.0, r_cluster=1, rec_file=None, refresh=True):
    ranks = {}
    print 'Method:\t' + str(method)

    image_dir = os.path.join(main_dir, 'cluster_%s_%.2f' % (method, r_label))
    # if os.path.exists(image_dir):
    #     shutil.rmtree(image_dir)
    # os.mkdir(image_dir)

    labels, files = parse_list(os.path.join(main_dir, 'mxnet_lists', 'noisy_train.lst'))
    feature_file = os.path.join(main_dir, 'mxnet_features', 'noisy_train_feature.npy')
    prob_file = os.path.join(main_dir, 'mxnet_features', 'noisy_train_prob.npy')

    all_features = np.load(feature_file)
    n_reduce = 30
    print n_reduce, r_cluster, r_label
    # if n_reduce != 1:
    #     pca = PCA(n_components=n_reduce)
    #     all_features = pca.fit_transform(all_features)
        # cov = pca.get_covariance()
        # cov = np.diagonal(cov)

    all_prob = np.load(prob_file)
    if len(labels) == len(all_features) and len(all_features) == len(all_prob):
        print 'data checked'
    if not os.path.exists(lst_file) or refresh:
        lst_content = []
        total = 0
        label_count = 0
        total_t = 0
        c_y = []
        c_x = None
        for name, label in pick_range.items():
            # print name, label_count
            label_count += 1
            idx = np.where(labels == label)[0]
            image_lst = []
            truth = []
            for i in idx:
                image_lst.append(files[i].decode('utf8'))
                truth.append(int(files[i].split('.')[-2].split('_')[-1]))
            truth = np.array(truth)
            features = all_features[idx, :]

            prob = all_prob[idx, :]
            entropy = np.multiply(prob, np.log(prob))
            entropy = -np.sum(entropy, axis=1)

            n_sample = int(round(len(prob) * r_label))
            n_cluster = int(round(len(prob) * r_cluster))

            # c_method = KMeans(n_clusters=n_cluster)
            # c_method = AgglomerativeClustering(n_clusters=n_cluster)
            # c_method = SpectralClustering(n_clusters=n_cluster)
            c_method = MiniBatchKMeans(n_clusters=n_cluster, batch_size=50, init_size=max(n_cluster, 150))
            # c_method = Birch(threshold=0.2, n_clusters=n_cluster)
            clusters = c_method.fit_predict(features)
            if len(np.unique(clusters)) < n_cluster:
                print name, len(np.unique(clusters)), n_cluster
            # os.mkdir(os.path.join(image_dir, name))
            rank, tmp_x, tmp_y = cluster_rank(clusters, n_cluster, features, prob, truth)
            c_y.extend(tmp_y)
            if c_x is None:
                c_x = tmp_x
            else:
                c_x = np.vstack((c_x, tmp_x))
            select = []
            #
            # if type(method) is float:
            #     for c in range(len(rank)):
            #         c_idx = np.where(clusters == c)[0]
            #
            #         if tmp_x[c, 0] < method:
            #             for i in range(len(c_idx)):
            #                 if truth[c_idx[i]] == 1:
            #                     total_t += 1
            #                 line = '%d\t%d\t%s\n' % (total, label, image_lst[c_idx[i]])
            #                 lst_content.append(line)
            #                 total += 1
            #                 select.append(c_idx[i])
            #                 # print t_count / len(c_idx)
            #     ranks[name] = select
            #
            # else:
            #     n_label = cal_label(method, tmp_x[:, 0], n_sample)
            #     # print np.sum(n_label), n_sample
            #     discard = False
            #     for c in range(len(rank)):
            #         c_label = rank[c]
            #         c_idx = np.where(clusters == c_label)[0]
            #
            #         p_count = 0
            #         n_count = 0
            #
            #         # tmp = np.argsort(-entropy[c_idx])
            #         for i in range(n_label[c_label]):
            #             discard = True
            #             pick = np.random.randint(len(c_idx))
            #             # pick = tmp[i]
            #             if truth[c_idx[pick]] == 1:
            #                 p_count += 1
            #             else:
            #                 n_count += 1
            #
            #         if p_count > n_count or not discard:
            #             for i in range(len(c_idx)):
            #
            #                 if truth[c_idx[i]] == 1:
            #                     total_t += 1
            #                 line = '%d\t%d\t%s\n' % (total, label, image_lst[c_idx[i]])
            #                 lst_content.append(line)
            #                 total += 1
            #                 select.append(c_idx[i])
            #                 # print t_count / len(c_idx)
            #     ranks[name] = select


        # c_total = len(c_y)
        # c_y = np.array(c_y)
        #
        # rand_idx = np.random.permutation(c_total)
        # c_train_x = c_x[rand_idx[:int(c_total*0.9)], :]
        # c_train_y = c_y[rand_idx[:int(c_total*0.9)]]
        #
        # c_val_x = c_x[rand_idx[int(c_total*0.9):], :]
        # c_val_y = c_y[rand_idx[int(c_total*0.9):]]
        #
        # c_train_x, c_val_x = zscore(c_train_x, c_val_x)
        #
        # clf = linear_model.Lasso(alpha=0.05)
        # clf.fit(c_train_x, c_train_y)
        # print clf.coef_
        # py = clf.predict(c_val_x)
        # print np.mean(np.square(py - c_val_y))
        # print py[:20]
        # print c_val_y[:20]

        random.shuffle(lst_content)
        with codecs.open(lst_file, 'w', 'utf-8') as lstFile:
            lstFile.writelines(lst_content)
    c_y = np.array(c_y).reshape(len(c_y), 1)
    np.save(os.path.join(main_dir, 'mxnet_features', 'cFeatures_%.2f_%.2f_%.1f.npy' % (r_label, r_cluster, n_reduce)),
            np.hstack((c_y, c_x)))
    tmp = np.hstack((c_y, 1-c_y))
    print np.mean(np.min(tmp, axis=1))
    print float(np.sum(c_y > 0.6897)) / len(c_y)
    print np.mean(c_x[:, 0])

    print "%f percent clean data" % (total_t / float(len(lst_content)))
    if rec_file is not None and (refresh or not os.path.exists(rec_file)):
        cmd = '/home/jiaxuzhu/developer/mxnet/bin/im2rec %s %s/ %s' %\
              (lst_file, os.path.join(main_dir, 'noisy_images_train'), rec_file)
        os.system(cmd)
    return ranks, total
