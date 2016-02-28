import os
import random
import codecs
import sys
import json
import mxnet as mx
import numpy as np
import shutil
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from scipy.stats import rv_discrete
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
from sklearn import linear_model


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
        cmd = '/home/jiaxuzhu/developer/mxnet/bin/im2rec %s %s/ %s' % (lst_file, image_dir, rec_file)
        os.system(cmd)
        print 'rec file generated'


def data_rank(prob, features, truth, label, method, n_sample):
    p = np.hstack((prob[:, label:label+1], 1 - prob[:, label:label+1]))
    x = np.ma.log(p)
    x.filled(0.0)
    entropy = np.multiply(p, x)
    # entropy = np.multiply(prob, np.log(prob))
    entropy = -np.sum(entropy, axis=1)
    if method == 'random':
        tmp = np.random.permutation(len(prob))
        rank = []
        for i in range(len(prob)):
            if truth[tmp[i]] == 1:
                rank.append(tmp[i])
            if i >= n_sample and len(rank) > 0:
                break
        n_label = i

    elif method == 'maxE':
        tmp = np.argsort(-entropy)
        rank = []
        for i in range(len(prob)):
            if truth[tmp[i]] == 1:
                rank.append(tmp[i])
            # print truth[tmp[i]], entropy[tmp[i]], prob[tmp[i], label]
            if i >= n_sample and len(rank) > 0:
                break
        n_label = i

    elif method == 'biasE':
        delta = 0.5 - prob[:, label]
        # delta = -delta
        delta = np.where(delta > 0, delta, 0)
        tmp = np.argsort(-(entropy + 0.05 * delta))
        rank = []
        negative = []
        for i in range(len(prob)):
            if truth[tmp[i]] == 1:
                rank.append(tmp[i])
            else:
                negative.append(tmp[i])
            print truth[tmp[i]], entropy[tmp[i]], prob[tmp[i], label]
            if i >= n_sample and len(rank) > 0:
                break
        n_label = i

    elif method == 'sparse':
        delta = prob[:, label] - 0.5
        tmp = np.argsort(-(entropy + 0 * delta))
        rank = []
        count = 0
        centers = None
        d = np.ones(len(prob))
        n_pass = 0
        for i in range(len(prob)):
            if d[tmp[i]] < 0.3:
                n_pass += 1
                continue
            if count == 0:
                centers = features[tmp[i]:tmp[i]+1, :]
            else:
                centers = np.vstack((centers, features[tmp[i]:tmp[i]+1, :]))
            d = d_matrix(features, centers)
            d = d / np.max(d)

            # print truth[tmp[i]], entropy[tmp[i]], prob[tmp[i], label]
            if truth[tmp[i]] == 1:
                rank.append(tmp[i])
            count += 1
            if count >= n_sample and len(rank) > 0:
                break
        n_label = n_pass

    return rank, n_label

def find_neighbors(main_dir, pick_range):
    labels, files = parse_list(os.path.join(main_dir, 'mxnet_lists', 'noisy_train.lst'))
    feature_file = os.path.join(main_dir, 'mxnet_features', 'noisy_train_feature.npy')
    for name, label in pick_range.items():
        pass

def data_propose(main_dir, pick_range, lst_file, method, ratio=1.0, rec_file=None, refresh=True):
    ranks = {}
    print 'Method:\t' + method

    # image_dir = os.path.join(main_dir, '%s_%.2f' % (method, ratio))
    # if os.path.exists(image_dir):
    #     shutil.rmtree(image_dir)
    # os.mkdir(image_dir)

    labels, files = parse_list(os.path.join(main_dir, 'mxnet_lists', 'noisy_train.lst'))
    feature_file = os.path.join(main_dir, 'mxnet_features', 'noisy_train_feature.npy')
    prob_file = os.path.join(main_dir, 'mxnet_features', 'noisy_train_prob.npy')

    all_features = np.load(feature_file)
    all_prob = np.load(prob_file)
    if len(labels) == len(all_features) and len(all_features) == len(all_prob):
        print 'data checked'

    if not os.path.exists(lst_file) or refresh:
        lst_content = []
        total = 0
        label_count = 0
        total_label = 0

        for name, label in pick_range.items():
            print name, label_count
            label_count += 1
            idx = np.where(labels == label)[0]
            image_lst = []
            truth = []
            for i in idx:
                image_lst.append(files[i].decode('utf8'))
                truth.append(int(files[i].split('.')[-2].split('_')[-1]))
            features = all_features[idx, :]

            prob = all_prob[idx, :]

            if ratio > 1:
                n_sample = ratio
            else:
                n_sample = int(round(len(prob) * ratio))
            if method == 'global':
                idx = np.where(labels != label)[0]
                adversary = all_features[idx, :]
                d = cdist(features, adversary, 'euclidean')
                d = np.min(d, axis=1)
                tmp = np.argsort(d)
                rank = []
                for i in range(len(prob)):
                    if truth[tmp[i]] == 1:
                        rank.append(tmp[i])
                    if i >= n_sample and len(rank) > 0:
                        break
                n_label = i
            else:
                rank, n_label = data_rank(prob, features, truth, label, method, n_sample)
            total_label += n_label
            # print np.sum(np.array(truth) == 1).astype('float') / len(truth)
            for i in range(len(rank)):
                line = '%d\t%d\t%s\n' % (total, label, image_lst[rank[i]])
                lst_content.append(line)
                total += 1
            ranks[name] = rank

            # os.mkdir(os.path.join(image_dir, name))
            # for idx in rank:
            #     src = os.path.join(main_dir, 'noisy_images_train', image_lst[idx])
            #     dst = os.path.join(image_dir, image_lst[idx])
            #     shutil.copy2(src, dst)

        random.shuffle(lst_content)
        with codecs.open(lst_file, 'w', 'utf-8') as lstFile:
            lstFile.writelines(lst_content)
    print "%f percent clean data and total %d" % (float(len(lst_content)) / total_label, total_label)
    if rec_file is not None and (refresh or not os.path.exists(rec_file)):
        cmd = '/home/jiaxuzhu/developer/mxnet/bin/im2rec %s %s/ %s' %\
              (lst_file, os.path.join(main_dir, 'noisy_images_train'), rec_file)
        os.system(cmd)
    return ranks, total

def noisy_propose(main_dir, pick_range, lst_file, ranks, ratio, k, rec_file=None, refresh=True):
    label_count = 0
    lst_content = []
    total = 0
    p_count = 0
    image_dir = os.path.join(main_dir, 'noise_%.2f_%.2f' % (ratio, k))
    # if os.path.exists(image_dir):
    #     shutil.rmtree(image_dir)
    # os.mkdir(image_dir)


    noisy_labels, noisy_files = parse_list(os.path.join(main_dir, 'mxnet_lists', 'noisy_train.lst'))
    feature_file = os.path.join(main_dir, 'mxnet_features', 'noisy_train_feature_2.npy')
    features_noisy = np.load(feature_file)

    feature_file = os.path.join(main_dir, 'mxnet_features', 'noisy_train_feature.npy')
    features_old = np.load(feature_file)

    for name, label in pick_range.items():
        label_count += 1
        idx = np.where(noisy_labels == label)[0]
        features = features_noisy[idx, :]
        features_0 = features_old[idx, :]
        clean_data = features[ranks[name], :]
        clean_data_0 = features_0[ranks[name], :]
        # print name, label
        # idx = np.arange(len(features_noisy))
        # features = features_noisy
        # image_lst = noisy_files
        image_lst = []
        for i in idx:
            image_lst.append(noisy_files[i])

        if len(clean_data) == 0:
            print name
        d = cdist(features, clean_data, 'euclidean')
        d_0 = cdist(features_0, clean_data_0, 'euclidean')

        d_idx = np.argmin(d, axis=1)
        d = np.min(d, axis=1)


        d_0 = np.min(d_0, axis=1)

        # os.mkdir(os.path.join(image_dir, name))
        n_sample = int(round(len(features) * k))

        bin_count = np.zeros(len(ranks[name]))

        # score = np.abs(d - d_0) / d_0
        score = -d
        tmp = np.argsort(-score)

        for i in range(n_sample):
            line = '%d\t%d\t%s/%s\n' % (total, label, 'noisy_images_train', image_lst[tmp[i]].decode('utf8'))
            truth = int(image_lst[tmp[i]].split('.')[-2].split('_')[-1])
            if truth == 1:
                p_count += 1
            # src = os.path.join(main_dir, 'noisy_images_train', image_lst[tmp[i]])
            # dst = os.path.join(image_dir, image_lst[tmp[i]])
            # shutil.copy2(src, dst)
            lst_content.append(line)
            total += 1

        random.shuffle(lst_content)
        with codecs.open(lst_file, 'w', 'utf-8') as lstFile:
            lstFile.writelines(lst_content)

    print float(p_count) / total
    if rec_file is not None and (refresh or not os.path.exists(rec_file)):
        cmd = '/home/jiaxuzhu/developer/mxnet/bin/im2rec %s %s/ %s' %\
              (lst_file, main_dir, rec_file)
        os.system(cmd)

def get_entropy(p):
    np.hstack((p, 1-p))
    x = np.ma.log(p)
    x.filled(0.0)
    entropy = np.multiply(p, x)
    entropy = -np.sum(entropy, axis=1)
    return entropy

def cluster_feature(features, prob, truth):
    c_feature = []
    # y
    y = float(np.sum(truth == 1)) / len(truth)

    # entropy features
    entropy = np.multiply(prob, np.log(prob))
    entropy = - np.sum(entropy, axis=1)
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

    return y, c_feature

def zscore(train_x, val_x):
    mu = np.mean(train_x, axis=0)
    sigma = np.std(train_x, axis=0)
    print np.where(sigma == 0)[0]
    train_x = np.divide(np.subtract(train_x, mu), sigma)
    val_x = np.divide(np.subtract(val_x, mu), sigma)
    return train_x, val_x

def cluster_rank(pc, n_cluster, features, prob, truth):
    c_x = None
    c_y = []
    for c in range(n_cluster):
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
    rank = np.argsort(c_x[:, 0])
    return rank, c_x, c_y


def kmeans_propose(main_dir, pick_range, lst_file, method, ratio=1.0, k=1, rec_file=None, refresh=True):
    ranks = {}
    print 'Method:\t' + method

    image_dir = os.path.join(main_dir, 'cluster_%s_%.2f' % (method, ratio))
    # if os.path.exists(image_dir):
    #     shutil.rmtree(image_dir)
    # os.mkdir(image_dir)

    labels, files = parse_list(os.path.join(main_dir, 'mxnet_lists', 'noisy_train.lst'))
    feature_file = os.path.join(main_dir, 'mxnet_features', 'noisy_train_feature.npy')
    prob_file = os.path.join(main_dir, 'mxnet_features', 'noisy_train_prob.npy')

    all_features = np.load(feature_file)
    n_reduce = 30
    print n_reduce, k
    if n_reduce != 1:
        pca = PCA(n_components=n_reduce)
        all_features = pca.fit_transform(all_features)
        # cov = pca.get_covariance()
        # cov = np.diagonal(cov)

    all_prob = np.load(prob_file)
    if len(labels) == len(all_features) and len(all_features) == len(all_prob):
        print 'data checked'
    clean_var = []
    if not os.path.exists(lst_file) or refresh:
        lst_content = []
        total = 0
        label_count = 0
        total_t = 0
        c_y = []
        c_x = None
        for name, label in pick_range.items():
            print name, label_count
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

            if ratio > 1:
                n_sample = ratio
            else:
                n_sample = int(round(len(prob) * ratio))
            n_cluster = n_sample * k

            kmeans = KMeans(n_clusters=n_cluster)
            kmeans.fit(features)
            clusters = kmeans.predict(features)

            # os.mkdir(os.path.join(image_dir, name))
            rank, tmp_x, tmp_y = cluster_rank(clusters, n_cluster, features, prob, truth)
            c_y.extend(tmp_y)
            if c_x is None:
                c_x = tmp_x
            else:
                c_x = np.vstack((c_x, tmp_x))
            select = []
            if k == 1:
                for c in rank:
                    c_idx = np.where(clusters == c)[0]
                    pick = np.random.randint(len(c_idx))
                    if truth[c_idx[pick]] == 1:
                        t_count = 1.0
                        for i in range(len(c_idx)):
                            if truth[c_idx[i]] == 1:
                                total_t += 1
                                t_count += 1
                            line = '%d\t%d\t%s\n' % (total, label, image_lst[c_idx[i]])
                            lst_content.append(line)
                            total += 1
                            select.append(c_idx[i])
                        # print t_count / len(c_idx)
                ranks[name] = select
            else:
                step = int((n_cluster - n_sample)/2)
                for i in range(len(rank)):
                    c = rank[i]
                    if i >= step + n_sample:
                        continue
                    c_idx = np.where(clusters == c)[0]
                    if i < step:
                        for j in range(len(c_idx)):
                            if truth[c_idx[j]] == 1:
                                total_t += 1
                            line = '%d\t%d\t%s\n' % (total, label, image_lst[c_idx[j]])
                            lst_content.append(line)
                            total += 1
                            select.append(c_idx[j])
                    else:
                        pick = np.random.randint(len(c_idx))
                        if truth[c_idx[pick]] == 1:
                            for j in range(len(c_idx)):
                                if truth[c_idx[j]] == 1:
                                    total_t += 1
                                line = '%d\t%d\t%s\n' % (total, label, image_lst[c_idx[j]])
                                lst_content.append(line)
                                total += 1
                                select.append(c_idx[j])
                        # print t_count / len(c_idx)
                ranks[name] = select

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
        # clf = linear_model.Lasso(alpha=0.1)
        # clf.fit(c_train_x, c_train_y)
        # print clf.coef_
        # py = clf.predict(c_val_x)
        # print np.mean(np.square(py - c_val_y))
        # print py[:20]
        # print c_val_y[:20]

        random.shuffle(lst_content)
        with codecs.open(lst_file, 'w', 'utf-8') as lstFile:
            lstFile.writelines(lst_content)

    np.save(os.path.join(main_dir, 'mxnet_features', 'clean_var_%.2f_%d_%.1f.npy' % (ratio, k, n_reduce)), c_x)
    print np.mean(c_x[:, 0])

    print "%f percent clean data" % (total_t / float(len(lst_content)))
    if rec_file is not None and (refresh or not os.path.exists(rec_file)):
        cmd = '/home/jiaxuzhu/developer/mxnet/bin/im2rec %s %s/ %s' %\
              (lst_file, os.path.join(main_dir, 'noisy_images_train'), rec_file)
        os.system(cmd)
    return ranks, total
