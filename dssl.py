import os
import codecs
import numpy as np
from scipy.spatial.distance import pdist, cdist
from sklearn.semi_supervised import LabelSpreading, LabelPropagation
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.cluster import *
from sklearn.manifold import *
from sklearn.decomposition import PCA
from sklearn.manifold import MDS, TSNE

from utils import *




def graph_laplacian(sim_mat, normed=True):
    n_nodes = sim_mat.shape[0]
    # lap = -np.asarray(sim_mat)  # minus sign leads to a copy
    #
    # # set diagonal to zero
    # lap.flat[::n_nodes + 1] = 0
    # w = -lap.sum(axis=0)
    # if normed:
    #     w = np.sqrt(w)
    #     w_zeros = (w == 0)
    #     w[w_zeros] = 1
    #     lap /= w
    #     lap /= w[:, np.newaxis]
    #     lap.flat[::n_nodes + 1] = (1 - w_zeros).astype(lap.dtype)
    # else:
    #     lap.flat[::n_nodes + 1] = w.astype(lap.dtype)
    #
    # return lap

    tmp = np.copy(sim_mat)
    # tmp.flat[::n_nodes + 1] = 0
    d = np.divide(1.0, np.sqrt(np.sum(tmp, axis=1)))
    d = np.diag(d)
    lap = np.dot(d, np.dot(tmp, d))
    return lap

def label_propagation(graph_matrix, y, pre_prob, alpha=0.2):

    classes = np.arange(0, pre_prob.shape[1])
    classes = (classes[classes != -1])

    n_samples, n_classes = len(y), len(classes)

    y = np.asarray(y)
    unlabeled = y == -1
    label = y != 0
    clamp_weights = np.ones((n_samples, 2))
    clamp_weights *= alpha
    # clamp_weights[:, 1] *= 2
    # initialize distributions

    # label_distributions_ = pre_prob
    # label_distributions_ = np.zeros((n_samples, n_classes))
    # for label in classes:
    #     label_distributions_[y == label, classes == label] = 1
    #     label_distributions_[y == label, classes != label] = 0

    # beta = 0.5
    # label_distributions_ = beta * label_distributions_ + (1 - beta) * pre_prob

    y_static = np.copy(y).reshape(n_samples, 1)
    # if alpha > 0.:
    #     y_static *= 1 - alpha
    # y_static[unlabeled] = 0

    # max_iter = 10
    # for iter in range(max_iter):
    #     label_distributions_ = np.dot(
    #         graph_matrix, label_distributions_)
        # clamp
        # label_distributions_ = np.multiply(
        #     clamp_weights, label_distributions_)
        # label_distributions_ += y_static

    tmp = np.eye(n_samples) - alpha * graph_matrix
    label_distributions_ = np.dot(np.linalg.inv(tmp), y_static)
    # normalizer = np.sum(label_distributions_, axis=1)[:, np.newaxis]
    # label_distributions_ /= normalizer


    # label_distributions_ /= np.max(label_distributions_)
    # beta = 0.65
    # label_distributions_ = beta * label_distributions_ + (1 - beta) * pre_prob
    # transduction = classes[np.argmax(label_distributions_, axis=1)]
    # return transduction.ravel()
    return label_distributions_

def argmax_F(w, u, V, A):
    if len(A) > 0:
        tmp = w[:, list(A)]
        tmp = np.max(tmp, axis=1)
    else:
        tmp = np.zeros(len(u))
    f_value = np.zeros(len(u))
    for v in V:
        # f_value[v] = np.dot(np.maximum(tmp, w[:, v]), u)
        f_value[v] = np.sum(np.maximum(tmp, w[:, v]))
        # f_value[v] = u[v]
    return np.argmax(f_value)


def baseline(prob, key):
    # plt.hist(prob, bins=np.arange(0, 1.05, 0.05))
    # plt.show()
    py = np.argmax(prob, axis=1)
    for i in range(len(py)):
        if py[i] == key:
            py[i] = 1
        else:
            py[i] = 0
    return py


if __name__ == '__main__':
    data_dir = '/Users/jiaxuzhu/Dropbox/TuSimple/dssl/'
    lst_file = os.path.join(data_dir, 'noisy_train.lst')
    feature_file = os.path.join(data_dir, 'noisy_train_feature.npy')
    prob_file = os.path.join(data_dir, 'noisy_train_prob.npy')
    data = parse_data(lst_file, feature_file, prob_file)

    gamma = -1
    r_sample = 0.03
    t_precision = 0
    t_recall = 0
    t_acc = 0
    n_class = 0
    k = 12
    features = None
    probs = None
    truths = None
    c_count = 0
    idx_lst = None
    for key, item in data.items():
        c_count += 1
        # if key != '100':
        #     continue

        n_data = len(item['feature'])
        n_sample = int(round(n_data * r_sample))
        print key, n_data, n_sample

        # threshold baseline
        # py = baseline(item['prob'], int(key))

        # print score(py, item['truth'])

        W = similarity_matrix(item['feature'], k, full=True)

        # S = graph_laplacian(W)

        # bi_prob = item['prob'][:, int(key):int(key) + 1]
        # bi_prob = np.hstack((1 - bi_prob, bi_prob))
        #
        u = get_entropy(item['prob'])
        # g = get_gradient(W, u)


        # draw_graph(item['truth'], u, W, k, py)

        V = set(range(0, n_data))
        A = set()

        for i in range(n_sample):
            pick = argmax_F(W, u, V, A)
            A.add(pick)
            V.remove(pick)
        A = list(A)

        if features is None:
            features = item['feature'][A, :]
        else:
            features = np.vstack((features, item['feature'][A, :]))
        if truths is None:
            truths = np.asarray(item['truth'])[A]
        else:
            truths = np.hstack((truths, np.asarray(item['truth'])[A]))
        if idx_lst is None:
            idx_lst = np.ones(len(A)) * c_count
        else:
            idx_lst = np.hstack((idx_lst, np.ones(len(A)) * c_count))


        #
        # y = np.zeros(n_data)
        #
        # y[A] = np.asarray(item['truth'])[A] - item['prob'][A, int(key)]
        # print np.asarray(item['truth'])[A], item['prob'][A, int(key)], y[A]
        # lap = graph_laplacian(W, normed=True)
        #
        # delta_p = label_propagation(lap, y, bi_prob)
        # tmp = np.copy(item['prob'])
        # tmp[:, int(key)] += delta_p.ravel() * 100
        #
        # py = baseline(tmp, int(key))
        # draw_graph(item['truth'], tmp[:, int(key)], W, k, py, A)
        #
        # print score(py, item['truth'])
        # acc = score(py, item['truth'])
        # t_acc += acc
        # n_class += 1
        # break
    # print t_acc / n_class

    n_data = len(features)
    print n_data
    # pca = PCA(n_components=30)
    # features = pca.fit_transform(features[idx, :])
    # W = similarity_matrix(features, k, full=False)
    # draw_graph(truths, idx_lst, W, k)
    mds = TSNE(n_components=2)
    p = mds.fit_transform(features)
    print 'projected'
    idx = np.where(truths == 1)
    plt.scatter(p[idx, 0], p[idx, 1], c='b', s=50)
    idx = np.where(truths == 0)
    plt.scatter(p[idx, 0], p[idx, 1], c='r', s=50)
    plt.show()