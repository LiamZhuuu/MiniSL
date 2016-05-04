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
import random

def parse_data(lst_file, feature_file, prob_file):
    data = {}
    with codecs.open(lst_file, 'r', 'utf8') as lstFile:
        idx = 0
        for line in lstFile:
            line_data = line.strip('\n').split('\t')
            label = line_data[1]
            image = line_data[2]
            truth = int(image.split('.')[0].split('_')[-1])
            if label not in data:
                data[label] = {
                    'idx': [],
                    'truth': [],
                    'image': []
                }
            data[label]['idx'].append(idx)
            data[label]['truth'].append(truth)
            data[label]['image'].append(image)
            idx += 1
    features = np.load(feature_file)

    print features.shape
    probs = np.load(prob_file)
    print probs.shape
    for key in data.keys():
        idx = np.asarray(data[key]['idx'])
        data[key]['feature'] = features[idx, :]
        data[key]['prob'] = probs[idx, :]
    return data


def gen_recs(sub_dir, image_dir, prefix, lines, refresh=False):
    lst_path = os.path.join(sub_dir['mxnet_lsts'], '%s.lst' % prefix)
    rec_path = os.path.join(sub_dir['mxnet_recs'], '%s.rec' % prefix)
    if refresh or not os.path.exists(rec_path):
        random.shuffle(lines)
        with codecs.open(lst_path, 'w', 'utf-8') as lstFile:
            lstFile.writelines(lines)
        cmd = '/home/jiaxuzhu/developer/mxnet/bin/im2rec %s %s/ %s' % (lst_path, image_dir, rec_path)
        os.system(cmd)
        print 'rec file generated'


def draw_graph(truth, prob, w, k, py=None, A=None, file_name=None):

    g = nx.Graph()
    labels = {}
    nn = np.argsort(-w, axis=1)
    for i in range(w.shape[0]):
        g.add_node(i)
        labels[i] = '%.1f' % prob[i]
        for j in range(k):
            g.add_edge(i, nn[i][j])
    pos = nx.spring_layout(g)
    t = np.asarray(truth)
    idx = np.where(t == 0)[0]
    nx.draw_networkx_nodes(g, pos,
                           nodelist=idx.tolist(),
                           node_color='r',
                           node_size=500)
    idx = np.where(t == 1)[0]
    nx.draw_networkx_nodes(g, pos,
                           nodelist=idx.tolist(),
                           node_color='b',
                           node_size=500)
    if py is not None:
        tmp = np.where((t != py) & (t == 1))[0].tolist()
        nx.draw_networkx_nodes(g, pos,
                               nodelist=tmp,
                               node_size=500,
                               node_color='y')

        tmp = np.where((t != py) & (t == 0))[0].tolist()
        nx.draw_networkx_nodes(g, pos,
                               nodelist=tmp,
                               node_size=500,
                               node_color='g')
    if A is not None:
        nx.draw_networkx_nodes(g, pos,
                               nodelist=A,
                               node_size=500,
                               node_color='m')
    nx.draw_networkx_edges(g, pos, width=1.0, alpha=0.5)
    nx.draw_networkx_labels(g, pos, labels, font_size=12, font_color='w')
    # mpl.rc("figure", facecolor="white")
    plt.show()


def score(py, truth):
    # l = len(py)
    # return float(np.sum(py == truth)) / l
    t = np.asarray(truth)
    tp = np.where((py == 1) & (t == 1))[0]
    acc = float(np.sum(py == truth)) / len(truth)
    return acc


def similarity_matrix(feature, k, full=True):
    d = cdist(feature, feature)
    nn = np.sort(d, axis=1)
    idx = np.argsort(d, axis=1)
    tuning = nn[:, k/2:k/2+1]
    tuning = np.dot(tuning, tuning.T)
    sim = np.exp(-np.divide(np.square(d), tuning))
    if not full:
        tmp = np.zeros(sim.shape)
        for i in range(sim.shape[0]):
            for j in range(k):
                tmp[i, idx[i, j]] = sim[i, idx[i, j]]
                tmp[idx[i, j], i] = tmp[i, idx[i, j]]
        sim = np.copy(tmp)
    return sim

def get_entropy(prob):
    x = prob
    x = np.multiply(x, np.log(x))
    return -np.sum(x, axis=1)


def get_gradient(w, v):
    g = np.zeros(len(v))
    for i in range(len(v)):
        idx = w[i, :] != 0
        tmp = v[idx]
        g[i] = np.var(tmp)
    return g

def rank_order(feature, th):
    n_data = feature.shape[0]
    d = cdist(feature, feature)
    idx = np.argsort(d, axis=1)
    D = np.zeros(d.shape)
    r = np.zeros(d.shape)
    for i in range(n_data):
        for j in range(n_data):
            tmp = idx[i, j]
            for k in range(j):
                D[i, tmp] += np.where(idx[tmp, :] == idx[i, k])[0][0]
    for i in range(n_data):
        for j in range(i+1, n_data):
            o1 = np.where(idx[i, :] == j)[0][0]
            o2 = np.where(idx[j, :] == i)[0][0]
            tmp = D[i, j] + D[j, i]
            tmp /= min(o1, o2)
            if tmp < th:
                r[i, j] = 1
    return r
