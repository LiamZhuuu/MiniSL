import os
import ujson
from ft_lib import *
import mxnet as mx
from simpleNet import simpleNet
import logging
import copy
import imagehash
from PIL import Image
import time
import math

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

main_dir = '/home/jiaxuzhu/data/tieba_1000/faces_x'
sub_dir = {
    'clean_val': 'clean_val',
    'clean_train': 'clean_train',
    'mxnet_lsts': 'mxnet_lists',
    'mxnet_recs': 'mxnet_recs',
    'mxnet_models': 'mxnet_models',
    'mxnet_logs': 'mxnet_logs',
    'mxnet_features': 'mxnet_features',
    'noisy_train': 'noisy_train'
}

for key, value in sub_dir.items():
    sub_dir[key] = os.path.join(main_dir, value)

for key, value in sub_dir.items():
    if not os.path.exists(value):
        os.mkdir(value)

with open(os.path.join(main_dir, 'name_lst.json'), 'r') as nameLst:
    name_lst = json.load(nameLst)

init_data = {
    'clean_val': 'clean_val',
    'clean_train': 'clean_train',
    'noisy_train': 'noisy_train'
}

for data in init_data.keys():
    lst_file = os.path.join(sub_dir['mxnet_lsts'], '%s.lst' % data)
    rec_file = os.path.join(sub_dir['mxnet_recs'], '%s.rec' % data)
    if not os.path.exists(rec_file):
        gen_list(sub_dir[init_data[data]], lst_file, name_lst,
                 rec_file=rec_file)

image_truth = {}
lst_file = os.path.join(sub_dir['mxnet_lsts'], 'noisy_test.lst')
rec_file = os.path.join(sub_dir['mxnet_recs'], 'noisy_test.rec')
with codecs.open(os.path.join(main_dir, 'image_truth.json'), 'r', 'utf8') as jsonFile:
    image_truth = ujson.load(jsonFile)

with codecs.open(os.path.join(main_dir, 'image_lst.json'), 'r', 'utf8') as jsonFile:
    image_lst = ujson.load(jsonFile)

if not os.path.exists(rec_file):
    total = 0
    lst_content = []
    for name, label in name_lst.items():
        print name
        name_dir = os.path.join(sub_dir['noisy_train'], name)
        images = image_lst[name]
        for image in images:
            line = '%d\t%d\t%s\n' % (total, label, os.path.join(name, image))
            lst_content.append(line)
            total += 1

    with codecs.open(lst_file, 'w', 'utf8') as lstFile:
        lstFile.writelines(lst_content)
    cmd = '/home/jiaxuzhu/developer/mxnet/bin/im2rec %s %s/ %s' % \
            (lst_file, os.path.join(main_dir, 'noisy_train'), rec_file)
    os.system(cmd)




print 'Initial list and record file done'
n_class = 200
b_size = 128

init_prefix = os.path.join(sub_dir['mxnet_models'], 'noisy_init', 'noisy_init')
if not os.path.exists(os.path.join(sub_dir['mxnet_models'], 'noisy_init')):
    os.mkdir(os.path.join(sub_dir['mxnet_models'], 'noisy_init'))

mean_img = '/home/jiaxuzhu/data/cub_noisy/mxnet_models/inception_bn/mean_224.nd'

if not os.path.exists(init_prefix + '-symbol.json'):
    inception_prefix = '/home/jiaxuzhu/data/cub_noisy/mxnet_models/inception_bn/Inception_BN'
    inception_iter = 39
    inception_net = simpleNet(inception_prefix, inception_iter, mean_image=mean_img)
    init_net = simpleNet()
    init_net.load_from(inception_net, n_class, 20, ctx=[mx.gpu(0), mx.gpu(1), mx.gpu(2), mx.gpu(3)])
    init_net.load_data(sub_dir['mxnet_recs'], 'noisy_train.rec', b_size, shape=224)
    init_net.load_data(sub_dir['mxnet_recs'], 'clean_val.rec', b_size, shape=224)
    log_file = os.path.join(sub_dir['mxnet_logs'], 'noisy_training.log')
    init_net.train('noisy_train.rec', 'clean_val.rec', init_prefix, l_rate=0.02)

init_iter = 14

# init_net = simpleNet(init_prefix, init_iter, mean_image=mean_img)
# init_net.load_data(sub_dir['mxnet_recs'], 'clean_val.rec')
# print init_net.test_acc('clean_val.rec')

r_label = 0.01
r_cluster = 0.05
n_reduce = 1
n_round = 2

method = 'mini'
is_total = False
test = False
# test = True
init_epoch = 10

ctx = [mx.gpu(0), mx.gpu(1), mx.gpu(2), mx.gpu(3)]
b_size = 128

if is_total:
    init_net = simpleNet(init_prefix, init_iter, n_epoch=10, mean_image=mean_img, ctx=ctx)
    clean_prefix = 'total_clean'
    init_net.load_data(sub_dir['mxnet_recs'], 'clean_train.rec', b_size, shape=224)
    init_net.load_data(sub_dir['mxnet_recs'], 'clean_val.rec', b_size, shape=224)

    if not os.path.exists(os.path.join(sub_dir['mxnet_models'], clean_prefix)):
        os.mkdir(os.path.join(sub_dir['mxnet_models'], clean_prefix))

    print init_net.test_acc('clean_val.rec')
    init_net.train('clean_train.rec', 'clean_val.rec',
                   os.path.join(sub_dir['mxnet_models'], clean_prefix, clean_prefix), l_rate=1e-4)

    init_net.data_iters['clean_val.rec'].reset()
    print init_net.test_acc('clean_val.rec')
else:

    if test:
        clean_prefix = 'cluster_%s_%s_%s' % (method, str(r_label), str(r_cluster))
        init_net = simpleNet(os.path.join(sub_dir['mxnet_models'], clean_prefix, clean_prefix), 9,
                             n_epoch=init_epoch, mean_image=mean_img, ctx=ctx)
        init_net.load_data(sub_dir['mxnet_recs'], 'clean_val.rec', b_size, shape=224)
        print init_net.test_acc('clean_val.rec')

    else:
        clean_prefix = '%s_%s_%s_%d' % (method, str(r_label), str(r_cluster), n_round)
        if n_round > 1:
            init_prefix = '%s_%s_%s_%d' % (method, str(r_label), str(r_cluster), n_round-1)
	    init_prefix = os.path.join(sub_dir['mxnet_models'], init_prefix, init_prefix)
            init_iter = 10

        label_file = os.path.join(sub_dir['mxnet_features'], '%s_label.json' % clean_prefix)
        if not os.path.exists(label_file):
            image_label = {}
            for name, label in name_lst.items():
                image_label[name] = [-1] * len(image_lst[name])
            with codecs.open(label_file, 'w', 'utf8') as jsonFile:
                ujson.dump(image_label, jsonFile)
        with codecs.open(label_file, 'r', 'utf8') as jsonFile:
                image_label = ujson.load(jsonFile)

        lst_file = os.path.join(sub_dir['mxnet_lsts'], '%s.lst' % clean_prefix)
        rec_file = os.path.join(sub_dir['mxnet_recs'], '%s.rec' % clean_prefix)
        init_net = simpleNet(init_prefix, init_iter, n_epoch=10, mean_image=mean_img, ctx=ctx)
        init_net.load_data(sub_dir['mxnet_recs'], 'clean_val.rec', b_size, shape=224)
        # print init_net.test_acc('clean_val.rec')

        image_label = kmeans_propose(main_dir, name_lst, lst_file, method=method, r_label=r_label, r_cluster=r_cluster,
                                rec_file=rec_file, net=copy.copy(init_net), n_round=n_round, n_reduce=n_reduce,
                                image_label=image_label, image_truth=image_truth, image_lst=image_lst)
	with codecs.open(os.path.join(sub_dir['mxnet_features'], '%s_%s_%s_%d_label.json' % (method, str(r_label), str(r_cluster), n_round+1)), 'w', 'utf8') as labelFile:
	    ujson.dump(image_label, labelFile)  
        init_net.load_data(sub_dir['mxnet_recs'], '%s.rec' % clean_prefix, b_size, shape=224)

        if not os.path.exists(os.path.join(sub_dir['mxnet_models'], clean_prefix)):
            os.mkdir(os.path.join(sub_dir['mxnet_models'], clean_prefix))

        # init_net.train('%s.rec' % clean_prefix, 'clean_val.rec', os.path.join(sub_dir['mxnet_models'], clean_prefix, clean_prefix), l_rate=1e-4)

        # init_net.net.ctx = [mx.gpu(3)]
        # print init_net.test_acc('clean_val.rec')

