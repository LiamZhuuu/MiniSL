import os
import json
from ft_lib import *
import mxnet as mx
from simpleNet import simpleNet
import logging
import imagehash
from PIL import Image
import time
import math

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

main_dir = '/home/jiaxuzhu/data/tieba_1000/faces_x/'
sub_dir = {
    'noisy_data': 'images',
    'clean_val': 'clean_images_val',
    'clean_train': 'clean_images_train',
    'mxnet_lsts': 'mxnet_lists',
    'mxnet_recs': 'mxnet_recs',
    'mxnet_models': 'mxnet_models',
    'mxnet_logs': 'mxnet_logs',
    'mxnet_features': 'mxnet_features',
    'noisy_images_train': 'noisy_images_train'
}
for key, value in sub_dir.items():
    sub_dir[key] = os.path.join(main_dir, value)

for key, value in sub_dir.items():
    if not os.path.exists(value):
        os.mkdir(value)

with open(os.path.join(main_dir, 'name_lst.json'), 'r') as nameLst:
    name_lst = json.load(nameLst)

init_data = {
    'clean_val_all': 'clean_val',
    'clean_train_all': 'clean_train',
    'noisy_train': 'noisy_images_train'
}

for data in init_data.keys():
    lst_file = os.path.join(sub_dir['mxnet_lsts'], '%s.lst' % data)
    if not os.path.exists(lst_file):
        gen_list(sub_dir[init_data[data]], lst_file, name_lst,
                 rec_file=os.path.join(sub_dir['mxnet_recs'], '%s.rec' % data))

print 'Initial list and record file done'
n_class = 181
b_size = 128

init_prefix = os.path.join(sub_dir['mxnet_models'], 'noisy_init', 'noisy_init')
if not os.path.exists(os.path.join(sub_dir['mxnet_models'], 'noisy_init')):
    os.mkdir(os.path.join(sub_dir['mxnet_models'], 'noisy_init'))

mean_img = '/home/jiaxuzhu/data/tieba_1000/faces_x/inception-bn/mean_224.nd'

if not os.path.exists(init_prefix + '-symbol.json'):
    inception_prefix = '/home/jiaxuzhu/data/tieba_1000/faces_x/inception-bn/Inception_BN'
    inception_iter = 39
    inception_net = simpleNet(inception_prefix, inception_iter, mean_image=mean_img)
    init_net = simpleNet()
    init_net.load_from(inception_net, 181, 20, ctx=[mx.gpu(0), mx.gpu(1), mx.gpu(2), mx.gpu(3)])
    init_net.load_data(sub_dir['mxnet_recs'], 'noisy_train.rec', b_size, shape=224)
    init_net.load_data(sub_dir['mxnet_recs'], 'clean_val_all.rec', b_size, shape=224)
    log_file = os.path.join(sub_dir['mxnet_logs'], 'noisy_training.log')
    init_net.train('noisy_train.rec', 'clean_val_all.rec', init_prefix, l_rate=0.02)

init_iter = 14

# init_net = simpleNet(init_prefix, init_iter, mean_image=mean_img)
# init_net.load_data(sub_dir['mxnet_recs'], 'clean_val_all.rec')
# print init_net.test_acc('clean_val_all.rec')


init_data = {
    # 'clean_val_all': 'clean_val',
    'noisy_train': 'noisy_images_train'
}


for data in init_data.keys():
    lst_file = os.path.join(sub_dir['mxnet_lsts'], '%s.lst' % data)
    rec_file = os.path.join(sub_dir['mxnet_recs'], '%s.rec' % data)
    feature_file = os.path.join(sub_dir['mxnet_features'], '%s_feature.npy' % data)
    prob_file = os.path.join(sub_dir['mxnet_features'], '%s_prob.npy' % data)

    if not os.path.exists(feature_file) or not os.path.exists(prob_file):
        ctx = mx.gpu()
        init_net = simpleNet(init_prefix, init_iter, mean_image=mean_img, n_epoch=10, ctx=ctx)
        extractor = init_net.feature_extractor(ctx=ctx)
        data_iter = mx.io.ImageRecordIter(
            path_imgrec=rec_file,
            batch_size=500,
            data_shape=(3, 224, 224),
            mean_img=mean_img,
            shuffle=False
        )
        if not os.path.exists(feature_file):
            features = extractor.predict(data_iter)
            np.save(feature_file, features)
        if not os.path.exists(prob_file):
            data_iter.reset()
            prob = init_net.net.predict(data_iter)
            np.save(prob_file, prob)


r_label = 0.2
r_cluster = 0.2

# method = 'combine'
# method = 0.15
method = 'uniform'
is_total = False
test = False
# test = True
init_epoch = 10

ctx = [mx.gpu(0), mx.gpu(1), mx.gpu(2), mx.gpu(3)]
b_size = 128

if is_total:
    init_net = simpleNet(init_prefix, init_iter, n_epoch=10, mean_image=mean_img, ctx=ctx)
    clean_prefix = 'total_clean'
    init_net.load_data(sub_dir['mxnet_recs'], 'clean_train_all.rec', b_size, shape=224)
    init_net.load_data(sub_dir['mxnet_recs'], 'clean_val_all.rec', b_size, shape=224)

    if not os.path.exists(os.path.join(sub_dir['mxnet_models'], clean_prefix)):
        os.mkdir(os.path.join(sub_dir['mxnet_models'], clean_prefix))

    print init_net.test_acc('clean_val_all.rec')
    init_net.train('clean_train_all.rec', 'clean_val_all.rec',
                   os.path.join(sub_dir['mxnet_models'], clean_prefix, clean_prefix), l_rate=1e-4)

    init_net.data_iters['clean_val_all.rec'].reset()
    print init_net.test_acc('clean_val_all.rec')
else:

    clean_prefix = 'cluster_%s_%s_%s' % (method, str(r_label), str(r_cluster))
    if test:
        init_net = simpleNet(os.path.join(sub_dir['mxnet_models'], clean_prefix, clean_prefix), 9,
                             n_epoch=init_epoch, mean_image=mean_img, ctx=ctx)
        init_net.load_data(sub_dir['mxnet_recs'], 'clean_val_all.rec', b_size, shape=224)
        print init_net.test_acc('clean_val_all.rec')
    else:
        lst_file = os.path.join(sub_dir['mxnet_lsts'], '%s.lst' % clean_prefix)

        init_net = simpleNet(init_prefix, init_iter, n_epoch=init_epoch, mean_image=mean_img,
                             ctx=ctx)
        ranks, n_total = kmeans_propose(main_dir, name_lst, lst_file, method=method, r_label=r_label, r_cluster=r_cluster,
                                        rec_file=os.path.join(sub_dir['mxnet_recs'], '%s.rec' % clean_prefix))

        # with open(os.path.join(sub_dir['mxnet_features'], 'ranks', '%s.json' % clean_prefix), 'w') as rankFile:
        #     json.dump(ranks, rankFile)

        # init_net.load_data(sub_dir['mxnet_recs'], '%s.rec' % clean_prefix, b_size, shape=224)
        # init_net.load_data(sub_dir['mxnet_recs'], 'clean_val_all.rec', b_size, shape=224)
        # if not os.path.exists(os.path.join(sub_dir['mxnet_models'], clean_prefix)):
        #     os.mkdir(os.path.join(sub_dir['mxnet_models'], clean_prefix))
        #
        # print init_net.test_acc('clean_val_all.rec')
        #
        # init_net.train('%s.rec' % clean_prefix, 'clean_val_all.rec',
        #                os.path.join(sub_dir['mxnet_models'], clean_prefix, clean_prefix), l_rate=1e-4)
        #
        # print init_net.test_acc('clean_val_all.rec')

