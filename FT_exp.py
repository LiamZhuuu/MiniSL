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
    'noisy_train': 'noisy_images_train'
}

for data in init_data.keys():
    lst_file = os.path.join(sub_dir['mxnet_lsts'], '%s.lst' % data)
    if not os.path.exists(lst_file):
        gen_list(sub_dir[init_data[data]], lst_file, name_lst,
                 rec_file=os.path.join(sub_dir['mxnet_recs'], '%s.rec' % data))

print 'Initial list and record file done'
n_class = 181
b_size = 256
init_prefix = os.path.join(sub_dir['mxnet_models'], 'noisy_init', 'noisy_init')

mean_img = '/home/jiaxuzhu/data/tieba_1000/faces_x/mxnet_models/inception_bn/mean_224.nd'


if not os.path.exists(init_prefix + '-symbol.json'):
    inception_prefix = '/mnt/scratch/jxzhu/inception_bn/Inception_BN'
    inception_iter = 39
    inception_net = simpleNet(inception_prefix, inception_iter, mean_image=mean_img)
    init_net = simpleNet()
    init_net.load_from(inception_net, 181, 10)
    init_net.load_data(sub_dir['mxnet_recs'], 'noisy_train.rec', b_size)
    init_net.load_data(sub_dir['mxnet_recs'], 'clean_val_all.rec', b_size)
    log_file = os.path.join(sub_dir['mxnet_logs'], 'noisy_training.log')
    init_net.train('noisy_train.rec', 'clean_val_all.rec', init_prefix,
                   ctx=ctx, l_rate=0.001)

init_iter = 6

# init_net = simpleNet(init_prefix, init_iter, mean_image=mean_img)
# init_net.load_data(sub_dir['mxnet_recs'], 'clean_val_all.rec')
# print init_net.test_acc('clean_val_all.rec')


init_data = {
    'clean_val_all': 'clean_val',
    'noisy_train': 'noisy_images_train'
}


for data in init_data.keys():
    lst_file = os.path.join(sub_dir['mxnet_lsts'], '%s.lst' % data)
    rec_file = os.path.join(sub_dir['mxnet_recs'], '%s.rec' % data)
    feature_file = os.path.join(sub_dir['mxnet_features'], '%s_feature.npy' % data)
    prob_file = os.path.join(sub_dir['mxnet_features'], '%s_prob.npy' % data)

    if not os.path.exists(feature_file) or not os.path.exists(prob_file):
        ctx = [mx.gpu(0)]
        init_net = simpleNet(init_prefix, init_iter, n_epoch=10, mean_image=mean_img, ctx=ctx)
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

b_size = 120
ratios = [0.05, 0.1, 0.2, 0.5, 0.8, 1]

ratio = 1
method = 'random'
print ratio
k = ratio * 10
is_train = True
init_epoch = 5
clean_epoch = 5
clean_prefix = 'clean_%s_%s' % (method, str(ratio))

if is_train:

    lst_file = os.path.join(sub_dir['mxnet_lsts'], '%s.lst' % clean_prefix)

    init_net = simpleNet(init_prefix, init_iter, n_epoch=init_epoch, mean_image=mean_img, ctx=[mx.gpu()])
    # ranks, n_total = kmeans_propose(main_dir, name_lst, lst_file, method=method, ratio=ratio, k=2,
    #              rec_file=os.path.join(sub_dir['mxnet_recs'], '%s.rec' % clean_prefix))

    ranks, n_total = data_propose(main_dir, name_lst, lst_file, method=method, ratio=ratio,
                        rec_file=os.path.join(sub_dir['mxnet_recs'], '%s.rec' % clean_prefix))


    with open(os.path.join(sub_dir['mxnet_features'], 'ranks', '%s.json' % clean_prefix), 'w') as rankFile:
        json.dump(ranks, rankFile)

    step = 11 * int(math.ceil(n_total / b_size))
    init_net.load_data(sub_dir['mxnet_recs'], '%s.rec' % clean_prefix, b_size)
    init_net.load_data(sub_dir['mxnet_recs'], 'clean_val_all.rec', b_size)
    if not os.path.exists(os.path.join(sub_dir['mxnet_models'], clean_prefix)):
        os.mkdir(os.path.join(sub_dir['mxnet_models'], clean_prefix))

    init_net.train('%s.rec' % clean_prefix, 'clean_val_all.rec',
                   os.path.join(sub_dir['mxnet_models'], clean_prefix, clean_prefix), step, l_rate=5e-5)

    print init_net.test_acc('clean_val_all.rec')

else:

    with open(os.path.join(sub_dir['mxnet_features'], 'ranks', '%s.json' % clean_prefix), 'r') as rankFile:
        ranks = json.load(rankFile)
    clean_iter = init_epoch
    clean_net = simpleNet(os.path.join(sub_dir['mxnet_models'], clean_prefix, clean_prefix), clean_iter,
                          n_epoch=clean_epoch, mean_image=mean_img, ctx=[mx.gpu()])
    clean_net.load_data(sub_dir['mxnet_recs'], 'clean_val_all.rec', b_size)
    print clean_net.test_acc('clean_val_all.rec')

    extractor = clean_net.feature_extractor(ctx=mx.gpu())
    feature_file = os.path.join(main_dir, 'mxnet_features', 'noisy_train_feature_2.npy')
    if True and not os.path.exists(feature_file):
        data_iter = mx.io.ImageRecordIter(
                path_imgrec=os.path.join(sub_dir['mxnet_recs'], 'noisy_train.rec'),
                batch_size=500,
                data_shape=(3, 224, 224),
                mean_img=mean_img,
                shuffle=False
        )
        features = extractor.predict(data_iter)
        np.save(feature_file, features)

        print 'Feature Extracted'

    noisy_prefix = 'noisy_%s_%s' % (method, k)
    print noisy_prefix
    lst_file = os.path.join(sub_dir['mxnet_lsts'], '%s.lst' % noisy_prefix)

    if not os.path.exists(os.path.join(sub_dir['mxnet_models'], noisy_prefix)):
        os.mkdir(os.path.join(sub_dir['mxnet_models'], noisy_prefix))
    noisy_propose(main_dir, name_lst, lst_file, k=k, ratio=ratio, ranks=ranks,
                  rec_file=os.path.join(sub_dir['mxnet_recs'], '%s.rec' % noisy_prefix))

    clean_net.load_data(sub_dir['mxnet_recs'], '%s.rec' % noisy_prefix, b_size)
    step = 1
    clean_net.train('%s.rec' % noisy_prefix, 'clean_val_all.rec',
                    os.path.join(sub_dir['mxnet_models'], noisy_prefix, noisy_prefix), step, l_rate=5e-5)

    clean_net.data_iters['clean_val_all.rec'].reset()
    print clean_net.test_acc('clean_val_all.rec')

# if not is_train:
#     noisy_prefix = 'noisy_%s_%s' % (method, k)
#     noisy_iter = clean_epoch
#     noisy_net = simpleNet(os.path.join(sub_dir['mxnet_models'], noisy_prefix, noisy_prefix), noisy_iter,
#                          n_epoch=5, mean_image=mean_img, ctx=[mx.gpu(0), mx.gpu(1), mx.gpu(3)])
#
#     noisy_net.load_data(sub_dir['mxnet_recs'], 'clean_val_all.rec', b_size)
#     print noisy_net.test_acc('clean_val_all.rec')
