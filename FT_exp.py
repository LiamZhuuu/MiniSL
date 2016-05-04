import sys
sys.path.insert(0, '/home/jiaxuzhu/developer/mxnet/python')
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
from utils import parse_data
from exp_scripts import quality_quantity
import codecs
import cv2

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

main_dir = '/home/jiaxuzhu/data/tieba_1000/faces_x/'
sub_dir = {
    'clean_val': 'clean_val',
    'clean_train': 'clean_train',
    'mxnet_lsts': 'mxnet_lists',
    'mxnet_recs': 'mxnet_recs',
    'mxnet_models': 'mxnet_models',
    'mxnet_logs': 'mxnet_logs',
    'mxnet_features': 'mxnet_features',
    'noisy_train': 'noisy_train',
    'noisy_train_disturb': 'noisy_train_disturb',
    'clean_train_disturb': 'clean_train_disturb',
}
for key, value in sub_dir.items():
    sub_dir[key] = os.path.join(main_dir, value)

for key, value in sub_dir.items():
    if not os.path.exists(value):
        os.mkdir(value)

with open(os.path.join(main_dir, 'name_lst.json'), 'r') as nameLst:
    name_lst = json.load(nameLst)

init_data = [
    'clean_train',
    'clean_val',
    'noisy_train',
    'noisy_train_disturb',
    'clean_train_disturb'
]

for data in init_data:
    lst_file = os.path.join(sub_dir['mxnet_lsts'], '%s.lst' % data)
    rec_file = os.path.join(sub_dir['mxnet_recs'], '%s.rec' % data)
    if not os.path.exists(lst_file) or not os.path.exists(rec_file):
        gen_list(sub_dir[data], lst_file, name_lst,
                 rec_file=rec_file)

print 'Initial list and record file done'
n_class = 181

init_prefix = os.path.join(sub_dir['mxnet_models'], 'noisy_init', 'noisy_init')
if not os.path.exists(os.path.join(sub_dir['mxnet_models'], 'noisy_init')):
    os.mkdir(os.path.join(sub_dir['mxnet_models'], 'noisy_init'))

mean_img = '/home/jiaxuzhu/data/tieba_1000/faces_x/inception-bn/mean_224.nd'

b_size = 128
ctx = [mx.gpu(0), mx.gpu(1), mx.gpu(2), mx.gpu(3)]
if not os.path.exists(init_prefix + '-symbol.json'):
    inception_prefix = '/home/jiaxuzhu/data/tieba_1000/faces_x/inception-bn/Inception_BN'
    inception_iter = 39
    bn_net = simpleNet(inception_prefix, inception_iter, n_epoch=20, mean_image=mean_img, ctx=ctx)
    init_net = simpleNet()
    init_net.load_from(bn_net, n_class, 20, ctx=ctx)
    init_net.load_data(sub_dir['mxnet_recs'], 'noisy_train_disturb.rec', b_size, shape=224)
    init_net.load_data(sub_dir['mxnet_recs'], 'clean_val.rec', b_size, shape=224)
    init_net.train('noisy_train_disturb.rec', 'clean_val.rec', init_prefix, l_rate=1e-4)

# init_net = simpleNet(init_prefix, init_iter, mean_image=mean_img)
# init_net.load_data(sub_dir['mxnet_recs'], 'clean_val_all.rec')
# print init_net.test_acc('clean_val_all.rec')


# init_data = [
#     'noisy_train_disturb',
# ]
# init_iter = 10

# for data in init_data:
#     lst_file = os.path.join(sub_dir['mxnet_lsts'], '%s.lst' % data)
#     rec_file = os.path.join(sub_dir['mxnet_recs'], '%s.rec' % data)
#     feature_file = os.path.join(sub_dir['mxnet_features'], '%s_feature.npy' % data)
#     prob_file = os.path.join(sub_dir['mxnet_features'], '%s_prob.npy' % data)

#     prefix = os.path.join(sub_dir['mxnet_models'], data, data)
#     if not os.path.exists(feature_file) or not os.path.exists(prob_file):
#         print 'extract features and probs'
#         ctx = mx.gpu()
#         init_net = simpleNet(prefix, init_iter, mean_image=mean_img, n_epoch=10, ctx=ctx)
#         extractor = init_net.feature_extractor(ctx=ctx)
#         data_iter = mx.io.ImageRecordIter(
#             path_imgrec=rec_file,
#             batch_size=200,
#             data_shape=(3, 224, 224),
#             mean_img=mean_img,
#         )
#         if not os.path.exists(feature_file):
#             features = extractor.predict(data_iter)
#             print features.shape
#             np.save(feature_file, features)

#         if not os.path.exists(prob_file):
#             data_iter.reset()
#             prob = init_net.net.predict(data_iter)
#             print prob.shape
#             np.save(prob_file, prob)


# init_net = simpleNet(init_prefix, init_iter, mean_image=mean_img, n_epoch=10, ctx=ctx)
# init_net.load_data(sub_dir['mxnet_recs'], 'noisy_train.rec', 128, shape=224)
# print init_net.test_acc('noisy_train.rec')


is_total = True
is_test = False

b_size = 128
ctx = [mx.gpu(0), mx.gpu(1), mx.gpu(2), mx.gpu(3)]

if is_total:
    inception_prefix = '/home/jiaxuzhu/data/tieba_1000/faces_x/inception-bn/Inception_BN'
    inception_iter = 39
    bn_net = simpleNet(inception_prefix, inception_iter, n_epoch=20, mean_image=mean_img, ctx=ctx)
    init_net = simpleNet()
    init_net.load_from(bn_net, n_class, 40, ctx=ctx)
    prefix = 'clean_train_disturb'

    handler = logging.FileHandler(os.path.join(sub_dir['mxnet_logs'], '%s.log' % prefix))
    logger.addHandler(handler)

    init_net.load_data(sub_dir['mxnet_recs'], '%s.rec' % prefix, b_size, shape=224)
    init_net.load_data(sub_dir['mxnet_recs'], 'clean_val.rec', b_size, shape=224)

    if not os.path.exists(os.path.join(sub_dir['mxnet_models'], prefix)):
        os.mkdir(os.path.join(sub_dir['mxnet_models'], prefix))
    init_net.train('%s.rec' % prefix, 'clean_val.rec', 
        os.path.join(sub_dir['mxnet_models'], prefix, prefix ), l_rate=1e-4)

# elif test:
#     # prefix = 'quality_quantity_0.7'
#     prefix = 'noisy_init'
#     init_net = simpleNet(os.path.join(sub_dir['mxnet_models'], prefix, prefix), 20,
#                          n_epoch=init_epoch, mean_image=mean_img, ctx=ctx)
#     init_net.load_data(sub_dir['mxnet_recs'], 'noisy_train.rec', b_size, shape=224)
#     print init_net.test_acc('noisy_train.rec')

# else:
#     lst_file = os.path.join(sub_dir['mxnet_features'], 'noisy_train.lst')
#     feature_file = os.path.join(sub_dir['mxnet_features'], 'noisy_train_feature.npy')
#     prob_file = os.path.join(sub_dir['mxnet_features'], 'noisy_train_prob.npy')
#     data = parse_data(lst_file, feature_file, prob_file)

#     prefix = quality_quantity.script(sub_dir, data, ratio=0.6)

#     init_net = simpleNet(init_prefix, 5, n_epoch=init_epoch, mean_image=mean_img,
#                          ctx=ctx)
#     init_net.load_data(sub_dir['mxnet_recs'], '%s.rec' % prefix, b_size, shape=224)
#     init_net.load_data(sub_dir['mxnet_recs'], 'clean_val.rec', b_size, shape=224)
#     if not os.path.exists(os.path.join(sub_dir['mxnet_models'], prefix)):
#         os.mkdir(os.path.join(sub_dir['mxnet_models'], prefix))
#     init_net.train('%s.rec' % prefix, 'clean_val.rec',
#                    os.path.join(sub_dir['mxnet_models'], prefix, prefix ), l_rate=1e-4)
    #
    # print init_net.test_acc('clean_val_all.rec')

