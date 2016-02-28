# -*- coding: UTF-8 -*-

import mxnet as mx
import numpy as np
import os
import logging
import math


class XEntropy(mx.metric.EvalMetric):
    def __init__(self):
         super(XEntropy, self).__init__('XEntropy')

    def update(self, labels, preds):
        # mx.metric.check_label_shapes(labels, preds)
        for label, pred in zip(labels, preds):
            label = label.asnumpy()
            pred = pred.asnumpy()
            # if len(label.shape) == 1:
            #     label = label.reshape(label.shape[0], 1)
            # mx.metric.check_label_shapes(label, pred, shape=1)
            for i in range(len(label)):
                self.sum_metric += -math.log(pred[i][label[i]])
            self.num_inst += len(label)
    
class simpleNet:
    def __init__(self, model_prefix=None, model_iter=None, n_epoch=10, mean_image=None, ctx=mx.gpu()):
        if model_prefix is not None:
            symnol, arg_params, aux_params = mx.model.load_checkpoint(model_prefix, model_iter)
            self.net = mx.model.FeedForward(symnol,
                                            arg_params=arg_params, aux_params=aux_params,
                                            ctx=ctx, num_epoch=n_epoch)
        self.mean_image = mean_image
        self.data_iters = {}

    def load_from(self, former_net, n_class, n_epoch, ctx=mx.gpu()):
        internals = former_net.net.symbol.get_internals()
        self.mean_image = former_net.mean_image
        symbol = internals['flatten_output']
        symbol = mx.symbol.FullyConnected(data=symbol, name='fc_88', num_hidden=n_class)
        symbol = mx.symbol.SoftmaxOutput(data=symbol, name='softmax')
        self.net = mx.model.FeedForward(symbol=symbol, num_epoch=n_epoch, arg_params=former_net.net.arg_params,
                                        ctx=ctx, aux_params=former_net.net.aux_params, allow_extra_params=True)
        self.data_iters = {}

    def load_data(self, data_dir, rec_name, b_size=128, is_train=False):
        rec_file = os.path.join(data_dir, rec_name)
        if not os.path.exists(rec_file):
            logging.info('Record File doesn\'t exists.')
            return
        self.data_iters[rec_name] = mx.io.ImageRecordIter(
            path_imgrec=rec_file,
            batch_size=b_size,
            data_shape=(3, 224, 224),
            prefetch_buffer=1,
            mean_img=self.mean_image,
            rand_crop=is_train,
            rand_mirror=is_train,
            shuffle=True,
        )

    def train(self, train_data, val_data, dst_prefix, step, l_rate=0.01):

        if not train_data in self.data_iters:
            logging.log('Training Data Unloaded.')
            return

        if not val_data in self.data_iters:
            logging.log('Validation Data Unloaded.')
            return
        self.net.optimizer = mx.optimizer.SGD(learning_rate=l_rate, momentum=0.9)
        self.net.fit(self.data_iters[train_data], eval_data=self.data_iters[val_data], eval_metric='acc',
                     batch_end_callback=mx.callback.Speedometer(self.data_iters[train_data].batch_size, 50),
                     epoch_end_callback=mx.callback.do_checkpoint(dst_prefix))

    def predict(self, predict_data):
        if not predict_data in self.data_iters:
            logging.info('Predicting Data Unloaded.')
            return

        prob = self.net.predict(self.data_iters[predict_data])
        return prob

    def test_acc(self, test_data):
        self.data_iters[test_data].reset()
        return self.net.score(self.data_iters[test_data])

    def feature_extractor(self, ctx=mx.gpu()):
        internals = self.net.symbol.get_internals()
        symbol = internals['flatten_output']
        extractor = mx.model.FeedForward(ctx=ctx, symbol=symbol, num_epoch=1, arg_params=self.net.arg_params,
                                         aux_params=self.net.aux_params, allow_extra_params=True)
        return extractor

