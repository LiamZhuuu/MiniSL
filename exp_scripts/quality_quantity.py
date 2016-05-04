__author__ = 'Jiaxu Zhu'

import os
import numpy as np
import codecs
from utils import *


def script(sub_dir, image_dir, data, ratio):
    prefix = 'quality_quantity_%.1f' % ratio
    lines = []
    total = 0
    for key, item in data.items():
        u = get_entropy(item['prob'])
        q_idx = np.argsort(u)
        n_sample = int(round(len(u) * ratio))
        for i in range(n_sample):
            line = '%d\t%s\t%s\n' % (total, key, item['image'][q_idx[i]])
            lines.append(line)
            total += 1
    gen_recs(sub_dir, image_dir, prefix, lines)
    return prefix
