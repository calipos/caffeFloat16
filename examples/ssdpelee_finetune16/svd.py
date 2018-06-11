#coding=utf-8

import os.path as osp
import sys
import copy
import os
import numpy as np
import numpy.linalg as linalg

CAFFE_ROOT = '/media/hdd/lbl_trainData/git/caffe_build_'
if osp.join(CAFFE_ROOT,'python') not in sys.path:
        sys.path.insert(0,osp.join(CAFFE_ROOT,'python'))

import caffe



caffe.set_mode_gpu()
head_net = caffe.Net('./pelee_nobn_head.prototxt', 'pelee_nobn.caffemodel',caffe.TEST)
head_net.save('pelee_nobn_head.caffemodel')
caffe.set_mode_gpu()
tail_net = caffe.Net('./pelee_bn_tail.prototxt', 'pelee_bn.caffemodel',caffe.TEST)
tail_net.save('pelee_bn_tail.caffemodel')



finetune_net = caffe.Net('./pelee_stage4_tb_finetune_deploy.prototxt',caffe.TEST)
finetune_net.copy_from('pelee_nobn_head.caffemodel')
finetune_net.copy_from('pelee_bn_tail.caffemodel')
finetune_net.save('pelee_stage4_tb_finetune.caffemodel')