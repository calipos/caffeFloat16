#coding=utf-8
''' 选一个中间层，把前面的nobn部分截取  把后边的bn部分截取，分别生成各自的模型（注意，后边的网络部分可能存在多输入） '''
''' 把截取的两个部分手动合并到一起(把前部的convlution 换成convolutionFp16)，并且连续去加载两个部分的预训练模型，并存为新的预训练模型 '''
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



finetune_net = caffe.Net('./pelee_stage1_tb_finetune_deploy.prototxt',caffe.TEST)
finetune_net.copy_from('pelee_nobn_head.caffemodel')
finetune_net.copy_from('pelee_bn_tail.caffemodel')
finetune_net.save('pelee_stage1_tb_finetune.caffemodel')