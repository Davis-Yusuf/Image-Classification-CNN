from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from lib.mlp.fully_conn import *
from lib.mlp.layer_utils import *
from lib.cnn.layer_utils import *


""" Classes """
class TestCNN(Module):
    def __init__(self, dtype=np.float32, seed=None):
        self.net = sequential(
            ########## TODO: ##########
            ConvLayer2D(input_channels=3, kernel_size=3, number_filters=3, name="conv"),
            MaxPoolingLayer(pool_size=2, stride=2, name="max_pool"),
            flatten(),
            fc(input_dim=27, output_dim=5, init_scale=0.02, name="fc")
            ########### END ###########
        )


class SmallConvolutionalNetwork(Module):
    def __init__(self, keep_prob=0, dtype=np.float32, seed=None):
        self.net = sequential(
            ########## TODO: ##########
            ConvLayer2D(input_channels=3, kernel_size=5, number_filters=18, name="conv"),
            MaxPoolingLayer(pool_size=2, stride=2, name="max_pool"),
            flatten(),
            gelu(),
            fc(input_dim=3528, output_dim=20, init_scale=0.02, name="fc1")
            # dropout(keep_prob=0.25, seed=1234)
            ########### END ###########
        )