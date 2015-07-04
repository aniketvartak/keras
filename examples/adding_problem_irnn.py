from __future__ import absolute_import
from __future__ import print_function
import numpy as np
np.random.seed(1337) # for reproducibility

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.initializations import normal, identity
from keras.layers.recurrent import SimpleRNN, LSTM
from keras.optimizers import RMSprop
from keras.utils import np_utils

'''
    This is a reproduction of the IRNN experiment
    for the toy adding problem described
    by Quoc V. Le, Navdeep Jaitly, Geoffrey E. Hinton

    arXiv:1504.00941v2 [cs.NE] 7 Apr 201
    http://arxiv.org/pdf/1504.00941v2.pdf

    Optimizer is replaced with RMSprop which give more stable and steady
    improvement. (??)

'''
