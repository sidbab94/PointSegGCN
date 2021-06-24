import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow.keras.layers import (
    Input,
    Concatenate,
    concatenate,
    Add,
    Conv1D,
    Lambda
)
from layers import GConv

from tensorflow.keras import backend as K, Model
from pointnet.tf_util import conv1d_bn
from pointnet import pointnet_base


def Concat_GCN_nat(tr_params):
    F = tr_params['n_node_features']
    num_classes = tr_params['num_classes']

    X_in = Input(shape=(F,), name='X_in')
    A_in = Input(shape=(None,), sparse=True)

    levels = 4

    skips = []

    x = GConv(32)([X_in, A_in])
    X_1 = x

    for i in range(levels):
        x = GConv(32)([x, A_in])
        skips.append(x)

    skips = reversed(skips)

    for skip in skips:
        x = GConv(32)([x, A_in])
        x = Concatenate()([x, skip])

    x = Concatenate()([x, X_1])

    output = GConv(num_classes, activation='softmax', kernel_init='glorot_uniform')([x, A_in])
    model = Model(inputs=[X_in, A_in], outputs=output, name='Res_GCN_v7')
    return model


def Res_GCN(tr_params, levels=7):
    F = tr_params['n_node_features']
    num_classes = tr_params['num_classes']

    X_in = Input(shape=(F,), name='X_in')
    A_in = Input(shape=(None,), sparse=True)

    skips = []
    x, a = X_in, A_in
    x = GConv(32)([x, a])
    skips.append(x)

    for i in range(levels - 1):
        x = GConv(32)([x, a])
        x = Add()([x, skips[i]])
        skips.append(x)

    x = GConv(32)([x, a])
    x = Concatenate()([x, *skips])

    output = GConv(num_classes, activation='softmax', kernel_init='glorot_uniform')([x, A_in])

    model = Model(inputs=[X_in, A_in], outputs=output, name='Res_GCN_v7')

    return model


def Dense_GCN(cfg, levels=7):
    F = cfg['n_node_features']
    num_classes = cfg['num_classes']

    X_in = Input(shape=(F,), name='X_in')
    A_in = Input(shape=(None,), sparse=True)

    skips = []
    x, a = X_in, A_in

    x = GConv(32)([x, a])
    skips.append(x)

    for i in range(levels - 1):
        x = GConv(32, True)([x, a])
        x = Concatenate()([x, skips[i]])
        skips.append(x)

    # skips.pop()
    x = GConv(32)([x, a])
    x = Concatenate()([x, *skips])

    output = GConv(num_classes, activation='softmax', kernel_init='glorot_uniform')([x, A_in])

    model = Model(inputs=[X_in, A_in], outputs=output, name='Dense_GCN')

    return model


def point_net(cfg):

    F = cfg['n_node_features']
    num_classes = cfg['num_classes']

    X_in = Input(shape=(F,), name='X_in')

    net, _ = pointnet_base.get_model(tf.expand_dims(X_in, 1))

    net = tf.keras.layers.Dense(num_classes, activation='softmax')(net)

    model = Model(inputs=X_in, outputs=net, name='pointnet_seg')

    return model


def PointGCN(cfg, levels=3):

    F = cfg['n_node_features']
    num_classes = cfg['num_classes']

    X_in = Input(shape=(F,), name='X_in')
    A_in = Input(shape=(None,), sparse=True)

    net, _ = pointnet_base.get_model(tf.expand_dims(X_in, 1))

    skips = []
    x, a = X_in, A_in

    x = GConv(32)([net, a])
    skips.append(x)

    for i in range(levels - 1):
        x = GConv(32, True)([x, a])
        x = Concatenate()([x, skips[i]])
        skips.append(x)

    skips.pop()
    x = GConv(32)([x, a])
    x = Concatenate()([x, *skips])
    x = Concatenate()([x, net])

    output = GConv(num_classes, activation='softmax', kernel_init='glorot_uniform')([x, A_in])

    model = Model(inputs=[X_in, A_in], outputs=output, name='PointGCN')

    return model
