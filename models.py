import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow.keras.layers import (
    Input,
    Concatenate,
    concatenate,
    Add,
    Conv1D,
    Lambda,
    GlobalMaxPooling1D,
    MaxPool1D,
    Dense
)
from layers import GConv, ConcatAdj

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
        x = GConv(32, dropout=True)([x, A_in])
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


# def Dense_GCN(cfg, levels=7):
#     F = cfg['n_node_features']
#     num_classes = cfg['num_classes']
#
#     X_in = Input(shape=(F,), name='X_in')
#     A_in = Input(shape=(None,), sparse=True)
#
#     skips = []
#     x, a = X_in, A_in
#
#     x = GConv(32)([x, a])
#     skips.append(x)
#
#     for i in range(levels - 1):
#         x = GConv(32, True)([x, a])
#         x = Concatenate()([x, skips[i]])
#         skips.append(x)
#
#     skips.pop()
#     x = GConv(32)([x, a])
#     x = Concatenate()([x, *skips])
#
#     output = GConv(num_classes, activation='softmax', kernel_init='glorot_uniform')([x, A_in])
#
#     model = Model(inputs=[X_in, A_in], outputs=output, name='Dense_GCN')
#
#     return model

def adj_concat(a1, a2):
    M, N = a1.shape[0], a2.shape[0]
    new_inds = tf.concat((a1.indices, tf.add(a2.indices, M)), 0)
    new_vals = tf.concat((a1.values, a2.values), -1)
    a_out = tf.sparse.SparseTensor(indices=new_inds,
                                   values=new_vals, dense_shape=(M + N, M + N))
    return a_out


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


## Dense GCN with vertex-wise concatenation for X and block-diagonal concatenation for A
def Dense_GCN_v2(cfg, levels=3):

    F = cfg['n_node_features']
    num_classes = cfg['num_classes']

    X_in = Input(shape=(F,), name='X_in', batch_size=1)
    A_in = Input(shape=(None,), name='A_in', sparse=True, batch_size=1)

    x_skips = []
    a_skips = []
    x, a = X_in, A_in

    x = GConv(32)([x, a])
    x_skips.append(x)
    a_skips.append(a)

    for i in range(levels):
        x = GConv(32, True)([x, a])
        x = Concatenate(axis=0)([x, x_skips[i]])
        x_skips.append(x)
        a = ConcatAdj()(a, a_skips[i])
        a_skips.append(a)

    x = GConv(32)([x, a])
    x = Concatenate(axis=0)([x, *x_skips])

    for j in range(len(a_skips)):
        a = ConcatAdj()(a, a_skips[j])

    x = GConv(32)([x, a])

    mp_size = int(3 * 2 ** levels - 1)

    x = MaxPool1D(pool_size=mp_size, data_format='channels_last')(tf.expand_dims(x, 0))
    ## pending test against larger volumes
    x = Dense(256, activation='softmax')(x)
    x = Dense(128, activation='softmax')(x)
    x = Dense(64, activation='softmax')(x)
    output = Dense(num_classes, activation='softmax')(x)

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
