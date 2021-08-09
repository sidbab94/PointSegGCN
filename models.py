import os
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import (
    Input,
    Concatenate,
    Add,
    MaxPool1D,
    Dense
)

from layers import GConv, ConcatAdj

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def PointSegGCN(cfg):
    '''
    Builds PointSegGCN model from skip connections and GCN layers
    :param cfg: Model parameters retrieved from cfg file
    :return: Built TF model, ready for forward pass
    '''
    F = cfg['n_node_features']
    num_classes = cfg['num_classes']

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
    model = Model(inputs=[X_in, A_in], outputs=output, name='PointSegGCN_v1')
    return model


def Dense_GCN(cfg, levels=3):
    '''
    Builds a Dense GCN model with vertex-wise skip connections and MLP layers
    :param cfg: Model parameters retrieved from cfg file
    :param levels: No. of hierarchichal feature extraction levels
    :return: Built TF model, ready for forward pass
    '''
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
        # Block-diagonal concatenation for adjacency matrix
        a = ConcatAdj()(a, a_skips[i])
        a_skips.append(a)

    x = GConv(32)([x, a])
    x = Concatenate(axis=0)([x, *x_skips])

    for j in range(len(a_skips)):
        a = ConcatAdj()(a, a_skips[j])

    x = GConv(32)([x, a])

    # Max pooling kernel size computation
    mp_size = int(3 * 2 ** levels - 1)

    # MLP block
    x = MaxPool1D(pool_size=mp_size, data_format='channels_last')(tf.expand_dims(x, 0))
    x = Dense(256, activation='softmax')(x)
    x = Dense(128, activation='softmax')(x)
    x = Dense(64, activation='softmax')(x)
    output = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=[X_in, A_in], outputs=output, name='Dense_GCN')

    return model
