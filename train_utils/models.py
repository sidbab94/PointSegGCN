import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf

from tensorflow.keras.layers import Dropout, Input, Concatenate, Add
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from train_utils.layers import GConv, CRF


def conv_relu_bn(parents, filters, dropout=False, l2_reg=0.01):
    X_in, A_in = parents
    x = GCNConv(filters, activation='relu', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg))([X_in, A_in])
    # x = BatchNormalization()(x)
    if dropout:
        x = Dropout(0.2)(x)
    return x

def Concat_GCN(tr_params):

    l2_reg = tr_params['l2_reg']
    F = tr_params['n_node_features']
    num_classes = tr_params['num_classes']

    X_in = Input(shape=(F,), name='X_in')
    A_in = Input(shape=(None,), sparse=True)
    I_in = Input(shape=(), name="segment_ids_in", dtype=tf.int32)

    levels = 4

    skips = []

    x = GCNConv(32, activation='relu', kernel_regularizer=l2(l2_reg))([X_in, A_in])
    X_1 = x

    for i in range(levels):
        x = conv_relu_bn((x, A_in), 32, False)
        skips.append(x)

    skips = reversed(skips)

    for skip in skips:
        x = conv_relu_bn((x, A_in), 32, False)
        x = Concatenate()([x, skip])

    x = Concatenate()([x, X_1])

    output = GCNConv(num_classes, activation='softmax', name='gcn_6')([x, A_in])

    model = Model(inputs=[X_in, A_in, I_in], outputs=output, name='U-GCN4')
    return model

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


def Dense_GCN(tr_params, levels=7):

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
        x = Concatenate()([x, skips[i]])
        skips.append(x)

    skips.pop()
    x = GConv(32)([x, a])
    x = Concatenate()([x, *skips])

    output = GConv(num_classes, activation='softmax', kernel_init='glorot_uniform')([x, A_in])

    ## experimental CRF
    # output = CRF()(X_in, output)

    model = Model(inputs=[X_in, A_in], outputs=output, name='Dense_GCN')

    return model
