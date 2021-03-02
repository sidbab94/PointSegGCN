import tensorflow as tf

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from tensorflow.keras.layers import Dropout, BatchNormalization, Input, Concatenate, Dense, Add
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from spektral.layers import GCNConv, GlobalMaxPool, MinCutPool, GCSConv, EdgeConv, TopKPool
from train_utils.tf_utils import unPool


def conv_relu_bn(parents, filters, dropout=False, l2_reg=0.01):
    X_in, A_in = parents
    x = GCNConv(filters, activation='relu', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg))([X_in, A_in])
    # x = BatchNormalization()(x)
    if dropout:
        x = Dropout(0.2)(x)
    return x


def Res_GCN_v7(tr_params):

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

    model = Model(inputs=[X_in, A_in, I_in], outputs=output, name='Res_GCN_v7')
    return model

def topk_gcn(tr_params):

    l2_reg = tr_params['l2_reg']
    F = tr_params['n_node_features']
    num_classes = tr_params['num_classes']

    X_in = Input(shape=(F,), name='X_in')
    A_in = Input(shape=(None,), sparse=True)
    I_in = Input(shape=(), name="segment_ids_in", dtype=tf.int32)

    X_1 = conv_relu_bn((X_in, A_in), 16, l2_reg=l2_reg)
    X_2, A_2, I_2 = TopKPool(0.5, kernel_regularizer=l2(l2_reg))([X_1, A_in, I_in])

    X_3 = conv_relu_bn((X_2, A_2), 16, l2_reg=l2_reg)
    X_4, A_4, I_4 = TopKPool(0.5, kernel_regularizer=l2(l2_reg))([X_3, A_2, I_2])

    X_5 = conv_relu_bn((X_4, A_4), 16, l2_reg=l2_reg)
    X_6 = conv_relu_bn((X_5, A_4), 16, l2_reg=l2_reg)
    # X_6, A_6, I_6 = TopKPool(0.4, kernel_regularizer=l2(l2_reg))([X_5, A_4, I_4])

    concat = Concatenate(axis=0)([X_6, X_5, X_3])

    dense1 = mlp_block(concat, 64)
    dense2 = mlp_block(dense1, 32)
    output = Dense(num_classes, activation='softmax')(dense2)

    model = Model(inputs=[X_in, A_in, I_in], outputs=output, name='GraphSEG_v4')

    return model

def edgeconv(parents, l2_reg, filters=32):

    x, a = parents
    output = EdgeConv(filters, aggregate='max', activation='relu',
                      kernel_regularizer=l2(l2_reg), kernel_initializer='he_normal')([x, a])
    output = BatchNormalization()(output)

    return output

def mlp_block(input, units, do=True, bn=False):

    output = Dense(units, activation='relu', kernel_initializer='he_normal')(input)
    if do:
        output = Dropout(0.3)(output)
    if bn:
        output = BatchNormalization()(output)

    return output

def dgcnn_v1(model_cfg):

    l2_reg = model_cfg['l2_reg']
    F = model_cfg['n_node_features']
    num_classes = model_cfg['num_classes']

    X_in = Input(shape=(F,), name='X_in')
    A_in = Input(shape=(None,), sparse=True)

    X_1 = edgeconv((X_in, A_in), l2_reg)
    X_2 = edgeconv((X_1, A_in), l2_reg)
    X_3 = edgeconv((X_2, A_in), l2_reg)

    concat_1 = Concatenate()([X_1, X_2, X_3])

    mlp_1 = mlp_block(concat_1, 128, do=False, bn=True)
    mlp_2 = mlp_block(mlp_1, 128, do=False, bn=True)
    mlp_3 = mlp_block(mlp_2, 64, bn=True)
    mlp_4 = mlp_block(mlp_3, 32, bn=True)

    output = Dense(num_classes, activation='softmax')(mlp_4)

    model = Model(inputs=[X_in, A_in], outputs=output, name='EdgeConv_v1')
    return model


class Graph_U(Model):
    def __init__(self, tr_params, verbose=False):
        super(Graph_U, self).__init__()
        self.l2_reg = tr_params['l2_reg']
        self.F = tr_params['n_node_features']
        self.num_classes = tr_params['num_classes']

        self.num_levels = 3
        self.v = verbose

        self.x_agg = [None for i in range(2 * (self.num_levels))]
        self.x_pool = [None for i in range(self.num_levels)]
        self.a_pool = [None for i in range(self.num_levels)]
        self.x_unpool = [None for i in range(self.num_levels)]
        self.a_unpool = [None for i in range(self.num_levels)]
        self.s_pool = [None for i in range(self.num_levels)]

    def call(self, inputs):

        self.autoencoder(parents=inputs)

        softmax_inputs = [self.x_agg[-1], self.a_unpool[-1]]

        output = GCNConv(self.num_classes, activation='softmax')(softmax_inputs)

        return output

    def autoencoder(self, parents):

        X_in, A_in = parents
        ds_cluster = round(X_in.shape[0] * 0.5)
        if self.v:
            print('Min_cut clustering size: ', ds_cluster)
            print('X_in size: ', X_in.shape)
            print('A_in size: ', A_in.shape)
            print('-------')

        for i in range(self.num_levels):
            self.x_pool[i], self.a_pool[i], self.s_pool[i] = MinCutPool(k=ds_cluster, return_mask=True)([X_in, A_in])
            if self.v:
                print(' Downsampling level : ', i)
                print('After Pooling:')
                print('X_pool_{} size: {}'.format(i, self.x_pool[i].shape))
                print('A_pool_{} size: {}'.format(i, self.a_pool[i].shape))
                print('S_pool_{} size: {}'.format(i, self.s_pool[i].shape))
            self.x_agg[i] = self.gcn_block(inputs=(self.x_pool[i], self.a_pool[i]))
            if self.v:
                print('After Aggregation:')
                print('X_Agg_{} size: {}'.format(i, self.x_agg[i].shape))
                print('-------')
            ds_cluster = round(self.x_agg[i].shape[0] * 0.5)

        for j in range(self.num_levels):
            k = self.num_levels - j - 1
            self.x_unpool[j], self.a_unpool[j] = unPool()([self.x_agg[k], self.a_pool[k], self.s_pool[k]])
            if self.v:
                print(' Upsampling level : ', j)
                print('After Unpooling:')
                print('X_unpool_{} size: {}'.format(j, self.x_unpool[j].shape))
                print('A_unpool_{} size: {}'.format(j, self.a_unpool[j].shape))
            self.x_agg[j + self.num_levels] = self.gcn_block(inputs=(self.x_unpool[j], self.a_unpool[j]), dropout=True)
            if self.v:
                print('After Aggregation:')
                print('X_Agg_{} size: {}'.format(j + self.num_levels, self.x_agg[j + self.num_levels].shape))
                print('-------')

    def gcn_block(self, inputs, filters=32, dropout=False):
        x, a = inputs

        x = GCNConv(filters, activation='relu', kernel_regularizer=l2(self.l2_reg))([x, a])
        x = BatchNormalization()(x)
        if dropout:
            x = Dropout(0.1)(x)

        return x

