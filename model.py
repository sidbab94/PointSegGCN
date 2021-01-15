import tensorflow as tf

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from tensorflow.keras.layers import Dropout, BatchNormalization, Input, Add, UpSampling1D, Lambda, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from spektral.layers import GCNConv, GlobalMaxPool, ChebConv, MinCutPool, GCSConv
from train_utils.tf_utils import unPool


def conv_block(parents, filters, id, dropout=False, l2_reg=0.01):
    if len(parents) > 2:
        X_in, A_in, I_in = parents
    else:
        X_in, A_in = parents
    x = GCNConv(filters, activation='relu', kernel_regularizer=l2(l2_reg), name='gcn_' + str(id))([X_in, A_in])
    x = GCNConv(filters, activation='relu', kernel_regularizer=l2(l2_reg), name='gcn_' + str(id+1))([x, A_in])

    x = BatchNormalization(name='bn_' + str(id))(x)
    if dropout:
        x = Dropout(0.1, name='do_' + str(id))(x)
    if len(parents) > 2:
        x = GlobalMaxPool(name='mp_' + str(id))([x, I_in])
    output = x
    return output


def conv_relu_bn(parents, filters, dropout=False, l2_reg=0.01):
    X_in, A_in = parents
    x = GCNConv(filters, activation='relu', kernel_regularizer=l2(l2_reg))([X_in, A_in])
    x = BatchNormalization()(x)
    if dropout:
        x = Dropout(0.1)(x)
    return x


def gcn_block(filters, do=False):

    result = tf.keras.Sequential()
    result.add(
        GCNConv(filters, activation='relu', kernel_regularizer=l2(0.01)),
    )
    result.add(BatchNormalization())
    if do:
        result.add(Dropout(0.1))

    return result

def Res_GCN_v1(model_cfg):

    l2_reg = model_cfg['l2_reg']
    F = model_cfg['n_node_features']
    num_classes = model_cfg['num_classes']

    X_in = Input(shape=(F,), name='X_in')
    A_in = Input(shape=(None,), sparse=True)

    levels = 2

    skips = []

    x = GCNConv(32, activation='relu', kernel_regularizer=l2(l2_reg))([X_in, A_in])
    X_1 = x

    for i in range(levels):

        x = conv_relu_bn((x, A_in), 32, True)
        skips.append(x)

    skips = reversed(skips[:-1])

    for skip in skips:

        x = conv_relu_bn((x, A_in), 32, True)
        x = Concatenate()([x, skip])

    x = Concatenate()([x, X_1])

    output = GCNConv(num_classes, activation='softmax', name='gcn_6')([x, A_in])

    model = Model(inputs=[X_in, A_in], outputs=output, name='GraphSEG_v2')
    return model


def res_model_2(tr_params):

    l2_reg = tr_params['l2_reg']
    F = tr_params['n_node_features']
    num_classes = tr_params['num_classes']

    X_in = Input(shape=(F,), name='X_in')
    A_in = Input(shape=(None,), sparse=True)

    X_1 = GCNConv(64, activation='relu', kernel_regularizer=l2(l2_reg), name='gcn_1')([X_in, A_in])

    X_2 = conv_block((X_1, A_in), 64, 2, l2_reg=l2_reg)

    X_3 = conv_block((X_2, A_in), 64, 3, l2_reg=l2_reg)

    X_4 = conv_block((X_3, A_in), 64, 4, dropout=True, l2_reg=l2_reg)

    X_5 = conv_block((X_4, A_in), 64, 5, dropout=True, l2_reg=l2_reg)
    X_6 = Add(name='add_5_3')([X_5, X_3])

    X_7 = conv_block((X_6, A_in), 64, 6, dropout=True, l2_reg=l2_reg)
    X_8 = Add(name='add_7_2')([X_7, X_2])

    X_9 = GCNConv(64, activation='relu', kernel_regularizer=l2(l2_reg), name='gcn_5')([X_8, A_in])
    X_10 = Add(name='add_5_1')([X_9, X_1])

    output = GCNConv(num_classes, activation='softmax', name='gcn_output')([X_10, A_in])

    model = Model(inputs=[X_in, A_in], outputs=output, name='GraphSEG_v4')

    return model


def res_u_net(tr_params):
    l2_reg = tr_params['l2_reg']
    F = tr_params['n_node_features']
    num_classes = tr_params['num_classes']

    X_in = Input(shape=(F,), name='X_in')
    A_in = Input(shape=(None,), sparse=True)
    I_in = Input(shape=(), name='segment_ids_in', dtype=tf.int32)

    num_levels = 2
    gc = [None for i in range(2 * (num_levels + 1))]
    pool = [None for i in range(num_levels + 1)]
    upsamp = [None for i in range(num_levels + 1)]

    # gcn conv layer
    print(X_in)
    gc[0] = ChebConv(64, K=2, activation='relu', kernel_regularizer=l2(l2_reg), name='gcn_1')([X_in, A_in])
    # gc[0] = tf.expand_dims(gc[0], axis=0)
    # print(gc[0])

    # downsampling block
    for i in range(num_levels):
        pool[i] = tf.nn.pool(input=[gc[i]], window_shape=[2], pooling_type='MAX', padding='SAME', strides=[2])
        pool[i] = tf.squeeze(pool[i], axis=0)
        gc[i + 1] = ChebConv(64, K=2, activation='relu', kernel_regularizer=l2(l2_reg),
                             name='gcn_%d' % (i + 2))([pool[i], A_in])
    print('Down-sampling done.')

    # last global max pooling layer
    pool[num_levels] = tf.reduce_max(pool[num_levels - 1], axis=[1])

    # upsampling layer_0
    upsamp[0] = tf.stack([pool[-1] for i in range(tf.shape(A_in).shape[0] // 2 ** num_levels)], axis=1)

    # upsampling block
    for i in range(num_levels):
        j = num_levels + 1 + i
        gc[j] = ChebConv(64, 2, K=2, activation='relu', kernel_regularizer=l2(l2_reg),
                         name='gcn_%d' % (j + 1))([tf.concat([gc[num_levels - i], upsamp[i]], axis=2), A_in])
        upsamp[i + 1] = tf.keras.layers.UpSampling1D(2)(gc[j])

    # last gcn conv layer
    output = ChebConv(num_classes, K=2, activation='softmax',
                      name='gcn_output')([tf.concat([gc[0], upsamp[num_levels]], axis=2), A_in])

    model = Model(inputs=[X_in, A_in, I_in], outputs=output, name='GraphUSEG_v1')

    return model

class Res_U(Model):

    def __init__(self, tr_params):
        super(Res_U, self).__init__()
        self.l2_reg = tr_params['l2_reg']
        self.F = tr_params['n_node_features']
        self.num_classes = tr_params['num_classes']

        self.num_levels = 2
        self.gc = [None for i in range(2 * (self.num_levels + 1))]
        self.adj = [None for i in range(2 * (self.num_levels + 1))]
        self.x_pool = [None for i in range(self.num_levels + 1)]
        self.a_pool = [None for i in range(self.num_levels + 1)]
        self.upsamp = [None for i in range(self.num_levels + 1)]

        self.gcn_in = GCNConv(64, activation='relu', kernel_regularizer=l2(self.l2_reg))

    def call(self, inputs):

        self.X_in, self.A_in, self.I_in = inputs

        self.gc[0] = self.gcn_in([self.X_in, self.A_in])
        self.adj[0] = Lambda(self.sparse_to_dense)(self.A_in)
        # self.adj[0] = self.A_in.values
        print(self.adj[0])

        self.downstack()

        self.upsamp[0] = tf.stack([self.x_pool[-1] for i in range(tf.shape(self.A_in).shape[0] // 2 ** self.num_levels)],
                                  axis=1)

        self.upstack()

        output = GCNConv(self.num_classes, K=2, activation='softmax',
                          name='gcn_output')([tf.concat([self.gc[0], self.upsamp[self.num_levels]], axis=2), self.A_in])

        return output

    def downstack(self):

        for i in range(self.num_levels):
            self.x_pool[i] = tf.nn.pool(input=[self.gc[i]], window_shape=[2], pooling_type='MAX',
                                      padding='SAME', strides=[2])
            # self.x_pool[i] = tf.squeeze(self.x_pool[i], axis=0)

            self.a_pool[i] = tf.nn.pool(input=[self.adj[i]], window_shape=[2], pooling_type='MAX',
                                        padding='SAME', strides=[2])
            # self.a_pool[i] = tf.squeeze(self.a_pool[i], axis=0)

            self.gc[i + 1] = GCNConv(64, activation='relu', kernel_regularizer=l2(self.l2_reg),
                                      name='gcn_%d' % (i + 2))([self.x_pool[i], self.a_pool[i]])

        self.x_pool[self.num_levels] = tf.reduce_max(self.x_pool[self.num_levels - 1], axis=[1])
        self.a_pool[self.num_levels] = tf.reduce_max(self.a_pool[self.num_levels - 1], axis=[1])

    def upstack(self):

        self.upsamp[0] = tf.stack([self.x_pool[-1] for i in range(tf.shape(self.X_in).shape[0] // 2 ** self.num_levels)],
                                  axis=1)
        for i in range(self.num_levels):
            j = self.num_levels + 1 + i
            self.gc[j] = GCNConv(64, 2, activation='relu', kernel_regularizer=l2(self.l2_reg),
                                  name='gcn_%d' % (j + 1))([tf.concat([self.gc[self.num_levels - i],
                                                                       self.upsamp[i]], axis=2), self.A_in])
            self.upsamp[i + 1] = UpSampling1D(2)(self.gc[j])

    def sparse_to_dense(self, value):
        if isinstance(value, tf.sparse.SparseTensor):
            return tf.sparse.to_dense(value, validate_indices=False)
        return value

class Graph_U(Model):
    def __init__(self, tr_params, verbose=False):
        super(Graph_U, self).__init__()
        self.l2_reg = tr_params['l2_reg']
        self.F = tr_params['n_node_features']
        self.num_classes = tr_params['num_classes']

        self.num_levels = 1
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

        output = GCSConv(self.num_classes, activation='softmax')(softmax_inputs)

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
            self.x_agg[j+self.num_levels] = self.gcn_block(inputs=(self.x_unpool[j], self.a_unpool[j]), dropout=True)
            if self.v:
                print('After Aggregation:')
                print('X_Agg_{} size: {}'.format(j+self.num_levels, self.x_agg[j+self.num_levels].shape))
                print('-------')


    def gcn_block(self, inputs, filters=16, dropout=False):
        x, a = inputs

        x = GCSConv(filters, activation='relu', kernel_regularizer=l2(self.l2_reg))([x, a])
        x = BatchNormalization()(x)
        if dropout:
            x = Dropout(0.1)(x)

        return x