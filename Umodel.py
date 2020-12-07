import os
import yaml
import numpy as np

import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Dense, Layer, Flatten
from tensorflow.keras.regularizers import l2

from spektral.data import SingleLoader
from spektral.layers import ops
from spektral.layers import GCNConv, MinCutPool
from spektral.layers.ops import sp_matrix_to_sp_tensor

train_config = 'tr_config.yml'
assert os.path.isfile(train_config)
reader = yaml.safe_load(open(train_config, 'r'))
train_params = reader["training_params"]


def down_sample(filters, n_clusters, apply_reg=False):
    ds_module = Sequential()
    if apply_reg:
        reg = l2(train_params['l2_reg'])
    else:
        reg = None
    ds_module.add(
        MinCutPool(k=n_clusters)
    )
    ds_module.add(
        GCNConv(channels=filters, kernel_regularizer=reg)
    )
    return ds_module

def up_sample(filters, n_clusters, apply_reg=False):

    us_module = Sequential()
    if apply_reg:
        reg = l2(train_params['l2_reg'])
    else:
        reg = None
    us_module.add(
        GCNConv(channels=filters, kernel_regularizer=reg)
    )
    us_module.add(
        MC_unPool(k=n_clusters)
    )

    return us_module

# class MCPool(Layer):
#     def __init__(self, n_clusters):
#         super(MCPool, self).__init__()
#         self.k = n_clusters
#         self.d1 = Dense(16, activation='relu',
#                         kernel_initializer='glorot_uniform', bias_initializer='zeros')
#         self.d2 = Dense(self.k, activation='softmax',
#                         kernel_initializer='glorot_uniform', bias_initializer='zeros')
#
#     def call(self, A, X):
#         self.S = self.d1(X)
#         self.S = self.d2(self.S)
#
#         self.A_pool = tf.matmul(tf.transpose(self.S), tf.matmul(A, self.S))
#         self.X_pool = tf.matmul(tf.transpose(self.S), X)
#
#         self.A_pool = tf.linalg.set_diag(self.A_pool, tf.zeros(tf.shape(self.A_pool)[:-1]))
#         D_pool = tf.reduce_sum(self.A_pool, -1)
#         D_pool = tf.sqrt(D_pool)[:, None] + 1e-12
#         self.A_pool = (self.A_pool / D_pool) / tf.transpose(D_pool)
#
#         self.D = tf.reduce_sum(A, axis=-1)
#
#         mcut_loss = self.mcut()
#         self.add_loss(mcut_loss)
#
#         orth_loss = self.orthog()
#         self.add_loss(orth_loss)
#
#         return [self.X_pool, self.A_pool]
#
#     def mcut(self):
#         num = tf.linalg.trace(self.A_pool)
#
#         D_pooled = tf.matmul(tf.transpose(tf.matmul(self.D, self.S)), self.S)
#         den = tf.linalg.trace(D_pooled)
#
#         mincut_loss = -(num / den)
#
#         return mincut_loss
#
#     def orthog(self):
#         St_S = tf.matmul(tf.transpose(self.S), self.S)
#         I_S = tf.eye(self.k)
#
#         orthog_loss = tf.norm(St_S / tf.norm(St_S) - I_S / tf.norm(I_S))
#
#         return orthog_loss

class MC_unPool(Layer):
    def __init__(self, k):
        super(MC_unPool, self).__init__()
        self.k = k
        self.d1 = Dense(16, activation='relu',
                        kernel_initializer='glorot_uniform', bias_initializer='zeros')
        self.d2 = Dense(self.k, activation='softmax',
                        kernel_initializer='glorot_uniform', bias_initializer='zeros')

    def call(self, inputs):
        if len(inputs) == 3:
            X_pool, A_pool, I = inputs
            if K.ndim(I) == 2:
                I = I[:, 0]
        else:
            X_pool, A_pool = inputs
            I = None

        batch_mode = K.ndim(X_pool) == 3

        S = self.d1(X_pool)
        S = self.d2(S)

        A_rec = ops.matmul_A_B_AT(S, A_pool)
        num = tf.linalg.trace(A_rec)
        D = ops.degree_matrix(A_pool)
        den = tf.linalg.trace(ops.matmul_AT_B_A(S, D)) + K.epsilon()
        cut_loss = -(num / den)
        if batch_mode:
            cut_loss = K.mean(cut_loss)
        self.add_loss(cut_loss)

        # Orthogonality regularization
        SS = ops.matmul_AT_B(S, S)
        I_S = tf.eye(self.k, dtype=SS.dtype)
        ortho_loss = tf.norm(
            SS / tf.norm(SS, axis=(-1, -2), keepdims=True) - I_S / tf.norm(I_S),
            axis=(-1, -2)
        )
        if batch_mode:
            ortho_loss = K.mean(ortho_loss)
        self.add_loss(ortho_loss)

        # Pooling
        X_rec = ops.matmul_AT_B(S, X_pool)
        A_rec = tf.linalg.set_diag(
            A_rec, tf.zeros(K.shape(A_rec)[:-1], dtype=A_rec.dtype)
        )  # Remove diagonal
        A_rec = ops.normalize_A(A_rec)

        output = [X_rec, A_rec]

        if I is not None:
            I_mean = tf.math.segment_mean(I, I)
            I_pooled = ops.repeat(I_mean, tf.ones_like(I_mean) * self.k)
            output.append(I_pooled)

        if self.return_mask:
            output.append(S)

        return output

class GUNet(Model):
    def __init__(self):
        super(GUNet, self).__init__()

        self.gcn_start = GCNConv(64, activation='elu')
        k = 5
        self.ds_stack = [
            down_sample(128, k),
            down_sample(256, k, apply_reg=True)
        ]
        self.us_stack = [
            up_sample(256, k),
            up_sample(128, k, apply_reg=True)
        ]
        self.gcn_last = GCNConv(64, activation=None)

    def call(self, inputs):
        skips = []
        x, a = inputs

        for down in self.ds_stack:
            x = down([x, a])
            skips.append(x)

        skips = reversed(skips[:-1])

        for up, skip in zip(self.us_stack, skips):
            x = up([x, a])
            x = tf.keras.layers.Concatenate()([x, skip])

        x = self.gcn_last([x, a])
        return x





