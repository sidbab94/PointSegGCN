import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import Layer


class GConv(Layer):

    def __init__(self, units, activation='relu',
                 kernel_init='he_normal'):
        super(GConv, self).__init__()
        self.units = units
        self.kernel_init = kernel_init
        self.act = tf.keras.layers.Activation(activation)

    def build(self, input_shape):
        input_dim = input_shape[0][-1]
        self.w = self.add_weight(
            name='w',
            shape=(input_dim, self.units),
            initializer=self.kernel_init,
            regularizer=l2(0.001),
            trainable=True
        )
        self.b = self.add_weight(
            name='b',
            shape=(self.units,), initializer="random_normal", trainable=True
        )

    def call(self, inputs):
        x, a = inputs
        output = K.dot(x, self.w)
        output = tf.sparse.sparse_dense_matmul(a, output)
        output = K.bias_add(output, self.b)
        output = self.act(output)
        return output

    def get_config(self):
        base_config = super(GConv, self).get_config()
        base_config.update({"units": self.units})
        return base_config


class CRF(Layer):

    def __init__(self, activation='softmax'):
        super(CRF, self).__init__()
        self.activation = activation
        self.classes = 20

        # Gaussian weights
        self.theta = tf.Variable(tf.random.truncated_normal([3], mean=1, stddev=0.01))
        self.W = tf.Variable(tf.random.truncated_normal([2], mean=0.5, stddev=0.01))

        # self.compatible_weight = tf.Variable(tf.random.truncated_normal(shape=[self.classes, self.classes], stddev=0.01))

    def call(self, inputs, features):
        xyz, rgb = inputs[:, :3], inputs[:, -3:]

        features_normed = tf.nn.softmax(features)

        for i in range(15):

            # compute weights with Gaussian kernels
            ker_appearance = tf.reduce_sum(tf.square(xyz * self.theta[0]), axis=1) + tf.reduce_sum(
                tf.square(rgb * self.theta[1]), axis=1)

            ker_smooth = tf.reduce_sum(tf.square(xyz * self.theta[2]), axis=1)

            ker_appearance = tf.exp(-ker_appearance)

            ker_smooth = tf.exp(-ker_smooth)

            Q_weight = self.W[0] * ker_appearance + self.W[1] * ker_smooth  # (batch_size, num_point, knn_num)

            Q_weight = tf.expand_dims(Q_weight, axis=1)  # (batch_size, num_point, 1, knn_num)

            # message passing
            Q_til_weighted = tf.matmul(tf.transpose(Q_weight), features_normed)  # (batch_size, num_point, 1, channels)

            # print(Q_til_weighted)
            # Q_til_weighted = tf.squeeze(Q_til_weighted, axis=-1)  # (batch_size, num_point, channels)

            # compatibility transform
            # Q_til_weighted = tf.nn.conv1d(Q_til_weighted, self.compatible_weight, 1, padding='SAME')

            # adding unary potentials
            features += Q_til_weighted

            output = tf.nn.softmax(features)

        return output

    def get_config(self):
        base_config = super(CRF, self).get_config()
        base_config.update({"activation": self.activation})
        return base_config
