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
