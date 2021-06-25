import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import Layer
from tensorflow.keras.models import Model


class GConv(Layer):

    def __init__(self, units, dropout=False, activation='relu',
                 kernel_init='he_normal', bias_init='random_normal', **kwargs):
        super(GConv, self).__init__()
        self.units = units
        self.dropout = dropout
        self.kernel_init = kernel_init
        self.bias_init = bias_init
        self.act_name = activation
        self.act = tf.keras.layers.Activation(activation)

    def build(self, input_shape):
        # print(input_shape)
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
            shape=(self.units,), initializer=self.bias_init, trainable=True
        )

    def call(self, inputs):
        x, a = inputs
        if self.dropout:
            x = tf.nn.dropout(x, rate=0.2, seed=1)
        output = tf.matmul(x, self.w)
        output = tf.sparse.sparse_dense_matmul(a, output)
        # output = K.bias_add(output, self.b)
        output = self.act(output + self.b)
        return output

    def get_config(self):
        config = {
            'units':
                self.units,
        }
        base_config = super(GConv, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))



class ConcatAdj(Layer):

    def __init__(self, block_diag=True, **kwargs):
        super(ConcatAdj, self).__init__()
        self.block_diag = block_diag

    def call(self, a1, a2):
        M, N = a1.shape[0], a2.shape[0]
        new_inds = tf.concat((a1.indices,
                              tf.add(a2.indices, tf.constant(M, dtype=tf.int64))), 0)
        new_vals = tf.concat((a1.values, a2.values), -1)
        a_out = tf.sparse.SparseTensor(indices=new_inds,
                                       values=new_vals, dense_shape=(M + N, M + N))
        return a_out

    def get_config(self):
        config = {
            'block_diag':
                self.block_diag,
        }
        base_config = super(ConcatAdj, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class MLPBlock(Model):

    def __init__(self, filters):
        super(MLPBlock, self).__init__()
        filters1, filters2, filters3 = filters

        self.conv2a = tf.keras.layers.Conv1D(filters1, 1)
        # self.bn2a = tf.keras.layers.BatchNormalization()

        self.conv2b = tf.keras.layers.Conv1D(filters2, 1)
        # self.bn2b = tf.keras.layers.BatchNormalization()

        self.conv2c = tf.keras.layers.Conv1D(filters3, 1)
        # self.bn2c = tf.keras.layers.BatchNormalization()

    def call(self, input_tensor, training=False):

        x = tf.expand_dims(input_tensor, 0)

        x = self.conv2a(x)
        # x = self.bn2a(x, training=training)
        x = tf.nn.relu(x)

        x = self.conv2b(x)
        # x = self.bn2b(x, training=training)
        x = tf.nn.relu(x)

        x = self.conv2c(x)
        # x = self.bn2c(x, training=training)

        return tf.nn.softmax(x)


class CyclicalLR(tf.keras.optimizers.schedules.LearningRateSchedule):

    def __init__(self, base_lr=0.001, max_lr=0.01, step_size=2000., name=None):
        self.base_lr = base_lr
        self.max_lr = max_lr
        self.step_size = step_size
        self.name = name

        self.scale_fn = lambda x: 1 / (2. ** (x - 1))
        self.clr_iterations = 0.
        self.trn_iterations = 0.

    def __call__(self, step):
        with tf.name_scope(self.name or "CyclicalLearningRate"):
            base_lr = tf.convert_to_tensor(
                self.base_lr, name="base_lr"
            )
            dtype = base_lr.dtype
            max_lr = tf.cast(self.max_lr, dtype)
            step_size = tf.cast(self.step_size, dtype)
            step_as_dtype = tf.cast(step, dtype)
            cycle = tf.floor(1 + step_as_dtype / (2 * step_size))
            x = tf.abs(step_as_dtype / step_size - 2 * cycle + 1)

            return base_lr + (max_lr - base_lr) * tf.maximum(tf.cast(0, dtype), (1 - x)) * self.scale_fn(cycle)

    def get_config(self):
        return {
            "initial_learning_rate": self.base_lr,
            "maximal_learning_rate": self.max_lr,
            "step_size": self.step_size,
            "name": self.name
        }


