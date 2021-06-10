import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import Layer


class GConv(Layer):

    def __init__(self, units, activation='relu',
                 kernel_init='he_normal', bias_init='random_normal', **kwargs):
        super(GConv, self).__init__()
        self.units = units
        self.kernel_init = kernel_init
        self.bias_init = bias_init
        self.act_name = activation
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
            shape=(self.units,), initializer=self.bias_init, trainable=True
        )

    def call(self, inputs):
        x, a = inputs
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
