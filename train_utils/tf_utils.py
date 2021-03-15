import tensorflow as tf
from scipy.sparse import spdiags
from tensorflow.keras import backend as K
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import Layer
from spektral.layers import ops

class unPool(tf.keras.layers.Layer):
    """
    Performs an unpooling operation based on Min-Cut pooling
    (https://danielegrattarola.github.io/posts/2019-07-25/mincut-pooling.html)

    Call arguments:
        pooled_inputs: pooled point cloud array, adjacency matrix and
                        clustering mask produced by parent MinCut Pool layer
    """
    def __init__(self):
        super(unPool, self).__init__()

    def call(self, pooled_inputs):
        X_pool, A_pool, S = pooled_inputs
        A_pool = dense_to_sparse()(A_pool)

        ST = tf.transpose(S)
        A_ST = tf.sparse.sparse_dense_matmul(A_pool, ST)
        S_A_ST = tf.matmul(S, A_ST)

        A_unpool = S_A_ST

        X_unpool = ops.matmul_A_B(S, X_pool)
        outputs = [X_unpool, A_unpool]

        return outputs

class dense_to_sparse(tf.keras.layers.Layer):
    """
    Converts dense matrix back to sparse format

    Call arguments:
        dense: Dense matrix to convert
    """
    def __init__(self):
        super(dense_to_sparse, self).__init__()

    def call(self, dense):
        zero = tf.constant(0, dtype=tf.float32)
        where = tf.not_equal(dense, zero)
        indices = tf.where(where)
        values = tf.gather_nd(dense, indices)
        sparse = tf.SparseTensor(indices, values, dense.shape)

        return sparse

class GConv(Layer):
    def __init__(self, filters, activation='relu',
                 kernel_init='he_normal'):
        super(GConv, self).__init__()
        self.filters = filters
        self.kernel_init = kernel_init
        self.kernel_reg = l2(0.001)
        self.act = tf.keras.layers.Activation(activation)

    def build(self, input_shape):
        input_dim = input_shape[0][-1]
        self.w = self.add_weight(
            shape=(input_dim, self.filters),
            initializer=self.kernel_init,
            regularizer=self.kernel_reg,
            trainable=True
        )
        self.b = self.add_weight(
            shape=(self.filters,), initializer="random_normal", trainable=True
        )

    def call(self, inputs):
        x, a = inputs
        a = self.normalize_A(a)
        output = K.dot(x, self.w)
        output = tf.sparse.sparse_dense_matmul(a, output)
        output = K.bias_add(output, self.b)
        output = self.act(output)
        return output

    def normalize_A(self, a):

        n, m = a.shape
        diags = a.sum(axis=1).flatten()
        D = spdiags(diags, [0], m, n, format="csr")
        D = D.power(-0.5)
        L = D.dot(a).dot(D)
        return L
