import tensorflow as tf
from spektral.layers import ops

class unPool(tf.keras.layers.Layer):
    def __init__(self):
        super(unPool, self).__init__()

    def call(self, pooled_inputs):
        X_pool, A_pool, S = pooled_inputs
        ST = tf.transpose(S)
        A_ST = tf.sparse.sparse_dense_matmul(A_pool, ST)
        S_A_ST = tf.matmul(S, A_ST)
        A_unpool = S_A_ST

        X_unpool = ops.matmul_A_B(S, X_pool)
        outputs = [X_unpool, A_unpool]

        return outputs

class dense_to_sparse(tf.keras.layers.Layer):
    def __init__(self):
        super(dense_to_sparse, self).__init__()

    def call(self, dense):
        zero = tf.constant(0, dtype=tf.float32)
        where = tf.not_equal(dense, zero)
        indices = tf.where(where)
        values = tf.gather_nd(dense, indices)
        sparse = tf.SparseTensor(indices, values, dense.shape)

        return sparse
