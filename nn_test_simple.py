import numpy as np
import os

import tensorflow as tf
from tensorflow.keras.layers import Input, Dropout, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.metrics import CategoricalAccuracy

from spektral.layers import GCNConv, GlobalMaxPool, MinCutPool
from spektral.layers.ops import sp_matrix_to_sp_tensor
from spektral.data import DisjointLoader

from spktrl_dataset import PCGraph


BASE_DIR = 'D:/SemanticKITTI/dataset/sequences'
tr_seq_no = '00'
va_seq_no = '01'
data_tr = PCGraph(BASE_DIR, tr_seq_no, stop_idx=20)
data_va = PCGraph(BASE_DIR, va_seq_no, stop_idx=4)

l2_reg = 5e-4
ep = 10

class Net(Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.conv1 = GCNConv(32, activation='elu')
        self.conv2 = GCNConv(64, activation='elu', kernel_regularizer=l2(l2_reg))
        # self.do1 = Dropout(0.5)
        self.conv3 = GCNConv(32, activation='elu')
        self.pool1 = GlobalMaxPool()
        self.conv4 = GCNConv(28, activation='softmax')

    def call(self, inputs):
        x, a = inputs
        x = self.conv1([x, a])
        x = self.conv2([x, a])
        # x = self.do1([x, a])
        x = self.conv3([x, a])
        output = self.conv4([x, a])
        return output

model = Net()
model.compile('adam', 'sparse_categorical_crossentropy',
              metrics=['sparse_categorical_accuracy'])


loader_tr = DisjointLoader(dataset=data_tr, batch_size=1, node_level=True, epochs=ep)
loader_va = DisjointLoader(dataset=data_va, batch_size=1, node_level=True)

def evaluate(loader):
    step = 0
    results = []
    for batch in loader:
        step += 1
        x, adj, y = batch[0]
        l, a = model.test_on_batch([x, adj], y)
        results.append((l, a))
        if step == loader.steps_per_epoch:
            return np.mean(results, 0)


# opt = Adam(lr=0.001)
# loss_fn = SparseCategoricalCrossentropy()
# acc_fn = CategoricalAccuracy()
# @tf.function(input_signature=loader_tr.tf_signature(), experimental_relax_shapes=True)
# def train_step(inputs, target):
#     with tf.GradientTape() as tape:
#         print(len(inputs))
#         predictions = model(inputs, training=True)
#         loss = loss_fn(target, predictions)
#         loss += sum(model.losses)
#         acc = acc_fn(target, predictions)
#     gradients = tape.gradient(loss, model.trainable_variables)
#     opt.apply_gradients(zip(gradients, model.trainable_variables))
#     return loss, acc

# print(data.n_labels)
print('Fitting model')
# current_batch = epoch = model_loss = model_acc = 0
# best_val_loss = np.inf
# best_weights = None
patience = 10
best_val_loss = 99999
current_patience = patience
results_tr = []
weights_tr = []
step = 0
for batch in loader_tr:
    step += 1
    x, adj, y = batch[0]
    # print(len(inp))
    l, a = model.train_on_batch([x, adj], y)
    results_tr.append((l, a))
    weights_tr.append(len(y))

    if step == loader_tr.steps_per_epoch:
        results_va = evaluate(loader_va)
        if results_va[0] < best_val_loss:
            best_val_loss = results_va[0]
            current_patience = patience
        else:
            current_patience -= 1
            if current_patience == 0:
                print('Early stopping')
                break
        # Print results
        results_tr = np.average(results_tr, 0, weights=weights_tr)
        print('Train loss: {:.4f}, acc: {:.4f} | '
              'Valid loss: {:.4f}, acc: {:.4f} | '
              .format(*results_tr, *results_va))

        # Reset epoch
        results_tr = []
        weights_tr = []
        step = 0


    # outs = train_step(inputs=(current_batch, inp), target=y)
    # model_loss += outs[0]
    # model_acc += outs[1]
    # current_batch += 1
    # if current_batch == loader_tr.steps_per_epoch:
    #     model_loss /= loader_tr.steps_per_epoch
    #     model_acc /= loader_tr.steps_per_epoch
    #     epoch += 1
    #
    #     print('Ep. {} - Loss: {:.2f} - Acc: {:.2f}'
    #           .format(epoch, model_loss, model_acc))
    #
    #     model_loss = 0
    #     model_acc = 0
    #     current_batch = 0





# #####
# X = np.genfromtxt('samples/testpc.csv', delimiter=',')
# X = X[:, :3]
# Y = get_labels('samples/testpc.label')
# A = adjacency(X, nn=5)
#
# n_classes = len(np.unique(Y))
# N, F = X.shape[0], X.shape[1]
# print(N, F)
#
# adj = GCNConv.preprocess(A)
# adj = sp_matrix_to_sp_tensor(adj)
#
# x_in = Input(shape=(F,))
# a_in = Input((N,), sparse=True)
# x_1 = GCNConv(32, activation='relu')([x_in, a_in])
# x_1 = Dropout(0.5)(x_1)
# x_2 = GCNConv(64, activation='relu')([x_1, a_in])
# x_2 = Dropout(0.5)(x_2)
# x_3 = GCNConv(n_classes, activation='softmax')([x_2, a_in])
#
# model = Model(inputs=[x_in, a_in], outputs=x_3)
# optimizer = Adam(lr=0.001)
# model.compile(optimizer=optimizer, loss=SparseCategoricalCrossentropy())
# model.summary()

