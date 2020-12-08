import numpy as np
from yaml import safe_load

import tensorflow as tf
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.metrics import CategoricalAccuracy
from spektral.layers import GCNConv, GCSConv, TopKPool, GlobalAvgPool
from spektral.data import DisjointLoader

from spktrl_dataset import PCGraph

BASE_DIR = 'D:/SemanticKITTI/dataset/sequences'
tr_seq_no = '00'
va_seq_no = '01'
te_seq_no = '02'
print('Preparing graph dataset for train sequences..')
data_tr = PCGraph(BASE_DIR, tr_seq_no, stop_idx=10)
print('Preparing graph dataset for validation sequences..')
data_va = PCGraph(BASE_DIR, va_seq_no, stop_idx=1)
print('Preparing graph dataset for test sequences..')
data_te = PCGraph(BASE_DIR, te_seq_no, stop_idx=1)
print('Done.')

F = data_tr.n_node_features
tr_params = safe_load(open('tr_config.yml', 'r'))['training_params']

loader_tr = DisjointLoader(dataset=data_tr, batch_size=1, node_level=True, epochs=tr_params['epochs'])
loader_va = DisjointLoader(dataset=data_va, batch_size=1, node_level=True)
loader_te = DisjointLoader(dataset=data_te, batch_size=1, node_level=True)

# class Net(Model):
#     def __init__(self, **kwargs):
#         super().__init__(**kwargs)
#         self.skp_conv1 = GCSConv(32, activation='relu')
#         self.tkpool1 = TopKPool(ratio=0.5)
#         self.skp_conv2 = GCSConv(32, activation='relu')
#         self.tkpool2 = TopKPool(ratio=0.5)
#         self.skp_conv3 = GCSConv(32, activation='relu')
#         self.glo_pool = GlobalAvgPool()
#         self.seg_conv = GCNConv(28, activation='softmax')
#
#     def call(self, inputs):
#         x_in, a_in = inputs
#
#         x1 = self.skp_conv1([x_in, a_in])
#         x1, a1, I1 = self.tkpool1([x1, a_in, I_in])
#         x2 = self.skp_conv2([x1, a1])
#         x2, a2, I2 = self.tkpool1([x2, a1, I1])
#         x3 = self.skp_conv3([x2, a2])
#         x3 = self.glo_pool([x3, I2])
#         output = self.seg_conv([x3, a2])
#
#         return output

X_in = Input(shape=(F, ), name='X_in')
A_in = Input(shape=(None,), sparse=True)
I_in = Input(shape=(), name='segment_ids_in', dtype=tf.int32)

X_1 = GCSConv(32, activation='relu')([X_in, A_in])
X_1, A_1, I_1 = TopKPool(ratio=0.5)([X_1, A_in, I_in])
X_2 = GCSConv(32, activation='relu')([X_1, A_1])
X_2, A_2, I_2 = TopKPool(ratio=0.5)([X_2, A_1, I_1])
X_3 = GCSConv(32, activation='relu')([X_2, A_2])
X_3 = GlobalAvgPool()([X_3, I_2])
output = GCNConv(28, activation='softmax')([X_3, A_2])

model = Model(inputs=[X_in, A_in, I_in], outputs=output)
opt = Adam(lr=tr_params['learning_rate'])
loss_fn = SparseCategoricalCrossentropy()
acc_fn = CategoricalAccuracy()


################################################################################
# FIT MODEL
################################################################################
print(len(loader_tr.tf_signature()))
@tf.function(input_signature=loader_tr.tf_signature(), experimental_relax_shapes=True)
def train_step(inputs, target):
    with tf.GradientTape() as tape:
        print(target)
        predictions = model(inputs, training=True)
        loss = loss_fn(target, predictions)
        loss += sum(model.losses)
        acc = acc_fn(target, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    opt.apply_gradients(zip(gradients, model.trainable_variables))
    return loss, acc


def evaluate(loader, ops_list):
    output = []
    step = 0
    while step < loader.steps_per_epoch:
        step += 1
        inputs, target = loader.__next__()
        pred = model(inputs, training=False)
        outs = [o(target, pred) for o in ops_list]
        output.append(outs)
    return np.mean(output, 0)


print('Fitting model')
current_batch = epoch = model_loss = model_acc = 0
best_val_loss = np.inf
best_weights = None
patience = tr_params['es_patience']

for batch in loader_tr:
    # # outs = train_step(*batch)
    # (tup), y = batch
    # print(np.unique(tup[1]))
    # efe
    outs = train_step(*batch)
    model_loss += outs[0]
    model_acc += outs[1]
    current_batch += 1
    if current_batch == loader_tr.steps_per_epoch:
        model_loss /= loader_tr.steps_per_epoch
        model_acc /= loader_tr.steps_per_epoch
        epoch += 1

        # Compute validation loss and accuracy
        val_loss, val_acc = evaluate(loader_va, [loss_fn, acc_fn])
        print('Ep. {} - Loss: {:.2f} - Acc: {:.2f} - Val loss: {:.2f} - Val acc: {:.2f}'
              .format(epoch, model_loss, model_acc, val_loss, val_acc))

        # Check if loss improved for early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience = tr_params['es_patience']
            print('New best val_loss {:.3f}'.format(val_loss))
            best_weights = model.get_weights()
        else:
            patience -= 1
            if patience == 0:
                print('Early stopping (best val_loss: {})'.format(best_val_loss))
                break
        model_loss = 0
        model_acc = 0
        current_batch = 0

################################################################################
# EVALUATE MODEL
################################################################################
print('Testing model')
model.set_weights(best_weights)  # Load best model
test_loss, test_acc = evaluate(loader_te, [loss_fn, acc_fn])
print('Done. Test loss: {:.4f}. Test acc: {:.2f}'.format(test_loss, test_acc))
