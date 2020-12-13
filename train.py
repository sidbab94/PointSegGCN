import numpy as np
from os import listdir
from yaml import safe_load
import tensorflow as tf
from tensorflow.keras.layers import Dropout, BatchNormalization, Input
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
tf.config.run_functions_eagerly(True)

from spektral.layers import GCNConv, GlobalMaxPool, GlobalAttentionPool
from spektral.transforms import LayerPreprocess
from spektral.data import DisjointLoader

from spktrl_dataset import PCGraph
from lovasz_loss import lovasz_softmax_flat


BASE_DIR = 'D:/SemanticKITTI/dataset/sequences'
seq_list = np.sort(listdir(BASE_DIR))
split_params = safe_load(open('semantic-kitti.yaml', 'r'))['split']
tr_seq_list = list(seq_list[split_params['train']])
va_seq_list = list(seq_list[split_params['valid']])

# print('==================================================================================')
# print('1.   Preparing Graph dataset for training sequences..')
# data_tr = PCGraph(BASE_DIR, seq_list=tr_seq_list, stop_idx=100, transforms=LayerPreprocess(GCNConv))
# print('     Preprocessing..')
# print('     .. Done.')
# print('----------------------------------------------------------------------------------')
# print('2.   Preparing Graph dataset for validation sequences..')
# data_va = PCGraph(BASE_DIR, seq_list=va_seq_list, stop_idx=100, transforms=LayerPreprocess(GCNConv))
# print('     Preprocessing..')
# print('     .. Done.')
# print('----------------------------------------------------------------------------------')
# print('==================================================================================')

print('==================================================================================')
print('1.   Preparing Graph dataset for training sequences..')
data_tr = PCGraph(BASE_DIR, seq_no='00', seq_list=None, stop_idx=1, transforms=LayerPreprocess(GCNConv))
print('     Preprocessing..')
print('     .. Done.')
print('----------------------------------------------------------------------------------')
print('2.   Preparing Graph dataset for validation sequences..')
data_va = PCGraph(BASE_DIR, seq_no='01', seq_list=None, stop_idx=1, transforms=LayerPreprocess(GCNConv))
print('     Preprocessing..')
print('     .. Done.')
print('----------------------------------------------------------------------------------')
print('==================================================================================')



tr_params = safe_load(open('tr_config.yml', 'r'))['training_params']

l2_reg = tr_params['l2_reg']
ep = tr_params['epochs']
num_classes = tr_params['num_classes']
patience = tr_params['es_patience']
loss_switch_ep = tr_params['lovasz_switch']

def conv_block(parent, filters, id, att_pool=False, batch_norm=False):
    X_in, A_in, I_in = parent
    x = GCNConv(filters, activation='relu', kernel_regularizer=l2(l2_reg), name='gcn_' + str(id))([X_in, A_in])
    x = GCNConv(filters, activation='relu', kernel_regularizer=l2(l2_reg), name='gcn_' + str(id+1))([x, A_in])
    if batch_norm:
        x = BatchNormalization(name='bn_' + str(id))(x)
    if att_pool:
        output = GlobalAttentionPool(filters, name='ap_' + str(id))([x, I_in])
    else:
        output = GlobalMaxPool(name='mp_' + str(id))([x, I_in])
    return output

def func_model():
    F = data_tr.n_node_features
    X_in = Input(shape=(F, ), name='X_in')
    A_in = Input(shape=(None,), sparse=True)
    I_in = Input(shape=(), name='segment_ids_in', dtype=tf.int32)

    X_1 = conv_block((X_in, A_in, I_in), 32, 1)
    X_2 = conv_block((X_1, A_in, I_in), 64, 2, batch_norm=True)
    X_3 = conv_block((X_2, A_in, I_in), 128, 3, batch_norm=True)
    X_4 = conv_block((X_3, A_in, I_in), 64, 4, att_pool=True)
    X_5 = Dropout(0.5, name='do_1')(X_4)
    X_6 = conv_block((X_5, A_in, I_in), 32, 5, att_pool=True)
    X_7 = Dropout(0.5, name='do_2')(X_6)

    output = GCNConv(num_classes, activation='softmax', kernel_regularizer=l2(l2_reg), name='gcn_7')([X_7, A_in])

    model = Model(inputs=[X_in, A_in, I_in], outputs=output)
    return model

# model = subcl_model()
model = func_model()
print(model.summary())

opt = Adam(lr=tr_params['learning_rate'])
loss_cross_entropy = tf.keras.losses.SparseCategoricalCrossentropy()

train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
train_miou = tf.keras.metrics.MeanIoU(name='train_miou', num_classes=num_classes)
val_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='val_accuracy')
val_miou = tf.keras.metrics.MeanIoU(name='val_miou', num_classes=num_classes)

loader_tr = DisjointLoader(dataset=data_tr, batch_size=1, node_level=True, shuffle=True, epochs=ep)
loader_va = DisjointLoader(dataset=data_va, batch_size=1, shuffle=True, node_level=True)


def train_step(inputs, target, lovasz=False):
    with tf.GradientTape() as tape:
        predictions = model(inputs, training=True)
        if lovasz:
            loss_tr = lovasz_softmax_flat(predictions, target)
        else:
            loss_tr = loss_cross_entropy(target, predictions)
    gradients = tape.gradient(loss_tr, model.trainable_variables)
    opt.apply_gradients(zip(gradients, model.trainable_variables))
    pred_labels = np.argmax(predictions, axis=-1)
    train_miou(y, pred_labels)
    train_accuracy(target, predictions)
    return loss_tr

def evaluate(loader, lovasz=False):
    step = 0
    for batch in loader:
        step += 1
        x, adj, _ = batch[0]
        y = np.squeeze(batch[1])
        I_in = np.array(np.arange(x.shape[0]))
        predictions = model([x, adj, I_in], training=False)
        if lovasz:
            loss_va = lovasz_softmax_flat(predictions, y)
        else:
            loss_va = loss_cross_entropy(y, predictions)
        val_accuracy(y, predictions)
        pred_labels = np.argmax(predictions, axis=-1)
        val_miou(y, pred_labels)
        if step == loader.steps_per_epoch:
            return loss_va

print('     TRAINING START...')
print('----------------------------------------------------------------------------------')

best_val_loss = 99999
step = 0
epoch = 0
lovasz = False

for batch in loader_tr:
    step += 1
    train_accuracy.reset_states()
    train_miou.reset_states()
    val_accuracy.reset_states()
    val_miou.reset_states()
    x, adj, _ = batch[0]
    y = np.squeeze(batch[1])
    I_in = np.array(np.arange(x.shape[0]))
    loss_tr = train_step([x, adj, I_in], y, lovasz)
    if step == loader_tr.steps_per_epoch:
        epoch += 1
        if epoch == loss_switch_ep:
            lovasz = True
        loss_va = evaluate(loader_va, lovasz)
        if loss_va < best_val_loss:
            best_val_loss = loss_va
            current_patience = patience
        else:
            current_patience -= 1
            if current_patience == 0:
                print('----- Early stopping @ {} epochs! -----'.format(epoch))
                break

        print('Epoch: {} | '
              'Train loss: {:.4f}, Train accuracy: {:.4f}, Train MeanIoU: {:.4f} ||'
              'Valid loss: {:.4f}, Valid accuracy: {:.4f}, Valid MeanIoU: {:.4f} | '
              .format(epoch,
                      loss_tr.numpy(), train_accuracy.result() * 100, train_miou.result() * 100,
                      loss_va.numpy(), val_accuracy.result() * 100, val_miou.result() * 100))
        step = 0

print('----------------------------------------------------------------------------------')
print('     TRAINING END...')
save_path = 'models/infer_v1_3'
model.save(save_path)
print('     Model saved to {}'.format(save_path))
print('==================================================================================')
