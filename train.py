import numpy as np
from os import listdir
import datetime
from yaml import safe_load
import tensorflow as tf
from tensorflow.keras.layers import Dropout, BatchNormalization, Input
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
tf.config.run_functions_eagerly(True)

from spektral.layers import GCNConv, GlobalMaxPool, GlobalAttnSumPool
from spektral.transforms import LayerPreprocess
from spektral.data import DisjointLoader

from spktrl_dataset import PCGraph
from lovasz_loss import lovasz_softmax_flat


BASE_DIR = 'D:/SemanticKITTI/dataset/sequences'
seq_list = np.sort(listdir(BASE_DIR))
split_params = safe_load(open('semantic-kitti.yaml', 'r'))['split']
tr_seq_list = list(seq_list[split_params['train']])
va_seq_list = list(seq_list[split_params['valid']])

print('==================================================================================')
print('1.   Preparing Graph dataset for training sequences..')
data_tr = PCGraph(BASE_DIR, seq_list=tr_seq_list, stop_idx=100, transforms=LayerPreprocess(GCNConv))
print('     Preprocessing..')
print('     .. Done.')
print('----------------------------------------------------------------------------------')
print('2.   Preparing Graph dataset for validation sequences..')
data_va = PCGraph(BASE_DIR, seq_no='08', stop_idx=200, transforms=LayerPreprocess(GCNConv))
print('     Preprocessing..')
print('     .. Done.')
print('----------------------------------------------------------------------------------')
# print('==================================================================================')

# print('==================================================================================')
# print('1.   Preparing Graph dataset for training sequences..')
# data_tr = PCGraph(BASE_DIR, seq_no='00', seq_list=None, stop_idx=10, transforms=LayerPreprocess(GCNConv))
# print('     .. Done.')
# print('----------------------------------------------------------------------------------')
# print('2.   Preparing Graph dataset for validation sequences..')
# data_va = PCGraph(BASE_DIR, seq_no='01', seq_list=None, stop_idx=1, transforms=LayerPreprocess(GCNConv))
# print('     .. Done.')
# print('----------------------------------------------------------------------------------')
# print('==================================================================================')


tr_params = safe_load(open('tr_config.yml', 'r'))['training_params']

l2_reg = tr_params['l2_reg']
EPOCHS = tr_params['epochs']
num_classes = tr_params['num_classes']
patience = tr_params['es_patience']
loss_switch_ep = tr_params['lovasz_switch']

def conv_block(parents, filters, id, pool=False, batch_norm=False):
    X_in, A_in, I_in = parents
    x = GCNConv(filters, activation='relu', kernel_regularizer=l2(l2_reg), name='gcn_' + str(id))([X_in, A_in])
    x = GCNConv(filters, activation='relu', kernel_regularizer=l2(l2_reg), name='gcn_' + str(id+1))([x, A_in])
    if batch_norm:
        x = BatchNormalization(name='bn_' + str(id))(x)
    if pool:
        x = GlobalMaxPool(name='mp_' + str(id))([x, I_in])
    output = x
    return output

def func_model_1():
    F = data_tr.n_node_features
    X_in = Input(shape=(F, ), name='X_in')
    A_in = Input(shape=(None,), sparse=True)
    I_in = Input(shape=(), name='segment_ids_in', dtype=tf.int32)

    X_1 = conv_block((X_in, A_in, I_in), 64, 1, batch_norm=True)
    X_2 = conv_block((X_1, A_in, I_in), 64, 2, batch_norm=True, pool=True)
    X_3 = conv_block((X_2, A_in, I_in), 64, 3, batch_norm=True, pool=True)
    X_4 = conv_block((X_3, A_in, I_in), 64, 4, batch_norm=True, pool=True)
    X_5 = conv_block((X_4, A_in, I_in), 64, 5, pool=True)
    X_6 = Dropout(0.5, name='do_1')(X_5)
    X_7 = tf.keras.layers.concatenate([X_6, X_5, X_4, X_3, X_2], axis=1)
    X_8 = Dropout(0.5, name='do_2')(X_7)
    X_9 = conv_block((X_8, A_in, I_in), 64, 6)
    output = GCNConv(num_classes, activation='softmax', kernel_regularizer=l2(l2_reg), name='gcn_7')([X_9, A_in])

    model = Model(inputs=[X_in, A_in, I_in], outputs=output, name='GraphSEG_v1')
    return model

def func_model_2():
    F = data_tr.n_node_features
    X_in = Input(shape=(F, ), name='X_in')
    A_in = Input(shape=(None,), sparse=True)
    I_in = Input(shape=(), name='segment_ids_in', dtype=tf.int32)

    X_1 = conv_block((X_in, A_in, I_in), 32, 1, batch_norm=True)
    X_2 = conv_block((X_1, A_in, I_in), 64, 2, batch_norm=True, pool=True)
    X_3 = conv_block((X_2, A_in, I_in), 64, 3, batch_norm=True, pool=True)
    X_4 = Dropout(0.5, name='do_1')(X_3)
    X_5 = tf.keras.layers.concatenate([X_4, X_3, X_2])
    X_6 = Dropout(0.5, name='do_2')(X_5)
    X_7 = conv_block((X_6, A_in, I_in), 64, 4)
    output = GCNConv(num_classes, activation='softmax', kernel_regularizer=l2(l2_reg), name='output_gcn')([X_7, A_in])

    model = Model(inputs=[X_in, A_in, I_in], outputs=output, name='GraphSEG_v2')
    return model

model = func_model_1()
model.summary()
with open('ModelSummary.txt', 'w') as fh:
    model.summary(print_fn=lambda f: fh.write(f + '\n'))



opt = Adam(lr=tr_params['learning_rate'])
loss_cross_entropy = tf.keras.losses.SparseCategoricalCrossentropy()

train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
train_miou = tf.keras.metrics.MeanIoU(name='train_miou', num_classes=num_classes)
val_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='val_accuracy')
val_miou = tf.keras.metrics.MeanIoU(name='val_miou', num_classes=num_classes)

current_time = datetime.datetime.now().strftime("%Y-%m-%d--%H.%M.%S")
train_log_dir = 'TB_logs/' + current_time + '/train'
valid_log_dir = 'TB_logs/' + current_time + '/valid'
train_summary_writer = tf.summary.create_file_writer(train_log_dir)
valid_summary_writer = tf.summary.create_file_writer(valid_log_dir)
# tb_callback = tf.keras.callbacks.TensorBoard('TB_logs')

loader_tr = DisjointLoader(dataset=data_tr, batch_size=1, node_level=True, shuffle=True, epochs=ep)
loader_va = DisjointLoader(dataset=data_va, batch_size=1, shuffle=True, node_level=True)


def train_step(loader, lovasz=False):
    step = 0
    for batch in loader:
        step += 1
        x, adj, _ = batch[0]
        y = np.squeeze(batch[1])
        I_in = np.array(np.arange(x.shape[0]))
        inputs = [x, adj, I_in]
        target = y
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
        if step == loader_tr.steps_per_epoch:
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

step = 0
lovasz = False

for epoch in range(EPOCHS):

    train_accuracy.reset_states()
    train_miou.reset_states()
    val_accuracy.reset_states()
    val_miou.reset_states()

    if epoch == loss_switch_ep:
        lovasz = True

    loss_tr = train_step(loader_tr, lovasz)
    with train_summary_writer.as_default():
        tf.summary.scalar('loss', loss_tr.numpy(), step=epoch)
        tf.summary.scalar('accuracy', train_accuracy.result() * 100, step=epoch)
        tf.summary.scalar('mIoU', train_miou.result() * 100, step=epoch)

    loss_va = evaluate(loader_va, lovasz)
    with valid_summary_writer.as_default():
        tf.summary.scalar('loss', loss_va.numpy(), step=epoch)
        tf.summary.scalar('accuracy', val_accuracy.result() * 100, step=epoch)
        tf.summary.scalar('mIoU', val_miou.result() * 100, step=epoch)

    print('Epoch: {} | '
          'Train loss: {:.4f}, Train accuracy: {:.4f}, Train MeanIoU: {:.4f} ||'
          'Valid loss: {:.4f}, Valid accuracy: {:.4f}, Valid MeanIoU: {:.4f} | '
          .format(epoch,
                  loss_tr.numpy(), train_accuracy.result() * 100, train_miou.result() * 100,
                  loss_va.numpy(), val_accuracy.result() * 100, val_miou.result() * 100))

print('----------------------------------------------------------------------------------')
print('     TRAINING END...')

save_path = 'models/infer_v1_3'
model.save(save_path)
print('     Model saved to {}'.format(save_path))
print('==================================================================================')
