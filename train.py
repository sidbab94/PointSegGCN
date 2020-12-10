import numpy as np
from os import listdir
from yaml import safe_load
import tensorflow as tf
from tensorflow.keras.layers import Dropout, BatchNormalization, Input
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
tf.config.run_functions_eagerly(True)

from spektral.layers import GCNConv, GlobalMaxPool, GCSConv
from spektral.transforms import LayerPreprocess
from spektral.data import DisjointLoader

from spktrl_dataset import PCGraph


BASE_DIR = 'D:/SemanticKITTI/dataset/sequences'
seq_list = np.sort(listdir(BASE_DIR))
split_params = safe_load(open('semantic-kitti.yaml', 'r'))['split']
tr_seq_list = list(seq_list[split_params['train']])
va_seq_list = list(seq_list[split_params['valid']])

print('==================================================================================')
print('1.   Preparing Graph dataset for training sequences..')
data_tr = PCGraph(BASE_DIR, seq_list=tr_seq_list, stop_idx=20, transforms=LayerPreprocess(GCNConv))
print('     Preprocessing..')
print('     .. Done.')
print('----------------------------------------------------------------------------------')
print('2.   Preparing Graph dataset for validation sequences..')
data_va = PCGraph(BASE_DIR, seq_list=va_seq_list, stop_idx=10, transforms=LayerPreprocess(GCNConv))
print('     Preprocessing..')
print('     .. Done.')
print('----------------------------------------------------------------------------------')
print('==================================================================================')

tr_params = safe_load(open('tr_config.yml', 'r'))['training_params']

l2_reg = tr_params['l2_reg']
ep = tr_params['epochs']
num_classes = tr_params['num_classes']


class Net(Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.conv1 = GCNConv(32, activation='relu', kernel_regularizer=l2(l2_reg), name='gcn_1')
        self.conv2 = GCNConv(32, activation='relu', kernel_regularizer=l2(l2_reg), name='gcn_2')
        self.bn1 = BatchNormalization(name='bn_1')
        self.mp1 = GlobalMaxPool(name='mp_1')
        self.conv3 = GCNConv(64, activation='relu', kernel_regularizer=l2(l2_reg), name='gcn_3')
        self.conv4 = GCNConv(64, activation='relu', kernel_regularizer=l2(l2_reg), name='gcn_4')
        self.bn2 = BatchNormalization(name='bn_2')
        self.mp2 = GlobalMaxPool(name='mp_2')
        self.do1 = Dropout(0.5, name='do_1')
        self.conv5 = GCNConv(num_classes, activation='softmax', kernel_regularizer=l2(l2_reg), name='gcn_5')

    def call(self, inputs):
        x, a, ids = inputs

        x = self.conv1([x, a])
        x = self.conv2([x, a])
        x = self.bn1(x)
        x = self.mp1([x, ids])

        x = self.conv3([x, a])
        x = self.conv4([x, a])
        x = self.bn2(x)
        x = self.mp2([x, ids])

        x = self.do1(x)
        output = self.conv5([x, a])
        return output

def subcl_model():
  return Net(name='3_block_GCN')


def func_model():
    F = data_tr.n_node_features
    X_in = Input(shape=(F, ), name='X_in')
    A_in = Input(shape=(None,), sparse=True)
    I_in = Input(shape=(), name='segment_ids_in', dtype=tf.int32)

    X_1 = GCNConv(32, activation='relu', kernel_regularizer=l2(l2_reg), name='gcn_1')([X_in, A_in])
    X_2 = GCNConv(32, activation='relu', kernel_regularizer=l2(l2_reg), name='gcn_2')([X_1, A_in])
    X_3 = BatchNormalization(name='bn_1')(X_2)
    X_4 = GlobalMaxPool(name='mp_1')([X_3, I_in])

    X_5 = GCNConv(64, activation='relu', kernel_regularizer=l2(l2_reg), name='gcn_3')([X_4, A_in])
    X_6 = GCNConv(64, activation='relu', kernel_regularizer=l2(l2_reg), name='gcn_4')([X_5, A_in])
    X_7 = BatchNormalization(name='bn_2')(X_6)
    X_8 = GlobalMaxPool(name='mp_2')([X_7, I_in])

    X_9 = GCNConv(32, activation='relu', kernel_regularizer=l2(l2_reg), name='gcn_5')([X_8, A_in])
    X_10 = GCNConv(32, activation='relu', kernel_regularizer=l2(l2_reg), name='gcn_6')([X_9, A_in])
    X_11 = BatchNormalization(name='bn_3')(X_10)
    X_12 = GlobalMaxPool(name='mp_3')([X_11, I_in])

    X_13 = Dropout(0.5, name='do_1')(X_12)
    output = GCNConv(num_classes, activation='softmax', kernel_regularizer=l2(l2_reg), name='gcn_7')([X_13, A_in])

    model = Model(inputs=[X_in, A_in, I_in], outputs=output)
    return model

# model = subcl_model()
model = func_model()

opt = Adam(lr=tr_params['learning_rate'])
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()

train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
val_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='val_accuracy')
val_miou = tf.keras.metrics.MeanIoU(name='val_miou', num_classes=num_classes)

loader_tr = DisjointLoader(dataset=data_tr, batch_size=1, node_level=True, epochs=ep)
loader_va = DisjointLoader(dataset=data_va, batch_size=1, node_level=True)


def train_step(inputs, target):
    with tf.GradientTape() as tape:
        predictions = model(inputs, training=True)
        loss_tr = loss_fn(target, predictions)
    gradients = tape.gradient(loss_tr, model.trainable_variables)
    opt.apply_gradients(zip(gradients, model.trainable_variables))

    train_accuracy(target, predictions)
    return loss_tr

def evaluate(loader):
    step = 0
    for batch in loader:
        step += 1
        x, adj, y = batch[0]
        I_in = np.array(np.arange(x.shape[0]))
        predictions = model([x, adj, I_in], training=False)
        loss_va = loss_fn(y, predictions)
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

for batch in loader_tr:
    step += 1
    train_accuracy.reset_states()
    val_accuracy.reset_states()
    val_miou.reset_states()
    x, adj, _ = batch[0]
    y = np.squeeze(batch[1])
    I_in = np.array(np.arange(x.shape[0]))
    loss_tr = train_step([x, adj, I_in], y)
    if step == loader_tr.steps_per_epoch:
        epoch += 1
        loss_va = evaluate(loader_va)
        if loss_va < best_val_loss:
            best_val_loss = loss_va
        print('Epoch: {} | '
              'Train loss: {:.4f}, Train accuracy: {:.4f} | '
              'Valid loss: {:.4f}, Valid accuracy: {:.4f} | '
              'Valid MeanIoU: {:.4f} | '
              .format(epoch,
                      loss_tr, train_accuracy.result() * 100,
                      loss_va, val_accuracy.result() * 100,
                      val_miou.result() * 100))
        step = 0

print('----------------------------------------------------------------------------------')
print('     TRAINING END...')
save_path = 'models/infer_v1_1'
model.save(save_path)
print('     Model saved to {}'.format(save_path))
print('==================================================================================')
