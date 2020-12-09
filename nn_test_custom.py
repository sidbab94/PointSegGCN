import numpy as np
from yaml import safe_load
import tensorflow as tf
from tensorflow.keras.layers import Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
tf.config.run_functions_eagerly(True)

from spektral.layers import GCNConv, GlobalMaxPool, MinCutPool
from spektral.transforms import LayerPreprocess
from spektral.data import DisjointLoader

from spktrl_dataset import PCGraph


BASE_DIR = 'D:/SemanticKITTI/dataset/sequences'
tr_seq_no = '00'
va_seq_no = '01'
data_tr = PCGraph(BASE_DIR, tr_seq_no, stop_idx=10)#, transforms=LayerPreprocess(GCNConv))
data_va = PCGraph(BASE_DIR, va_seq_no, stop_idx=1)#, transforms=LayerPreprocess(GCNConv))

tr_params = safe_load(open('tr_config.yml', 'r'))['training_params']

l2_reg = tr_params['l2_reg']
ep = tr_params['epochs']
num_classes = tr_params['num_classes']

class Net(Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.conv1 = GCNConv(32, activation='relu')
        self.conv2 = GCNConv(64, activation='relu', kernel_regularizer=l2(l2_reg))
        self.do1 = Dropout(0.5)
        self.conv3 = GCNConv(num_classes, activation='softmax')

    def call(self, inputs):
        x, a = inputs
        x = self.conv1([x, a])
        x = self.conv2([x, a])
        x = self.do1(x)
        output = self.conv3([x, a])
        return output


model = Net()
opt = Adam(lr=tr_params['learning_rate'])
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()

train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
val_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='val_accuracy')
val_miou = tf.keras.metrics.MeanIoU(name='val_miou', num_classes=num_classes)

loader_tr = DisjointLoader(dataset=data_tr, batch_size=1, node_level=True, epochs=ep)
loader_va = DisjointLoader(dataset=data_va, batch_size=1, node_level=True)


@tf.function
def train_step(inputs, target):
    with tf.GradientTape() as tape:
        predictions = model(inputs, training=True)
        loss_tr = loss_fn(target, predictions)
    gradients = tape.gradient(loss_tr, model.trainable_variables)
    opt.apply_gradients(zip(gradients, model.trainable_variables))

    train_accuracy(target, predictions)
    return loss_tr

@tf.function
def evaluate(loader):
    step = 0
    for batch in loader:
        step += 1
        x, adj, y = batch[0]
        predictions = model([x, adj], training=False)
        loss_va = loss_fn(y, predictions)
        val_accuracy(y, predictions)
        pred_labels = np.argmax(predictions, axis=-1)
        val_miou(y, pred_labels)
        if step == loader.steps_per_epoch:
            return loss_va


print('Fitting model')

best_val_loss = 99999
step = 0

for batch in loader_tr:
    step += 1
    train_accuracy.reset_states()
    val_accuracy.reset_states()
    val_miou.reset_states()
    x, adj, y = batch[0]
    loss_tr = train_step([x, adj], y)

    if step == loader_tr.steps_per_epoch:
        loss_va = evaluate(loader_va)
        if loss_va < best_val_loss:
            best_val_loss = loss_va
        print('Train loss: {:.4f}, Train accuracy: {:.4f} | '
              'Valid loss: {:.4f}, Valid accuracy: {:.4f} | '
              'Valid MeanIoU: {:.4f} | '
              .format(loss_tr, train_accuracy.result() * 100,
                      loss_va, val_accuracy.result() * 100,
                      val_miou.result() * 100))
        # Reset epoch
        step = 0

