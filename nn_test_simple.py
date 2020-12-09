import numpy as np

import tensorflow as tf
from tensorflow.keras.layers import Input, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.metrics import MeanIoU as miou

from spektral.layers import GCNConv, GlobalMaxPool, MinCutPool
from spektral.transforms import LayerPreprocess
from spektral.data import DisjointLoader

from spktrl_dataset import PCGraph


BASE_DIR = 'D:/SemanticKITTI/dataset/sequences'
tr_seq_no = '00'
va_seq_no = '01'
data_tr = PCGraph(BASE_DIR, tr_seq_no, stop_idx=2, transforms=LayerPreprocess(GCNConv))
data_va = PCGraph(BASE_DIR, va_seq_no, stop_idx=1, transforms=LayerPreprocess(GCNConv))

l2_reg = 5e-4
ep = 100

class Net(Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.conv1 = GCNConv(32, activation='elu')
        self.conv2 = GCNConv(64, activation='elu', kernel_regularizer=l2(l2_reg))
        self.do1 = Dropout(0.2)
        self.conv3 = GCNConv(32, activation='elu')
        self.pool1 = GlobalMaxPool()
        self.conv4 = GCNConv(19, activation='softmax')

    def call(self, inputs):
        x, a = inputs
        x = self.conv1([x, a])
        x = self.conv2([x, a])
        x = self.do1(x)
        # x = self.conv3([x, a])
        output = self.conv4([x, a])
        return output

model = Net()
model.compile('adam', 'sparse_categorical_crossentropy',
              metrics=[miou(num_classes=19)])

print(model.summary())
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


print('Fitting model')

patience = 10
best_val_loss = 99999
current_patience = patience
results_tr = []
weights_tr = []
step = 0
for batch in loader_tr:
    step += 1
    x, adj, y = batch[0]
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
