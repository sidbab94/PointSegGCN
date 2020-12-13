from visualization import ShowPC
from tensorflow.keras.models import load_model
from tensorflow.keras.metrics import MeanIoU
from spktrl_dataset import PCGraph
from spektral.transforms import LayerPreprocess
from spektral.layers import GCNConv
import numpy as np
from os import listdir
from yaml import safe_load
from spektral.data import DisjointLoader
from lovasz_loss import lovasz_softmax_flat as lovz

# TEST = True
TEST = False

BASE_DIR = 'D:/SemanticKITTI/dataset/sequences'
seq_list = np.sort(listdir(BASE_DIR))
split_params = safe_load(open('semantic-kitti.yaml', 'r'))['split']
if TEST:
    te_seq_list = list(seq_list[split_params['test']])
else:
    te_seq_list = list(seq_list[split_params['train']])
data_te = PCGraph(BASE_DIR, seq_no='00', stop_idx=1, test=TEST, transforms=LayerPreprocess(GCNConv))

def test(model_path='models/infer_v1_2'):
    loaded = load_model(filepath=model_path, compile=False)
    loader_te = DisjointLoader(dataset=data_te, batch_size=1, node_level=True, shuffle=False)
    i = 0
    if TEST:
        for batch in loader_te:
            i+=1
            x, adj, _ = batch
            I_in = np.array(np.arange(x.shape[0]))
            predictions = loaded([x, adj, I_in], training=False)
            pred_labels = np.argmax(predictions, axis=-1)
            ShowPC.draw_pc_sem_ins(pc_xyz=x[:, :3], pc_sem_ins=pred_labels)
            if i == loader_te.steps_per_epoch:
                break
    else:
        test_miou_obj = MeanIoU(name='test_miou', num_classes=20)
        # miou_list = []
        for batch in loader_te:
            i += 1
            test_miou_obj.reset_states()
            x, adj, _ = batch[0]
            y = np.squeeze(batch[1])
            I_in = np.array(np.arange(x.shape[0]))
            predictions = loaded([x, adj, I_in], training=False)

            print(lovz(predictions, y).numpy())

            pred_labels = np.argmax(predictions, axis=-1)
            test_miou_obj(y, pred_labels)
            curr_miou = test_miou_obj.result().numpy() * 100
            print('Mean IoU: ', curr_miou)
            ShowPC.draw_pc_sem_ins(pc_xyz=x[:, :3], pc_sem_ins=pred_labels)
            if i == loader_te.steps_per_epoch:
                break

test()


