from tensorflow.keras.models import load_model
import numpy as np
from yaml import safe_load
from preproc_utils.dataprep import get_split_files
from batch_gen import process_single
from train_utils.eval_met import iouEval
from visualization import ShowPC

semkitti_cfg = safe_load(open('config/semantic-kitti.yaml', 'r'))
BASE_DIR = 'D:/SemanticKITTI/dataset/sequences'
class_ignore = semkitti_cfg["learning_ignore"]
ignore = []
for cl, ign in class_ignore.items():
    if ign:
        x_cl = int(cl)
        ignore.append(x_cl)
        print("     Ignoring cross-entropy class ", x_cl, " in IoU evaluation")
test_miou = iouEval(20, ignore)

train_files, val_files, test_files = get_split_files(dataset_path=BASE_DIR, cfg=semkitti_cfg["split"], count=1)
test_file = train_files[0]

def test(test_file, model_path='models/infer_v1_8'):
    loaded = load_model(filepath=model_path, compile=False)
    x, a, y = process_single(test_file)
    predictions = loaded([x, a], training=False)
    pred_labels = np.argmax(predictions, axis=-1)
    print(np.unique(y), np.unique(pred_labels))
    test_miou.addBatch(pred_labels, y)
    te_miou, iou = test_miou.getIoU()
    print('Mean IoU: ', te_miou * 100)
    print('IoU: ', iou * 100)
    ShowPC.draw_pc_sem_ins(pc_xyz=x[:, :3], pc_sem_ins=pred_labels)

test(test_file)
