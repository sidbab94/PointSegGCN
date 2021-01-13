import os
from pathlib import Path
import tensorflow as tf
from tensorflow.keras.models import load_model
import random
from spektral.layers.ops import sp_matrix_to_sp_tensor
from train_utils.eval_metrics import iouEval
from model import res_model_2 as Net

from preprocess import *
from visualization import PC_Vis


def test_all(val_files, loaded_model, cfg, prep_obj):

    overall_val_miou = 0.0
    val_count = len(val_files)

    for file in val_files:

        curr_miou = test_single(file, loaded_model, cfg, prep_obj)

        overall_val_miou += curr_miou

    overall_val_miou = overall_val_miou/val_count

    print('mIoU, averaged across all validation samples: ', overall_val_miou)



def test_single(test_file, loaded_model, cfg, prep_obj, vis=False):
    x, a, y = prep_obj.assess_scan(test_file)
    a = sp_matrix_to_sp_tensor(a)
    predictions = loaded_model.predict_step([x, a])
    pred_labels = np.argmax(predictions, axis=-1)

    # print(np.unique(y), np.unique(pred_labels))

    class_ignore = cfg["class_ignore"]
    test_miou = iouEval(len(class_ignore), class_ignore)

    test_miou.addBatch(pred_labels, y)
    te_miou, iou = test_miou.getIoU()

    if vis:
        PC_Vis.eval(pc=x, y_true=y, cfg=cfg, y_pred=pred_labels)
    print('Mean IoU: ', te_miou * 100)
    print('IoU: ', iou * 100)

    return te_miou


if __name__ == '__main__':
    BASE_DIR = 'D:/SemanticKITTI/dataset/sequences'

    load_frm_model = True
    # cfg = safe_load(open('config/semantic-kitti.yaml', 'r'))

    cfg = get_cfg_params(base_dir=BASE_DIR)

    train_files, val_files, test_files = get_split_files(dataset_path=BASE_DIR, cfg=cfg)
    test_file = random.choice(train_files[:200])
    # test_file = val_files[0]
    print(test_file)

    prep = Preprocess(cfg)

    if load_frm_model:
        latest_model_path = sorted(Path('./models').iterdir(), key=os.path.getmtime)[-1]
        print(latest_model_path)
        loaded_model = load_model(filepath=latest_model_path, compile=False)

    else:
        loaded_model = Net(cfg)
        latest_checkpoint = tf.train.latest_checkpoint('./ckpt_weights')
        print(latest_checkpoint)
        load_status = loaded_model.load_weights(latest_checkpoint)
        # load_status.assert_consumed()
        print(loaded_model)

    test_single(test_file, loaded_model, cfg, prep, vis=True)
    # test_all(val_files, loaded_model, cfg, prep)






