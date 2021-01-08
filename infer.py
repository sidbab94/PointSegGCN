import os
from pathlib import Path
import tensorflow as tf
from tensorflow.keras.models import load_model
import random

from train_utils.eval_metrics import iouEval
from model import res_model_1 as Net

from preprocess import *
from visualization import PC_Vis

def test_single(test_file, loaded_model, cfg, prep_obj):
    x, a, y = prep_obj.assess_scan(test_file)

    predictions = loaded_model.predict_step([x, a])
    pred_labels = np.argmax(predictions, axis=-1)

    print(np.unique(y), np.unique(pred_labels))

    class_ignore = cfg["class_ignore"]
    test_miou = iouEval(len(class_ignore), class_ignore)

    test_miou.addBatch(pred_labels, y)
    te_miou, iou = test_miou.getIoU()
    print('Mean IoU: ', te_miou * 100)
    print('IoU: ', iou * 100)

    PC_Vis.eval(pc=x, y_true=y, cfg=cfg, y_pred=pred_labels)


if __name__ == '__main__':
    BASE_DIR = 'D:/SemanticKITTI/dataset/sequences'

    load_frm_model = True
    # cfg = safe_load(open('config/semantic-kitti.yaml', 'r'))

    cfg = get_cfg_params(base_dir=BASE_DIR)

    train_files, val_files, test_files = get_split_files(dataset_path=BASE_DIR, cfg=cfg)
    test_file = random.choice(val_files)
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
        load_status = loaded_model.load_weights('./ckpt_weights/2021-01-04--06.08.13')
        # load_status.assert_consumed()
        print(loaded_model)

    test_single(test_file, loaded_model, cfg, prep)






