import os
from pathlib import Path
import numpy as np
import random
import tensorflow as tf
from tensorflow.keras.models import load_model
from yaml import safe_load

from preproc_utils.dataprep import get_split_files, get_cfg_params
from batch_gen import process_single
from train_utils.eval_metrics import iouEval
from visualization import ShowPC
from vis_aux.visualize import ScanVis
from model import res_model_1 as Net


def test_single(test_file, loaded_model, cfg):
    x, a, y = process_single(test_file, cfg)

    # predictions = loaded_model([x, a], training=False)
    predictions = loaded_model.predict_step([x, a])
    pred_labels = np.argmax(predictions, axis=-1)

    print(np.unique(y), np.unique(pred_labels))

    class_ignore = cfg["learning_ignore"]
    test_miou = iouEval(len(class_ignore), class_ignore)

    test_miou.addBatch(pred_labels, y)
    te_miou, iou = test_miou.getIoU()
    print('Mean IoU: ', te_miou * 100)
    print('IoU: ', iou * 100)

    # ShowPC.draw_pc_sem_ins(pc_xyz=x[:, :3], pc_sem_ins=pred_labels)
    vis_obj = ScanVis(config=cfg, scan_file=test_file, orig_labels=y, pred_labels=pred_labels)
    vis_obj.run()

if __name__ == '__main__':
    BASE_DIR = 'D:/SemanticKITTI/dataset/sequences'

    load_frm_model = False
    cfg = safe_load(open('config/semantic-kitti.yaml', 'r'))

    tr_dict = get_cfg_params(base_dir=BASE_DIR)

    train_files, val_files, test_files = get_split_files(dataset_path=BASE_DIR, cfg=tr_dict, count=5)
    # test_file = random.choice(val_files)
    test_file = val_files[0]
    print(test_file)

    if load_frm_model:
        latest_model_path = sorted(Path('./models').iterdir(), key=os.path.getmtime)[-1]
        print(latest_model_path)
        loaded_model = load_model(filepath=latest_model_path, compile=False)

    else:
        loaded_model = Net(tr_dict)
        latest_checkpoint = tf.train.latest_checkpoint('./ckpt_weights')
        print(latest_checkpoint)
        load_status = loaded_model.load_weights('./ckpt_weights/2021-01-04--06.08.13')
        # load_status.assert_consumed()
        print(loaded_model)

    test_single(test_file, loaded_model, cfg)






