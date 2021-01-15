import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from time import time
from pathlib import Path
import tensorflow as tf
from tensorflow.keras.models import load_model
import random
from spektral.layers.ops import sp_matrix_to_sp_tensor
from train_utils.eval_metrics import iouEval
from model import Res_GCN_v1 as Net

from preprocess import *
from visualization import PC_Vis


def test_all(val_files, loaded_model, cfg, prep_obj):
    '''
    Performs inference on all files provided as input, displays mean IoU averaged over all samples

    :param val_files: list of scan files to perform evaluation on
    :param loaded_model: loaded and serialized model
    :param cfg: model config dict
    :param prep_obj: preprocessor object
    :return: None
    '''

    overall_val_miou = 0.0
    inf_times = []
    val_count = len(val_files)

    class_ignore = cfg["class_ignore"]
    test_miou = iouEval(len(class_ignore), class_ignore)


    for file in val_files:

        print('Processing: ', file)
        start = time()
        x, a, y = prep_obj.assess_scan(file)
        a = sp_matrix_to_sp_tensor(a)
        predictions = loaded_model.predict_step([x, a])
        inf_times.append(time() - start)
        pred_labels = np.argmax(predictions, axis=-1)

        test_miou.addBatch(pred_labels, y)
        te_miou, iou = test_miou.getIoU()

        overall_val_miou += te_miou*100

    overall_val_miou = overall_val_miou / val_count
    avg_inf_time = np.mean(inf_times)

    print('mIoU, averaged over all validation samples: ', overall_val_miou)
    print('Average inference time in seconds: ', avg_inf_time)


def map_iou(y_true, y_pred, cfg):
    '''
    Maps IoU for each class in prediction array to its corresponding 'valid' class in ground-truth

    :param y_true: point-wise ground truth labels
    :param y_pred: point-wise predicted labels
    :param cfg: model config dict
    :return: None
    '''

    class_ignore = cfg["class_ignore"]
    test_miou = iouEval(len(class_ignore), class_ignore)

    test_miou.addBatch(y_pred, y_true)
    te_miou, iou = test_miou.getIoU()

    iou_dict = dict(zip(range(0, cfg['num_classes']), (iou * 100)))

    valid_y = np.unique(y_true)

    # label_list = list(cfg['labels'].keys())
    label_list = list(cfg['learning_map_inv'].values())

    print('-----------------------')
    for i in range(cfg['num_classes']):
        curr_class = cfg['labels'][label_list[i]]
        if i in valid_y:
            curr_class = ' *** ' + curr_class + ' *** '

        print('IoU for class {} -- {}   :   {}'.format(i, curr_class, round(iou_dict[i], 2)))

    print('\n *** Class present in Ground Truth ***')

    print('-----------------------')
    print('Mean IoU: ', te_miou * 100)
    print('-----------------------')


def test_single(test_file, loaded_model, cfg, prep_obj, vis=False, iou_detail=True):
    '''

    :param test_file: scan file to perform inference on
    :param loaded_model: loaded and serialized model
    :param cfg: model config dict
    :param prep_obj: preprocessor object
    :param vis: boolean flag for comparative visualization (GT vs. Pred) using Open3D
    :return:
    '''

    x, a, y = prep_obj.assess_scan(test_file)
    a = sp_matrix_to_sp_tensor(a)
    predictions = loaded_model.predict_step([x, a])
    pred_labels = np.argmax(predictions, axis=-1)

    if iou_detail:
        map_iou(y, pred_labels, cfg)

    if vis:
        PC_Vis.eval(pc=x, y_true=y, cfg=cfg, y_pred=pred_labels)


if __name__ == '__main__':
    BASE_DIR = 'D:/SemanticKITTI/dataset/sequences'

    load_frm_model = True
    # cfg = safe_load(open('config/semantic-kitti.yaml', 'r'))

    cfg = get_cfg_params(base_dir=BASE_DIR)

    train_files, val_files, test_files = get_split_files(dataset_path=BASE_DIR, cfg=cfg)
    test_file = random.choice(val_files)
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
