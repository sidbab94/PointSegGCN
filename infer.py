import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from pathlib import Path
import random
import tensorflow as tf
from tensorflow.keras.models import load_model
from spektral.layers.ops import sp_matrix_to_sp_tensor

from preprocess import *
from time import time
from train_utils.eval_metrics import iouEval
from visualization import PC_Vis
from model import Res_GCN_v3 as network


def test_all(FLAGS):
    '''
    Performs inference on all files provided as input, displays mean IoU averaged over all samples

    :param FLAGS: Argument parser flags provided as input
    :return: None
    '''

    model_cfg = get_cfg_params(base_dir=FLAGS.dataset, train_cfg=FLAGS.trconfig, dataset_cfg=FLAGS.datacfg)
    prep = Preprocess(model_cfg)

    if FLAGS.ckpt:
        loaded_model = network(model_cfg)
        latest_checkpoint = tf.train.latest_checkpoint('./ckpt_weights')
        load_status = loaded_model.load_weights(latest_checkpoint)
        load_status.assert_consumed()
        print('Model deserialized and loaded from: ', latest_checkpoint)
    else:
        if FLAGS.model is None:
            latest_model_path = sorted(Path('./models').iterdir(), key=os.path.getmtime)[-1]
            loaded_model = load_model(filepath=latest_model_path, compile=False)
            print('No path provided. Latest saved model loaded from: ', latest_model_path)
        else:
            loaded_model = load_model(filepath=FLAGS.model, compile=False)

    _, val_files, _ = get_split_files(dataset_path=FLAGS.dataset, cfg=model_cfg)

    overall_val_miou = 0.0
    inf_times = []
    val_count = len(val_files)

    class_ignore = model_cfg["class_ignore"]
    test_miou = iouEval(len(class_ignore), class_ignore)

    for file in val_files:
        print('Processing: ', file)
        start = time()
        x, a, y = prep.assess_scan(file)
        a = sp_matrix_to_sp_tensor(a)
        predictions = loaded_model.predict_step([x, a])
        inf_times.append(time() - start)
        pred_labels = np.argmax(predictions, axis=-1)

        test_miou.addBatch(pred_labels, y)
        te_miou, iou = test_miou.getIoU()

        overall_val_miou += te_miou * 100

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


def test_single(FLAGS):
    '''
    Performs inference on single file provided as input, optionally visualizes output

    :param FLAGS: Argument parser flags provided as input
    :return: None
    '''

    model_cfg = get_cfg_params(base_dir=FLAGS.dataset, train_cfg=FLAGS.trconfig, dataset_cfg=FLAGS.datacfg)
    prep = Preprocess(model_cfg)

    if FLAGS.ckpt:
        loaded_model = network(model_cfg)
        latest_checkpoint = tf.train.latest_checkpoint('./ckpt_weights')
        load_status = loaded_model.load_weights(latest_checkpoint)
        load_status.assert_consumed()
        print('Model deserialized and loaded from: ', latest_checkpoint)
    else:
        if FLAGS.model is None:
            latest_model_path = sorted(Path('./models').iterdir(), key=os.path.getmtime)[-1]
            loaded_model = load_model(filepath=latest_model_path, compile=False)
            print('No path provided. Latest saved model loaded from: ', latest_model_path)
        else:
            loaded_model = load_model(filepath=FLAGS.model, compile=False)

    if FLAGS.file is None:
        train_files, val_files, test_files = get_split_files(dataset_path=FLAGS.dataset, cfg=model_cfg)
        test_file = random.choice(val_files)
        print('No path provided, performing random inference on: ', test_file)
    else:
        test_file = FLAGS.file


    start = time()
    x, a, y = prep.assess_scan(test_file)
    a = sp_matrix_to_sp_tensor(a)
    predictions = loaded_model.predict_step([x, a])
    print('Elapsed: ', time() - start)
    pred_labels = np.argmax(predictions, axis=-1)

    map_iou(y, pred_labels, model_cfg)

    if FLAGS.vis:
        PC_Vis.eval(pc=x, y_true=y, cfg=model_cfg, y_pred=pred_labels)

