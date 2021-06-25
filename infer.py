import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import random
import tensorflow as tf
from tensorflow.keras.models import model_from_json, load_model

from utils.func_timer import timing
from utils.jaccard import iouEval
from layers import GConv, ConcatAdj
from utils.visualization import PC_Vis
from utils.readers import get_cfg_params, get_split_files
from utils.preprocess import preprocess



def map_iou(y_true, y_pred, cfg):

    class_ignore = cfg["class_ignore"]

    test_miou = iouEval(len(class_ignore), class_ignore)

    test_miou.addBatch(y_pred, y_true)
    miou, iou = test_miou.getIoU()

    valid_y = np.unique(y_true)

    # label_list = list(cfg['labels'].keys())
    label_list = list(cfg['learning_label_map'].keys())
    print(label_list)

    valid_cls = {k: v for k, v in class_ignore.items() if v == False}

    print('-----------------------')
    for i in range(cfg['num_classes']):
        if i in valid_y:
            # curr_class = cfg['labels'][label_list[i]].capitalize()
            curr_class = cfg['learning_label_map'][i].capitalize()
            if i in valid_cls.keys():
                print('IoU for class {} -- {}   :   {}'.format(i, curr_class, round(iou[i] * 100, 2)))

    print('\n *** Class present in Ground Truth ***')


    print('-----------------------')
    print('Mean IoU: ', round(miou * 100, 2))
    print('-----------------------')


@timing
def infer(model, inputs):

    predictions = model.predict_step([*inputs])

    return predictions

@timing
def load_saved_model(cfg, prev=False):

    if prev:
        latest_model_path = 'models/infer_v4_0_DeepGCN_xyzirgb_nn10_200_bs4_cce_lov_aug'
        loaded_model = load_model(filepath=latest_model_path, compile=False)
        print('Model deserialized and loaded from: ', loaded_model)

    else:
        model_path = os.path.join('models', cfg['model_name'])
        json_file = open(model_path + '.json', 'r')
        loaded_model_json = json_file.read()
        loaded_model = model_from_json(loaded_model_json,
                                       custom_objects={'GConv': GConv, 'ConcatAdj': ConcatAdj})
        loaded_model.load_weights(model_path + '.h5')
        json_file.close()

    print('Model loaded.')

    return loaded_model


@timing
def test_single(file=None):

    cfg = get_cfg_params()
    loaded_model = load_saved_model(cfg, True)

    if file is None:
        if cfg['fwd_pass_check']:
            test_file = cfg['fwd_pass_sample']
        else:
            train_files, val_files, test_files = get_split_files(cfg=cfg)
            test_file = random.choice(val_files)
            print('No path provided, performing random inference on: ', test_file)
    else:
        test_file = file

    x, a, y = preprocess(test_file, cfg)
    predictions = infer(loaded_model, [x, a])

    pred_labels = np.argmax(predictions, axis=-1)

    map_iou(y, pred_labels, cfg)

    PC_Vis.eval(pc=x, y_true=y, cfg=cfg,
                y_pred=pred_labels, gt_colour=False)



if __name__ == '__main__':

    test_single()