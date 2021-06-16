import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import random
import tensorflow as tf
from tensorflow.keras.models import model_from_json

from utils.func_timer import timing
from utils.jaccard import iouEval
from layers import GConv
from utils.visualization import PC_Vis
from utils.readers import get_cfg_params, get_split_files
from utils.preprocess import preprocess



def map_iou(y_true, y_pred, cfg):

    class_ignore = cfg["class_ignore"]

    test_miou = iouEval(len(class_ignore), class_ignore)

    test_miou.addBatch(y_pred, y_true)
    miou, iou = test_miou.getIoU()

    valid_y = np.unique(y_true)

    label_list = list(cfg['labels'].keys())

    valid_cls = {k: v for k, v in class_ignore.items() if v == False}

    print('-----------------------')
    for i in range(cfg['num_classes']):
        if i in valid_y:
            curr_class = cfg['labels'][label_list[i]].capitalize()
            if i in valid_cls.keys():
                print('IoU for class {} -- {}   :   {}'.format(i, curr_class, round(iou[i] * 100, 2)))

    print('\n *** Class present in Ground Truth ***')


    print('-----------------------')
    print('Mean IoU: ', round(miou * 100, 2))
    print('-----------------------')


@tf.function
def infer(model, inputs):

    predictions = model.predict_step([inputs[0], inputs[1]])

    return predictions

def load_saved_model(cfg):

    model_path = os.path.join('models', cfg['model_name'])
    json_file = open(model_path + '.json', 'r')
    loaded_model_json = json_file.read()
    loaded_model = model_from_json(loaded_model_json, custom_objects={'GConv': GConv})
    loaded_model.load_weights(model_path + '.h5')
    json_file.close()

    return loaded_model

@timing
def test_single(file=None, test_run=False):

    cfg = get_cfg_params()
    loaded_model = load_saved_model(cfg)

    if file is None:
        train_files, val_files, test_files = get_split_files(cfg=cfg)
        test_file = random.choice(val_files)
        print('No path provided, performing random inference on: ', test_file)
    else:
        test_file = file

    x, a, y = preprocess(test_file, cfg, test_run)
    predictions = infer(loaded_model, [x, a])

    pred_labels = np.argmax(predictions, axis=-1)

    map_iou(y, pred_labels, cfg)

    PC_Vis.eval(pc=x, y_true=y, cfg=cfg,
                y_pred=pred_labels, gt_colour=False)

if __name__ == '__main__':

    # file = 'D:/SemanticKITTI/dataset/sequences/08/velodyne/002989.bin'
    file = './samples/pc.bin'

    test_single(file, True)