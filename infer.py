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



@tf.function
def predict(model, inputs):

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
def test_single(test_file, loaded_model, cfg):

    x, a, y = preprocess(test_file, cfg)
    predictions = predict(loaded_model, [x, a])

    pred_labels = np.argmax(predictions, axis=-1)

    return x, y, pred_labels


def inference(test_file='/media/baburaj/Seagate Backup Plus Drive/SemanticKITTI/dataset/sequences/08/velodyne/003638.bin'):

    cfg = get_cfg_params()
    class_ignore = cfg["class_ignore"]
    label_list = cfg['learning_label_map']
    test_miou = iouEval(len(class_ignore), class_ignore)
    loaded_model = load_saved_model(cfg)


    if cfg['infer_all']:

        IOU = np.zeros((20,), dtype=np.float32)

        _, val_files, _ = get_split_files(cfg=cfg)

        for file in val_files:

            x, y, y_pred = test_single(file, loaded_model, cfg)
            test_miou.addBatch(y_pred, y)

            _, iou = test_miou.getIoU()
            IOU += iou

        IOU /= len(val_files)

        print('==================================')
        print('Complete evaluation (4070 samples)')
        print('==================================')
        for i in range(cfg['num_classes']):
            curr_class = label_list[i].capitalize()
            print('IoU for class {} -- {}   :   {}'.format(i, curr_class, round(IOU[i] * 100, 2)))
        print('-----------------------')
        MIOU = np.mean(IOU[1:])
        print('Mean IoU: ', round(MIOU * 100, 2))
        print('-----------------------')

    else:

        if test_file is None:
            if cfg['fwd_pass_check']:
                test_file = cfg['fwd_pass_sample']
            else:
                train_files, val_files, test_files = get_split_files(cfg=cfg)
                test_file = random.choice(val_files)
                print('No path provided, performing random inference on: ', test_file)

        x, y, y_pred = test_single(test_file, loaded_model, cfg)
        test_miou.addBatch(y_pred, y)
        miou, iou = test_miou.getIoU()

        print('==================================')
        print('   Single evaluation (1 sample)   ')
        print('==================================')
        for i in range(cfg['num_classes']):
            if i in np.unique(y):
                curr_class = label_list[i].capitalize()
                print('IoU for class {} -- {}   :   {}'.format(i, curr_class, round(iou[i] * 100, 2)))
        print('-----------------------')
        print('Mean IoU: ', round(miou * 100, 2))
        print('-----------------------')

        if cfg['infer_vis']:
            PC_Vis.eval(pc=x, y_true=y, cfg=cfg,
                        y_pred=y_pred, gt_colour=False)





if __name__ == '__main__':

    inference()