import os
import datetime
import pymsteams
import numpy as np
from time import time
import tensorflow as tf
from tensorflow.keras.optimizers import Adam

from models import PointSegGCN as network
from layers import CyclicalLR
from utils import readers as io, loss_metrics
from utils.jaccard import iouEval
from utils.preprocess import preprocess

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)


def assign_loss_func(name):
    '''
    Assigns loss function to the training loop
    :param name: loss function name, specified in the config file
    :return: loss object which can be used directly in the train step
    '''
    loss_dict = {'lovasz': loss_metrics.lovasz_softmax_flat,
                 'focal_tv': loss_metrics.focal_tversky_loss,
                 'tversky': loss_metrics.tversky_loss,
                 'sparse_ce': loss_metrics.sparse_cross_entropy,
                 'sparse_focal': loss_metrics.sparse_categorical_focal_loss}

    return loss_dict.get(str(name))


def train_step(inputs, model, optimizer, miou_obj, loss_obj):
    '''
    Carries out a forward pass, loss computation, and backpropagation on a single training batch
    :param inputs: tuple of preprocessed model inputs
    :param model: compiled TF model object
    :param optimizer: TF Keras optimizer object which carries out backpropagation
    :param miou_obj: mIoU object for training dataset
    :param loss_obj: Loss computation object
    :return: training mIoU and Loss values
    '''
    X, A, Y, = inputs

    with tf.GradientTape() as tape:
        predictions = model([X, A], training=True)
        tr_loss = loss_obj(Y, predictions)

    gradients = tape.gradient(tr_loss, model.trainable_variables)

    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    pred_labels = np.argmax(predictions, axis=-1)
    miou_obj.addBatch(pred_labels, Y)
    tr_miou, _ = miou_obj.getIoU()
    tr_loss = np.mean(tr_loss)

    return tr_loss, tr_miou


def evaluate(inputs, model, miou_obj, loss_obj):
    '''
    Carries out a forward pass (without training) on a single evaluation batch
    :param inputs: tuple of preprocessed model inputs
    :param model: compiled TF model object
    :param miou_obj: mIoU object for validation dataset
    :param loss_obj: Loss computation object
    :return: validation mIoU and Loss values
    '''
    va_output = []

    X, A, Y = inputs

    predictions = model([X, A], training=False)
    pred_labels = np.argmax(predictions, axis=-1)

    va_loss = loss_obj(Y, predictions)

    miou_obj.addBatch(pred_labels, Y)
    va_miou, _ = miou_obj.getIoU()
    va_loss = np.mean(va_loss)

    va_output.append([va_loss, va_miou])

    outs_avg = np.mean(va_output, 0)
    outs_arr = np.array(outs_avg).flatten()

    return outs_arr[0], outs_arr[1]


def train(model,
          cfg,
          train_files,
          train_miou_obj,
          val_files,
          val_miou_obj,
          opt,
          loss_func,
          **kwargs):
    '''
    Umbrella training loop performing train_step, evaluation, and metric logging for the preprocessed dataset
    :param model: compiled TF model object
    :param cfg: parsed training config param file
    :param train_files: list of training file paths
    :param train_miou_obj: mIoU object for training set
    :param val_files: list of validation file paths
    :param val_miou_obj: mIoU object for validation set
    :param opt: TF Keras optimizer object
    :param loss_func: Loss computation object
    :param kwargs: additional named arguments, e.g. training and validation performance loggers for TensorBoard
    :return: Trained TF Keras model object
    '''
    tr_loss = tr_miou = va_loss = va_miou = 0

    for epoch in range(cfg['epochs']):

        start = time()
        # Training step
        for tr_file in train_files:
            tr_inputs = preprocess(tr_file, cfg, True)
            tr_loss, tr_miou = train_step(tr_inputs, model=model, optimizer=opt,
                                          miou_obj=train_miou_obj, loss_obj=loss_func)

        # Validation step
        for va_file in val_files:
            va_inputs = preprocess(va_file, cfg)
            va_loss, va_miou = evaluate(va_inputs, model=model, miou_obj=val_miou_obj, loss_obj=loss_func)

        # Write scalars to log for Tensorboard evaluation
        with kwargs['train_logger'].as_default():
            tf.summary.scalar('loss', tr_loss, step=epoch)
            tf.summary.scalar('mIoU', tr_miou * 100, step=epoch)

        with kwargs['valid_logger'].as_default():
            tf.summary.scalar('loss', va_loss, step=epoch)
            tf.summary.scalar('mIoU', va_miou * 100, step=epoch)

        print('Elapsed for epoch {} : {} s'.format(epoch + 1, time() - start))

        curr_stats = ('Epoch: {} ||| Train loss: {:.4f}, Train MeanIoU: {:.4f} | '
                      + 'Valid loss: {:.4f}, Valid MeanIoU: {:.4f} ||| ').format(epoch + 1,
                                                                                 tr_loss, tr_miou * 100,
                                                                                 va_loss, va_miou * 100)
        # Print current epoch stats to console
        print(curr_stats)

        tr_loss = 0
        tr_miou = 0

        if (int(epoch + 1) % 1 == 0):

            if cfg['send_stats_teams']:
                # Relay current epoch stats to preconfigured MS Teams channel (see cfg)
                kwargs['teamshook'].text(curr_stats)
                kwargs['teamshook'].send()


if __name__ == '__main__':

    # Parse necessary training parameters from .yaml config files into a dictionary
    cfg = io.get_cfg_params()

    # Build and compile model based on config parameters
    model = network(cfg)

    print(model.summary())

    # Initialize loggers
    tr_start_time = datetime.datetime.now().strftime("%Y-%m-%d--%H.%M.%S")
    train_log_dir = 'TB_logs/' + tr_start_time + '/train'
    valid_log_dir = 'TB_logs/' + tr_start_time + '/valid'
    train_logger = tf.summary.create_file_writer(train_log_dir)
    valid_logger = tf.summary.create_file_writer(valid_log_dir)
    teamshook = pymsteams.connectorcard(cfg['teams_hook'])

    if cfg['trial']:
        train_files, val_files, _ = io.get_split_files(cfg, shuffle=True)
        train_files = train_files[:cfg['trial_size']]
        val_files = val_files[:int(cfg['trial_size'] / 10)]
    elif cfg['fwd_pass_check']:
        # Sanity check - run train and eval step on single example
        train_files = [cfg['fwd_pass_sample']]
        val_files = train_files
    else:
        train_files, val_files, _ = io.get_split_files(cfg, shuffle=True)

    # Cyclical learning rate scheduler
    lr_schedule = CyclicalLR(base_lr=cfg['learning_rate'], max_lr=0.1)

    optimizer = Adam(learning_rate=lr_schedule)
    num_classes = cfg['num_classes']
    class_ignore = cfg["class_ignore"]
    train_miou_obj = iouEval(num_classes, class_ignore)
    val_miou_obj = iouEval(num_classes, class_ignore)
    loss_func = assign_loss_func(cfg['loss_fn'])

    print('----------------------------------------------------------------------------------')
    print('     TRAINING START...')
    print('----------------------------------------------------------------------------------')

    train(model=model,
          cfg=cfg,
          train_files=train_files,
          train_logger=train_logger,
          train_miou_obj=train_miou_obj,
          val_files=val_files,
          valid_logger=valid_logger,
          val_miou_obj=val_miou_obj,
          opt=optimizer,
          loss_func=loss_func,
          teamshook=teamshook)

    print('----------------------------------------------------------------------------------')
    print('     TRAINING END...')
    print('----------------------------------------------------------------------------------')

    if cfg['save_model']:
        save_dir = os.path.join('models', cfg['model_name'])
        save_path = save_dir + '.json'
        # Save trained model weights to .h5 file
        model.save_weights(save_dir + '.h5')
        model_json = model.to_json()
        with open(save_dir + '.json', "w") as json_file:
            # Save trained model architecture to .json file
            json_file.write(model_json)
        print('     Model saved to {}'.format(save_path))
        print('==================================================================================')
