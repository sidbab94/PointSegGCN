import datetime
from time import time
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import pymsteams
import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam

from utils.jaccard import iouEval
from models import Dense_GCN as network
from layers import CyclicalLR

from utils import readers as io, loss_metrics
from utils.preprocess import preprocess

np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)


def assign_loss_func(name):
    loss_dict = {'lovasz': loss_metrics.lovasz_softmax_flat,
                 'focal_tv': loss_metrics.focal_tversky_loss,
                 'tversky': loss_metrics.tversky_loss,
                 'sparse_ce': loss_metrics.sparse_cross_entropy,
                 'sparse_focal': loss_metrics.sparse_categorical_focal_loss}

    return loss_dict.get(str(name))


def train_step(inputs, model, optimizer, miou_obj, loss_obj):
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


def train():

    for epoch in range(cfg['epochs']):

        start = time()
        for tr_file in train_files:
            tr_inputs = preprocess(tr_file, cfg, True)
            tr_loss, tr_miou = train_step(tr_inputs, model=model, optimizer=opt,
                                          miou_obj=train_miou_obj, loss_obj=loss_func)

        print('Elapsed for epoch {} : {} s'.format(epoch + 1, time() - start))

        for va_file in val_files:
            va_inputs = preprocess(va_file, cfg)
            va_loss, va_miou = evaluate(va_inputs, model=model, miou_obj=val_miou_obj, loss_obj=loss_func)

        # Write scalars to log for Tensorboard evaluation
        with train_summary_writer.as_default():
            tf.summary.scalar('loss', tr_loss, step=epoch)
            tf.summary.scalar('mIoU', tr_miou * 100, step=epoch)

        with summary_writer.as_default():
            tf.summary.scalar('learning_rate',
                              opt._decayed_lr(tf.float32),
                              step=epoch)

        with valid_summary_writer.as_default():
            tf.summary.scalar('loss', va_loss, step=epoch)
            tf.summary.scalar('mIoU', va_miou * 100, step=epoch)

        curr_stats = ('Epoch: {} ||| Train loss: {:.4f}, Train MeanIoU: {:.4f} | '
                      + 'Valid loss: {:.4f}, Valid MeanIoU: {:.4f} ||| ').format(epoch + 1,
                                                                                 tr_loss, tr_miou * 100,
                                                                                 va_loss, va_miou * 100)

        print(curr_stats)

        ckpt.step.assign_add(1)
        tr_loss = 0
        tr_miou = 0

        if (int(ckpt.step) % 1 == 0):

            if cfg['send_stats_teams']:
                teamshook.text(curr_stats)
                teamshook.send()

            print('----------------------------------------------------------------------------------')

            if (va_miou > np.max(list_mious)) and cfg['ckpt_save']:
                save_path = manager.save()
                weights_path = './ckpt_weights/' + tr_start_time
                print("Saved checkpoint for step {}: {}".format(int(ckpt.step), save_path))
                print("Saved weights for step {}: {}".format(int(ckpt.step), weights_path))
                model.save_weights(weights_path, overwrite=True)

            print('----------------------------------------------------------------------------------')

        list_mious.append(va_miou)


if __name__ == '__main__':

    cfg = io.get_cfg_params()

    model = network(cfg)
    # model = PointNet2(1, cfg['num_classes'])

    print(model.summary())

    tr_start_time = datetime.datetime.now().strftime("%Y-%m-%d--%H.%M.%S")
    train_log_dir = 'TB_logs/' + tr_start_time + '/train'
    valid_log_dir = 'TB_logs/' + tr_start_time + '/valid'
    gen_log_dir = 'TB_logs/' + tr_start_time + '/general'
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)
    valid_summary_writer = tf.summary.create_file_writer(valid_log_dir)
    summary_writer = tf.summary.create_file_writer(gen_log_dir)

    teamshook = pymsteams.connectorcard(cfg['teams_hook'])

    num_classes = cfg['num_classes']
    class_ignore = cfg["class_ignore"]
    train_miou_obj = iouEval(num_classes, class_ignore)
    val_miou_obj = iouEval(num_classes, class_ignore)

    tr_loss = tr_miou = va_loss = va_miou = 0
    list_mious = [0.0]
    loss_func = assign_loss_func(cfg['loss_fn'])

    if cfg['trial']:
        train_files, val_files, _ = io.get_split_files(cfg, shuffle=True)
        train_files = train_files[:cfg['trial_size']]
        val_files = val_files[:int(cfg['trial_size'] / 10)]
    elif cfg['fwd_pass_check']:
        train_files = [cfg['fwd_pass_sample']]
        val_files = [cfg['fwd_pass_sample']]

    lr_schedule = CyclicalLR(base_lr=0.01, max_lr=0.1)

    opt = Adam(learning_rate=lr_schedule)

    ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimizer=opt)
    manager = tf.train.CheckpointManager(ckpt, './tf_ckpts', max_to_keep=2)

    if cfg['ckpt_restore']:
        ckpt.restore(manager.latest_checkpoint)
        if manager.latest_checkpoint:
            print("     Restored from checkpoint: {}".format(manager.latest_checkpoint))

    print('----------------------------------------------------------------------------------')
    print('     TRAINING START...')
    print('----------------------------------------------------------------------------------')

    train()

    print('----------------------------------------------------------------------------------')
    print('     TRAINING END...')
    print('----------------------------------------------------------------------------------')

    if cfg['save_model']:
        save_dir = os.path.join('models', cfg['model_name'])
        save_path = os.path.join(save_dir, '.json')
        model.save_weights(save_dir + '.h5')
        model_json = model.to_json()
        with open(save_dir + '.json', "w") as json_file:
            json_file.write(model_json)
        print('     Model saved to {}'.format(save_path))
        print('==================================================================================')
