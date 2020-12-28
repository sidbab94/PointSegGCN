import numpy as np
import datetime
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import ExponentialDecay

from preproc_utils.dataprep import get_split_files, get_cfg_params
from train_utils.lovasz_loss import lovasz_softmax_flat
from batch_gen import process_single
from train_utils.eval_met import iouEval
from model import res_model_1 as network


def train_step(batch, lovasz=False):

    inputs = [batch[0], batch[1]]
    target = batch[2]

    with tf.GradientTape() as tape:
        predictions = model(inputs, training=True)
        if lovasz:
            tr_loss = lovasz_softmax_flat(predictions, target)
        else:
            tr_loss = loss_cross_entropy(target, predictions)

    gradients = tape.gradient(tr_loss, model.trainable_variables)
    opt.apply_gradients(zip(gradients, model.trainable_variables))

    pred_labels = np.argmax(predictions, axis=-1)
    train_miou.addBatch(pred_labels, target)
    tr_miou, _ = train_miou.getIoU()

    return tr_loss, tr_miou


def evaluate(batch, lovasz=False):

    inputs = [batch[0], batch[1]]
    target = batch[2]

    predictions = model(inputs, training=False)
    if lovasz:
        va_loss = lovasz_softmax_flat(predictions, target)
    else:
        va_loss = loss_cross_entropy(target, predictions)

    pred_labels = np.argmax(predictions, axis=-1)
    val_miou.addBatch(pred_labels, target)
    va_miou, _ = val_miou.getIoU()

    return va_loss, va_miou


def train_ckpt_loop(train_cfg, restore_ckpt=False):

    if restore_ckpt:
        ckpt.restore(manager.latest_checkpoint)
        if manager.latest_checkpoint:
            print("Restored from {}".format(manager.latest_checkpoint))
        else:
            print("Initializing from scratch.")

    va_miou = 0.0

    EPOCHS = train_cfg['ep']
    loss_switch_ep = train_cfg['loss_switch_ep']
    lovasz = False

    for epoch in range(EPOCHS):

        prev_va_miou = va_miou
        tr_loss = tr_miou = va_loss = va_miou = 0.0

        if epoch == loss_switch_ep:
            lovasz = True

        for tr_file in train_files:
            tr_inputs = process_single(tr_file)
            tr_outs = train_step(tr_inputs, lovasz)
            tr_loss += tr_outs[0]
            tr_miou += tr_outs[1]

        tr_loss, tr_miou = tr_loss / (len(train_files)), tr_miou / (len(train_files))

        with train_summary_writer.as_default():
            tf.summary.scalar('loss', tr_loss, step=epoch+1)
            tf.summary.scalar('mIoU', tr_miou * 100, step=epoch+1)

        with summary_writer.as_default():
            tf.summary.scalar('learning_rate',
                              opt._decayed_lr(tf.float32),
                              step=epoch+1)

        for va_file in val_files:
            va_inputs = process_single(va_file)
            va_outs = train_step(va_inputs, lovasz)
            va_loss += va_outs[0]
            va_miou += va_outs[1]

        va_loss, va_miou = va_loss / (len(val_files)), va_miou / (len(val_files))

        with valid_summary_writer.as_default():
            tf.summary.scalar('loss', va_loss, step=epoch+1)
            tf.summary.scalar('mIoU', va_miou * 100, step=epoch+1)

        print('Epoch: {} ||| '
              'Train loss: {:.4f}, Train MeanIoU: {:.4f} | '
              'Valid loss: {:.4f}, Valid MeanIoU: {:.4f} ||| '
              .format(epoch + 1,
                      tr_loss, tr_miou * 100,
                      va_loss, va_miou * 100))

        ckpt.step.assign_add(1)
        if (int(ckpt.step) % 10 == 0) and (va_miou > prev_va_miou):
            save_path = manager.save()
            print("Saved checkpoint for step {}: {}".format(int(ckpt.step), save_path))


if __name__ == '__main__':
    BASE_DIR = 'D:/SemanticKITTI/dataset/sequences'

    train_cfg, semkitti_cfg = get_cfg_params(base_dir=BASE_DIR)

    TRIAL = True

    if TRIAL:
        train_files, val_files, _ = get_split_files(dataset_path=BASE_DIR, cfg=semkitti_cfg, shuffle=True, count=100)
    else:
        train_files, val_files, _ = get_split_files(dataset_path=BASE_DIR, cfg=semkitti_cfg, shuffle=True)

    class_ignore = semkitti_cfg["class_ignore"]
    ignore = []
    for cl, ign in class_ignore.items():
        if ign:
            x_cl = int(cl)
            ignore.append(x_cl)
            print("     Ignoring cross-entropy class ", x_cl, " in IoU evaluation")


    model = network(train_cfg)
    model.summary()

    lr_schedule = ExponentialDecay(
        train_cfg['learning_rate'],
        decay_steps=train_cfg['lr_decay'],
        decay_rate=0.96,
        staircase=True)
    opt = Adam(learning_rate=lr_schedule)
    loss_cross_entropy = tf.keras.losses.SparseCategoricalCrossentropy()

    ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimizer=opt)
    manager = tf.train.CheckpointManager(ckpt, './tf_ckpts', max_to_keep=3)

    current_time = datetime.datetime.now().strftime("%Y-%m-%d--%H.%M.%S")
    train_log_dir = 'TB_logs/' + current_time + '/train'
    valid_log_dir = 'TB_logs/' + current_time + '/valid'
    gen_log_dir = 'TB_logs/' + current_time + '/general'
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)
    valid_summary_writer = tf.summary.create_file_writer(valid_log_dir)
    summary_writer = tf.summary.create_file_writer(gen_log_dir)

    num_classes = train_cfg['num_classes']
    patience = train_cfg['patience']
    train_miou = iouEval(num_classes, ignore)
    val_miou = iouEval(num_classes, ignore)

    print('     TRAINING START...')
    print('----------------------------------------------------------------------------------')

    train_ckpt_loop(train_cfg)

    print('----------------------------------------------------------------------------------')
    print('     TRAINING END...')

    save_path = 'models/infer_v1_9_tr100'
    model.save(save_path)
    print('     Model saved to {}'.format(save_path))
    print('==================================================================================')
