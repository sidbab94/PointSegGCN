import numpy as np
from os import listdir
import datetime
from yaml import safe_load
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
tf.config.run_functions_eagerly(False)

from preproc_utils.dataprep import get_split_files
from train_utils.lovasz_loss import lovasz_softmax_flat

from batch_gen import process
from train_utils.eval_met import iouEval
from model import res_model_1 as network

def train_step(batch, lovasz=False):

    inputs = [batch.x, batch.a]
    target = batch.y

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

    inputs = [batch.x, batch.a]
    target = batch.y

    predictions = model(inputs, training=False)
    if lovasz:
        va_loss = lovasz_softmax_flat(predictions, target)
    else:
        va_loss = loss_cross_entropy(target, predictions)

    pred_labels = np.argmax(predictions, axis=-1)
    val_miou.addBatch(pred_labels, target)
    va_miou, _ = val_miou.getIoU()

    return va_loss, va_miou

if __name__ == '__main__':
    BASE_DIR = 'D:/SemanticKITTI/dataset/sequences'
    semkitti_cfg = safe_load(open('config/semantic-kitti.yaml', 'r'))
    tr_params = safe_load(open('config/tr_config.yml', 'r'))['training_params']

    split_params = semkitti_cfg['split']

    seq_list = np.sort(listdir(BASE_DIR))
    tr_seq_list = list(seq_list[split_params['train']])
    va_seq_list = list(seq_list[split_params['valid']])

    TRIAL = True

    if TRIAL:
        train_files, val_files, _ = get_split_files(dataset_path=BASE_DIR, cfg=semkitti_cfg["split"], count=10)
    else:
        train_files, val_files, _ = get_split_files(dataset_path=BASE_DIR, cfg=semkitti_cfg["split"])

    print('==================================================================================')
    print('1.   Preparing Graph dataset for training sequences..')
    print(train_files)
    train_batches = process(train_files, shuffle=True, vox=True)
    print('     .. Done.')
    print('----------------------------------------------------------------------------------')
    print('2.   Preparing Graph dataset for validation sequences..')
    val_batches = process(val_files, shuffle=True)
    print('     .. Done.')
    print('----------------------------------------------------------------------------------')

    EPOCHS = tr_params['epochs']
    num_classes = tr_params['num_classes']
    patience = tr_params['es_patience']
    loss_switch_ep = tr_params['lovasz_switch']

    class_ignore = semkitti_cfg["learning_ignore"]
    ignore = []
    for cl, ign in class_ignore.items():
        if ign:
            x_cl = int(cl)
            ignore.append(x_cl)
            print("     Ignoring cross-entropy class ", x_cl, " in IoU evaluation")

    train_miou = iouEval(num_classes, ignore)
    val_miou = iouEval(num_classes, ignore)

    model = network(tr_params)
    model.summary()

    opt = Adam(lr=tr_params['learning_rate'])
    loss_cross_entropy = tf.keras.losses.SparseCategoricalCrossentropy()

    current_time = datetime.datetime.now().strftime("%Y-%m-%d--%H.%M.%S")
    train_log_dir = 'TB_logs/' + current_time + '/train'
    valid_log_dir = 'TB_logs/' + current_time + '/valid'
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)
    valid_summary_writer = tf.summary.create_file_writer(valid_log_dir)

    print('     TRAINING START...')
    print('----------------------------------------------------------------------------------')
    lovasz = False
    for epoch in range(EPOCHS):
        tr_loss = tr_miou = va_loss = va_miou = 0.0

        if epoch == loss_switch_ep:
            lovasz = True

        for i in range(len(train_batches)):
            tr_batch = train_batches[i]
            tr_loss += train_step(tr_batch, lovasz)[0]
            tr_miou += train_step(tr_batch, lovasz)[1]
            prev_batch = tr_batch

        tr_loss, tr_miou = tr_loss / (len(train_batches)), tr_miou / (len(train_batches))

        with train_summary_writer.as_default():
            tf.summary.scalar('loss', tr_loss.numpy(), step=epoch)
            tf.summary.scalar('mIoU', tr_miou * 100, step=epoch)

        for i in range(len(val_batches)):
            va_batch = val_batches[i]
            va_loss += evaluate(va_batch, lovasz)[0]
            va_miou += evaluate(va_batch, lovasz)[1]

        va_loss, va_miou = va_loss / (len(val_batches)), va_miou / (len(val_batches))

        with valid_summary_writer.as_default():
            tf.summary.scalar('loss', va_loss.numpy(), step=epoch)
            tf.summary.scalar('mIoU', va_miou * 100, step=epoch)

        print('Epoch: {} ||| '
              'Train loss: {:.4f}, Train MeanIoU: {:.4f} | '
              'Valid loss: {:.4f}, Valid MeanIoU: {:.4f} ||| '
              .format(epoch,
                      tr_loss.numpy(), tr_miou * 100,
                      va_loss.numpy(), va_miou * 100))

    print('----------------------------------------------------------------------------------')
    print('     TRAINING END...')

    save_path = 'models/infer_v1_9_small'
    model.save(save_path)
    print('     Model saved to {}'.format(save_path))
    print('==================================================================================')
