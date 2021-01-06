import numpy as np
import datetime
import tensorflow as tf
import sys

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import ExponentialDecay

from preproc_utils.dataprep import get_split_files, get_cfg_params
from train_utils.lovasz_loss import lovasz_softmax_flat
from batch_gen import process_single, process_voxel, voxel
from train_utils.eval_metrics import iouEval
from model import Graph_U as network


def get_apply_grad_fn():
    @tf.function
    def apply_grad(inputs, target):
        with tf.GradientTape() as t:

            predictions = model(inputs)

            loss = lovasz_softmax_flat(predictions, target)

        grads = t.gradient(loss, model.trainable_weights)

        opt.apply_gradients(zip(grads, model.trainable_weights))

        return predictions, loss
    return apply_grad


# @tf.function(input_signature=[[tf.TensorSpec(shape=tf.TensorShape(None), dtype=tf.float32),
#                                tf.SparseTensorSpec(shape=tf.TensorShape(None), dtype=tf.float32)],
#              tf.TensorSpec(shape=tf.TensorShape(None), dtype=tf.int32)])
# @tf.function
def grad_func(inputs, target):
    with tf.GradientTape() as tape:
        predictions = model(inputs, training=True)
        curr_tr_loss = loss_cross_entropy(target, predictions)

    gradients = tape.gradient(curr_tr_loss, model.trainable_variables)
    opt.apply_gradients(zip(gradients, model.trainable_variables))

    return predictions, curr_tr_loss

def train_step(batch, lovasz=False, vox=False):

    inputs = [batch[0], batch[1]]
    target = batch[2]
    train_miou.reset_states()

    if vox:
        vox_pc_map = voxel(batch[0])
        tr_loss = tr_miou = 0

        for vox_id in range(len(vox_pc_map)):
            curr_voxel_data = process_voxel(vox_pc_map, vox_id, target)

            vox_inputs = [curr_voxel_data[0], curr_voxel_data[1]]
            vox_target = curr_voxel_data[2]

            predictions, curr_tr_loss = grad_func(vox_inputs, vox_target)

            # pred_labels = np.argmax(predictions, axis=-1)
            pred_labels = tf.argmax(predictions, axis=-1)

            # train_miou.addBatch(pred_labels, vox_target)
            # curr_tr_miou, _ = train_miou.getIoU()

            train_miou.update_state(vox_target, pred_labels)

            tr_loss += curr_tr_loss
            # tr_miou += curr_tr_miou
            tr_miou += train_miou.result()

        tr_loss, tr_miou = tr_loss / (len(vox_pc_map)), tr_miou / (len(vox_pc_map))

    else:
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

def evaluate(batch, lovasz=False, vox=False):

    inputs = [batch[0], batch[1]]
    target = batch[2]
    val_miou.reset_states()


    if vox:
        vox_pc_map = voxel(batch[0])
        va_loss = va_miou = 0

        for vox_id in range(len(vox_pc_map)):
            curr_voxel_data = process_voxel(vox_pc_map, vox_id, target)

            vox_inputs = [curr_voxel_data[0], curr_voxel_data[1]]
            vox_target = curr_voxel_data[2]

            predictions = model(vox_inputs, training=False)

            if lovasz:
                curr_va_loss = lovasz_softmax_flat(predictions, vox_target)
            else:
                curr_va_loss = loss_cross_entropy(vox_target, predictions)

            pred_labels = np.argmax(predictions, axis=-1)

            # val_miou.addBatch(pred_labels, vox_target)
            # curr_va_miou, _ = val_miou.getIoU()
            val_miou.update_state(vox_target, pred_labels)

            va_loss += curr_va_loss
            va_miou += val_miou.result()

        va_loss, va_miou = va_loss / (len(vox_pc_map)), va_miou / (len(vox_pc_map))
    else:
        predictions = model(inputs, training=False)
        if lovasz:
            va_loss = lovasz_softmax_flat(predictions, target)
        else:
            va_loss = loss_cross_entropy(target, predictions)

        pred_labels = np.argmax(predictions, axis=-1)
        val_miou.addBatch(pred_labels, target)
        va_miou, _ = val_miou.getIoU()

    return va_loss, va_miou

@tf.function
@tf.autograph.experimental.do_not_convert
def train_ckpt_loop(train_cfg, restore_ckpt=False, verbose=True, vox=True, save_weights=False):
    if restore_ckpt:
        ckpt.restore(manager.latest_checkpoint)
        print("Restored from {}".format(manager.latest_checkpoint))

    best_va_miou = 0.0

    EPOCHS = train_cfg['ep']
    loss_switch_ep = train_cfg['loss_switch_ep']
    lovasz = False

    for epoch in range(EPOCHS):

        tr_loss = tr_miou = va_loss = va_miou = 0.0

        if epoch == loss_switch_ep:
            lovasz = True
            print('//////////////////////////////////////////////////////////////////////////////////')
            print('Switching loss function to Lovàsz-Softmax..')
            print('//////////////////////////////////////////////////////////////////////////////////')

        for tr_file in train_files:
            if verbose:
                print('     --> Processing train file: ', tr_file)
            tr_inputs = process_single(tr_file, train_cfg, verbose)
            tr_outs = train_step(tr_inputs, lovasz, vox=vox)
            tr_loss += tr_outs[0]
            tr_miou += tr_outs[1]
            if verbose:
                print(tr_loss.numpy())
                tf.print("MIOU: ", train_miou.result(), output_stream=sys.stdout)

        tr_loss, tr_miou = tr_loss / (len(train_files)), tr_miou / (len(train_files))

        with train_summary_writer.as_default():
            tf.summary.scalar('loss', tr_loss, step=epoch + 1)
            tf.summary.scalar('mIoU', tr_miou * 100, step=epoch + 1)

        with summary_writer.as_default():
            tf.summary.scalar('learning_rate',
                              opt._decayed_lr(tf.float32),
                              step=epoch + 1)

        for va_file in val_files:
            if verbose:
                print('     --> Processing valid file: ', va_file)
            va_inputs = process_single(va_file, train_cfg, verbose)
            va_outs = evaluate(va_inputs, lovasz, vox=vox)
            va_loss += va_outs[0]
            va_miou += va_outs[1]
            if verbose:
                print('     --> Current Valid mIoU: ', round(va_miou * 100))

        va_loss, va_miou = va_loss / (len(val_files)), va_miou / (len(val_files))

        with valid_summary_writer.as_default():
            tf.summary.scalar('loss', va_loss, step=epoch + 1)
            tf.summary.scalar('mIoU', va_miou * 100, step=epoch + 1)

        print('Epoch: {} ||| '
              'Train loss: {:.4f}, Train MeanIoU: {:.4f} | '
              'Valid loss: {:.4f}, Valid MeanIoU: {:.4f} ||| '
              .format(epoch + 1,
                      tr_loss, tr_miou * 100,
                      va_loss, va_miou * 100))

        ckpt.step.assign_add(1)
        if (int(ckpt.step) % 5 == 0) and (va_miou > best_va_miou):
            print('----------------------------------------------------------------------------------')
            save_path = manager.save()
            weights_path = './ckpt_weights/' + current_time
            print("Saved checkpoint for step {}: {}".format(int(ckpt.step), save_path))
            if save_weights:
                print("Saved weights for step {}: {}".format(int(ckpt.step), weights_path))
                model.save_weights(weights_path, overwrite=False)
            best_va_miou = va_miou
            print('----------------------------------------------------------------------------------')


if __name__ == '__main__':
    BASE_DIR = 'D:/SemanticKITTI/dataset/sequences'

    train_cfg = get_cfg_params(base_dir=BASE_DIR)

    TRIAL = True

    if TRIAL:
        train_files, val_files, _ = get_split_files(dataset_path=BASE_DIR, cfg=train_cfg, shuffle=True, count=1)
    else:
        train_files, val_files, _ = get_split_files(dataset_path=BASE_DIR, cfg=train_cfg, shuffle=True)

    class_ignore = train_cfg["class_ignore"]

    model = network(train_cfg)
    # model.summary()

    lr_schedule = ExponentialDecay(
        train_cfg['learning_rate'],
        decay_steps=train_cfg['lr_decay'],
        decay_rate=0.96,
        staircase=True)
    opt = Adam(learning_rate=lr_schedule)
    loss_cross_entropy = tf.keras.losses.SparseCategoricalCrossentropy()

    ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimizer=opt)
    manager = tf.train.CheckpointManager(ckpt, './tf_ckpts', max_to_keep=2)

    current_time = datetime.datetime.now().strftime("%Y-%m-%d--%H.%M.%S")
    train_log_dir = 'TB_logs/' + current_time + '/train'
    valid_log_dir = 'TB_logs/' + current_time + '/valid'
    gen_log_dir = 'TB_logs/' + current_time + '/general'
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)
    valid_summary_writer = tf.summary.create_file_writer(valid_log_dir)
    summary_writer = tf.summary.create_file_writer(gen_log_dir)

    num_classes = train_cfg['num_classes']
    patience = train_cfg['patience']

    # train_miou = iouEval(num_classes, class_ignore)
    # val_miou = iouEval(num_classes, class_ignore)
    train_miou = tf.keras.metrics.MeanIoU(num_classes)
    val_miou = tf.keras.metrics.MeanIoU(num_classes)

    print('     TRAINING START...')
    print('----------------------------------------------------------------------------------')

    train_ckpt_loop(train_cfg)

    print('----------------------------------------------------------------------------------')
    print('     TRAINING END...')

    save_path = 'models/infer_v1_9_U_cheb1'
    model.save(save_path)
    print('     Model saved to {}'.format(save_path))
    print('==================================================================================')
