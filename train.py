import datetime

import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import ExponentialDecay

from train_utils.loss_metrics import lovasz_softmax_flat
from train_utils.eval_metrics import iouEval
from model import res_model_2 as network
from preprocess import *

# @tf.function(input_signature=[[tf.TensorSpec(shape=tf.TensorShape(None), dtype=tf.float32),
#                                tf.SparseTensorSpec(shape=tf.TensorShape(None), dtype=tf.float32)],
#              tf.TensorSpec(shape=tf.TensorShape(None), dtype=tf.int32)])
# @tf.function

def grad_func(inputs, target, loss_fn):
    with tf.GradientTape() as tape:
        predictions = model(inputs, training=True)
        curr_tr_loss = loss_fn(target, predictions)

    gradients = tape.gradient(curr_tr_loss, model.trainable_variables)
    opt.apply_gradients(zip(gradients, model.trainable_variables))

    return predictions, curr_tr_loss

def train_step(batch, prep_obj, lovasz=False, vox=False):

    inputs = [batch[0], batch[1]]
    target = batch[2]

    loss_fn = loss_cross_entropy
    # train_miou.reset_states()
    if lovasz:
        loss_fn = lovasz_softmax_flat

    if vox:
        prep_obj.voxelize_scan()
        no_voxels = len(prep_obj.vox_pc_map)

        tr_loss = tr_miou = 0

        for vox_id in range(no_voxels):

            curr_voxel_data = prep_obj.assess_voxel(vox_id)

            vox_inputs = [curr_voxel_data[0], curr_voxel_data[1]]
            vox_target = curr_voxel_data[2]

            predictions, curr_tr_loss = grad_func(vox_inputs, vox_target, loss_fn)

            pred_labels = np.argmax(predictions, axis=-1)
            # pred_labels = tf.argmax(predictions, axis=-1)

            train_miou.addBatch(pred_labels, vox_target)
            curr_tr_miou, _ = train_miou.getIoU()

            # train_miou.update_state(vox_target, pred_labels)

            tr_loss += curr_tr_loss
            tr_miou += curr_tr_miou
            # tr_miou += train_miou.result()

        tr_loss, tr_miou = tr_loss / no_voxels, tr_miou / no_voxels

    else:

        predictions, tr_loss = grad_func(inputs, target, loss_fn)

        pred_labels = np.argmax(predictions, axis=-1)
        train_miou.addBatch(pred_labels, target)
        tr_miou, _ = train_miou.getIoU()

    return tr_loss, tr_miou

def evaluate(batch, prep_obj, lovasz=False, vox=False):

    inputs = [batch[0], batch[1]]
    target = batch[2]

    loss_fn = loss_cross_entropy
    # val_miou.reset_states()

    if lovasz:
        loss_fn = lovasz_softmax_flat

    if vox:
        prep_obj.voxelize_scan()
        no_voxels = len(prep_obj.vox_pc_map)

        va_loss = va_miou = 0

        for vox_id in range(no_voxels):

            curr_voxel_data = prep_obj.assess_voxel(vox_id)

            vox_inputs = [curr_voxel_data[0], curr_voxel_data[1]]
            vox_target = curr_voxel_data[2]

            predictions = model(vox_inputs, training=False)

            curr_va_loss = loss_fn(vox_target, predictions)

            pred_labels = np.argmax(predictions, axis=-1)

            val_miou.addBatch(pred_labels, vox_target)
            curr_va_miou, _ = val_miou.getIoU()
            # val_miou.update_state(vox_target, pred_labels)

            va_loss += curr_va_loss
            # va_miou += val_miou.result()
            va_miou += curr_va_miou

        va_loss, va_miou = va_loss / no_voxels, va_miou / no_voxels

    else:
        predictions = model(inputs, training=False)

        va_loss = loss_fn(target, predictions)

        pred_labels = np.argmax(predictions, axis=-1)
        val_miou.addBatch(pred_labels, target)
        va_miou, _ = val_miou.getIoU()

    return va_loss, va_miou

# @tf.function
# @tf.autograph.experimental.do_not_convert
def train_ckpt_loop(model_cfg, prep_obj, restore_ckpt=False, verbose=False, vox=False, save_weights=True):
    if restore_ckpt:
        ckpt.restore(manager.latest_checkpoint)
        print("Restored from {}".format(manager.latest_checkpoint))

    best_va_miou = 0.0
    best_va_loss = 99999

    EPOCHS = model_cfg['ep']
    loss_switch_ep = model_cfg['loss_switch_ep']
    lovasz = False

    for epoch in range(EPOCHS):

        tr_loss = tr_miou = va_loss = va_miou = 0.0

        if (epoch + 1) == loss_switch_ep:
            lovasz = True
            print('//////////////////////////////////////////////////////////////////////////////////')
            print('Switching loss function to Lovàsz-Softmax..')
            print('//////////////////////////////////////////////////////////////////////////////////')

        for tr_file in train_files:
            if verbose:
                print('     --> Processing train file: ', tr_file)
            tr_inputs = prep_obj.assess_scan(tr_file)
            tr_outs = train_step(tr_inputs, prep_obj, lovasz, vox=vox)
            tr_loss += tr_outs[0]
            tr_miou += tr_outs[1]
            if verbose:
                print('         Current Train mIoU: ', round(tr_miou*100))

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
            va_inputs = prep_obj.assess_scan(va_file)
            va_outs = evaluate(va_inputs, prep_obj, lovasz, vox=vox)
            va_loss += va_outs[0]
            va_miou += va_outs[1]
            if verbose:
                print('         Current Valid mIoU: ', round(va_miou * 100))

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

        if (int(ckpt.step) % 2 == 0):

            print('----------------------------------------------------------------------------------')

            save_path = manager.save()
            weights_path = './ckpt_weights/' + current_time
            print("Saved checkpoint for step {}: {}".format(int(ckpt.step), save_path))

            if (va_miou > best_va_miou) and (va_loss < best_va_loss) and save_weights:
                print("Saved weights for step {}: {}".format(int(ckpt.step), weights_path))
                model.save_weights(weights_path, overwrite=True)
                best_va_miou = va_miou
                best_va_loss = va_loss

            print('----------------------------------------------------------------------------------')


if __name__ == '__main__':
    BASE_DIR = 'D:/SemanticKITTI/dataset/sequences'

    model_cfg = get_cfg_params(base_dir=BASE_DIR)

    TRIAL = True

    if TRIAL:
        train_files, val_files, _ = get_split_files(dataset_path=BASE_DIR, cfg=model_cfg, shuffle=True, count=50)
    else:
        train_files, val_files, _ = get_split_files(dataset_path=BASE_DIR, cfg=model_cfg, shuffle=True)

    class_ignore = model_cfg["class_ignore"]

    prep = Preprocess(model_cfg)

    model = network(model_cfg)
    # model.summary()

    lr_schedule = ExponentialDecay(
        model_cfg['learning_rate'],
        decay_steps=model_cfg['lr_decay'],
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

    num_classes = model_cfg['num_classes']
    patience = model_cfg['patience']

    train_miou = iouEval(num_classes, class_ignore)
    val_miou = iouEval(num_classes, class_ignore)
    # mean_loss_values = tf.metrics.Mean()
    # train_miou = tf.keras.metrics.MeanIoU(num_classes)
    # val_miou = tf.keras.metrics.MeanIoU(num_classes)

    print('     TRAINING START...')
    print('----------------------------------------------------------------------------------')

    train_ckpt_loop(model_cfg, prep, vox=False)

    print('----------------------------------------------------------------------------------')
    print('     TRAINING END...')

    save_path = 'models/infer_v2_1_res_rgb'
    model.save(save_path)
    print('     Model saved to {}'.format(save_path))
    print('==================================================================================')
