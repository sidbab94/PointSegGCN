import datetime

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import ExponentialDecay

from train_utils.loss_metrics import lovasz_softmax_flat
from train_utils.eval_metrics import iouEval
from model import res_model_1 as network

from preprocess import *
from preproc_utils.readers import get_cfg_params


def grad_func(inputs, target, loss_fn):

    with tf.GradientTape() as tape:
        predictions = model(inputs, training=True)
        curr_tr_loss = loss_fn(target, predictions)

    gradients = tape.gradient(curr_tr_loss, model.trainable_variables)
    opt.apply_gradients(zip(gradients, model.trainable_variables))

    return predictions, curr_tr_loss

def train_step(inputs, target, lovasz=False):

    X, A, _, = inputs
    Y = np.concatenate(target).ravel()

    loss_fn = loss_cross_entropy

    if lovasz:
        loss_fn = lovasz_softmax_flat

    predictions, tr_loss = grad_func([X, A], Y, loss_fn)
    pred_labels = np.argmax(predictions, axis=-1)
    train_miou.addBatch(pred_labels, Y)
    tr_miou, _ = train_miou.getIoU()

    return tr_loss, tr_miou

def evaluate(loader, lovasz=False):

    va_output = []
    step = 0

    while step < loader.steps_per_epoch:

        step += 1

        inputs, target = loader.__next__()
        X, A, _ = inputs
        Y = np.concatenate(target).ravel()

        loss_fn = loss_cross_entropy

        if lovasz:
            loss_fn = lovasz_softmax_flat

        predictions = model([X, A], training=False)

        va_loss = loss_fn(Y, predictions)

        pred_labels = np.argmax(predictions, axis=-1)

        val_miou.addBatch(pred_labels, Y)
        va_miou, _ = val_miou.getIoU()

        va_output.append([va_loss, va_miou])

    outs_avg = np.mean(va_output, 0)
    outs_arr = np.array(outs_avg).flatten()

    return outs_arr[0], outs_arr[1]


def train_loop(save_weights=True):

    ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimizer=opt)
    manager = tf.train.CheckpointManager(ckpt, './tf_ckpts', max_to_keep=2)

    curr_batch = epoch = tr_loss = tr_miou = best_miou = 0
    list_mious = [0.0]
    best_va_loss = 9999.999
    lovasz = False
    loss_switch_ep = model_cfg['loss_switch_ep']


    for batch in tr_loader:


        outs = train_step(*batch, lovasz)

        tr_loss += outs[0]
        tr_miou += outs[1]
        curr_batch += 1

        if curr_batch == tr_loader.steps_per_epoch:

            tr_loss /= tr_loader.steps_per_epoch
            tr_miou /= tr_loader.steps_per_epoch
            epoch += 1

            va_loss, va_miou = evaluate(va_loader, lovasz)

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

            print('Epoch: {} ||| '
                  'Train loss: {:.4f}, Train MeanIoU: {:.4f} | '
                  'Valid loss: {:.4f}, Valid MeanIoU: {:.4f} ||| '
                  .format(epoch,
                          tr_loss, tr_miou * 100,
                          va_loss, va_miou * 100))


            ckpt.step.assign_add(1)

            if (int(ckpt.step) % 2 == 0):

                print('----------------------------------------------------------------------------------')

                save_path = manager.save()
                weights_path = './ckpt_weights/' + tr_start_time
                print("Saved checkpoint for step {}: {}".format(int(ckpt.step), save_path))

                if (va_miou > np.max(list_mious)) and save_weights:

                    print("Saved weights for step {}: {}".format(int(ckpt.step), weights_path))
                    model.save_weights(weights_path, overwrite=True)
                    best_va_loss = va_loss

                print('----------------------------------------------------------------------------------')

            list_mious.append(va_miou)

            tr_loss = 0
            tr_miou = 0
            curr_batch = 0

            if epoch == loss_switch_ep:

                lovasz = True
                print('//////////////////////////////////////////////////////////////////////////////////')
                print('Switching loss function to Lovàsz-Softmax..')
                print('//////////////////////////////////////////////////////////////////////////////////')



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
            weights_path = './ckpt_weights/' + tr_start_time
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

    prep = Preprocess(model_cfg)
    model = network(model_cfg)

    tr_ds, va_ds = prep_datasets(BASE_DIR, prep, model_cfg, file_count=300, verbose=True)
    tr_loader = prep_loader(tr_ds, model_cfg)
    va_loader = prep_loader(va_ds, model_cfg)

    # model.summary()

    lr_schedule = ExponentialDecay(
        model_cfg['learning_rate'],
        decay_steps=model_cfg['lr_decay'],
        decay_rate=0.96,
        staircase=True)
    opt = Adam(learning_rate=lr_schedule)
    loss_cross_entropy = tf.keras.losses.SparseCategoricalCrossentropy()

    # ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimizer=opt)
    # manager = tf.train.CheckpointManager(ckpt, './tf_ckpts', max_to_keep=2)

    tr_start_time = datetime.datetime.now().strftime("%Y-%m-%d--%H.%M.%S")
    train_log_dir = 'TB_logs/' + tr_start_time + '/train'
    valid_log_dir = 'TB_logs/' + tr_start_time + '/valid'
    gen_log_dir = 'TB_logs/' + tr_start_time + '/general'
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)
    valid_summary_writer = tf.summary.create_file_writer(valid_log_dir)
    summary_writer = tf.summary.create_file_writer(gen_log_dir)

    num_classes = model_cfg['num_classes']
    class_ignore = model_cfg["class_ignore"]
    train_miou = iouEval(num_classes, class_ignore)
    val_miou = iouEval(num_classes, class_ignore)


    print('     TRAINING START...')
    print('----------------------------------------------------------------------------------')

    # train_ckpt_loop(model_cfg, prep, vox=False)
    train_loop()

    print('----------------------------------------------------------------------------------')
    print('     TRAINING END...')

    save_path = 'models/infer_v3_2_res_rgb_300_bs16'
    model.save(save_path)
    print('     Model saved to {}'.format(save_path))
    print('==================================================================================')
