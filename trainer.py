import datetime
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow_addons.optimizers import CyclicalLearningRate

from train_utils import loss_metrics
from train_utils.eval_metrics import iouEval

from preprocess import *
from model import Res_GCN_v7 as network

np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)


def assign_loss_func(name):
    '''
    Updates loss function corresponding to dictionary key-value pairs
    :param name: loss function name
    :return: loss function object
    '''

    loss_dict = {'cross_entropy': loss_metrics.cross_entropy,
                 'lovasz': loss_metrics.lovasz_softmax_flat,
                 'dice_loss': loss_metrics.dice_loss,
                 'dice_xentropy': loss_metrics.dice_cross_entropy,
                 'focal_tv': loss_metrics.focal_tversky_loss,
                 'tversky': loss_metrics.tversky_loss,
                 'gen_dice_loss': loss_metrics.generalized_dice_loss,
                 'focal_tv_weighted': loss_metrics.focal_tversky_weighted,
                 'wcce': loss_metrics.weighted_cross_entropy,
                 'sparse_ce': loss_metrics.sparse_cross_entropy}

    return loss_dict.get(str(name))


def train_step(inputs, target, model, optimizer, miou_obj, cfg, loss_fn='dice_loss'):
    '''
    Training step on single batch
    :param inputs: point cloud array, adjacency matrix
    :param target: point-wise ground truth label array
    :param model: network to perform with
    :param cfg: model configuration dictionary
    :param loss_fn: 'str' choice of loss function, updated within main training loop
    :return: training Loss and mIoU metrics
    '''

    X, A, I, = inputs
    Y = np.concatenate(target).ravel()
    # experimental class imbalancing solution
    class_weights = map_content(cfg)
    # class_weights = None

    loss_obj = assign_loss_func(loss_fn)

    with tf.GradientTape() as tape:
        predictions = model([X, A, I], training=True)
        tr_loss = loss_obj(Y, predictions, class_weights=class_weights)

    gradients = tape.gradient(tr_loss, model.trainable_variables)

    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    pred_labels = np.argmax(predictions, axis=-1)
    miou_obj.addBatch(pred_labels, Y)
    tr_miou, _ = miou_obj.getIoU()

    return tr_loss, tr_miou


def evaluate(loader, model, cfg, miou_obj, loss_fn='dice_loss'):
    '''
    Evaluation step every epoch, on validation dataset.
    :param loader: validation dataset loader
    :param model: network to perform with
    :param cfg: model configuration dictionary
    :param loss_fn: 'str' choice of loss function, updated within main training loop
    :return: validation Loss and mIoU metrics
    '''

    va_output = []
    step = 0

    # experimental class imbalancing solution
    class_weights = map_content(cfg)
    # class_weights = None

    loss_obj = assign_loss_func(loss_fn)

    while step < loader.steps_per_epoch:

        step += 1

        inputs, target = loader.__next__()
        X, A, I = inputs
        Y = np.concatenate(target).ravel()

        predictions = model([X, A, I], training=False)
        pred_labels = np.argmax(predictions, axis=-1)

        va_loss = loss_obj(Y, predictions, class_weights=class_weights)

        miou_obj.addBatch(pred_labels, Y)
        va_miou, _ = miou_obj.getIoU()

        va_output.append([va_loss, va_miou])

    outs_avg = np.mean(va_output, 0)
    outs_arr = np.array(outs_avg).flatten()

    return outs_arr[0], outs_arr[1]


def update_loader(train_files, prep_obj, curr_idx, epoch, model_cfg,
                  FLAGS, block_len=2000, verbose=False):
    '''
    Updates training dataset loader with modified file-list
    :param train_files: list of 'train' scan file paths
    :param prep_obj: preprocessor object
    :param curr_idx: dynamic loading step
    :param epoch: current epoch
    :param model_cfg: model configuration dictionary
    :param block_len: no of files to process per loader (10 * validation dataset size)
    :param verbose: terminal print out of progress
    :return: updated training dataset loader
    '''

    start = curr_idx * block_len
    stop = start + block_len

    tr_files_upd = train_files[start:stop]
    print('----------------------------------------------------------------------------------')
    print('Modified train file list start and stop indices: {}, {}'.format(start, stop))

    if verbose:
        print('Updating training data loader with block size of {} at epoch {} of {} ..'
              .format(block_len, epoch, model_cfg['ep']))
    print('----------------------------------------------------------------------------------')

    if len(tr_files_upd) != 0:
        tr_ds = prep_dataset(tr_files_upd, prep_obj, augment=FLAGS.augment, verbose=False)
        tr_loader = prep_loader(tr_ds, model_cfg)
        return tr_loader
    else:
        if epoch == model_cfg['ep']:
            if verbose:
                print('----------------------------------------------------------------------------------')
                print('Training dataset exhausted, stopping training at {} epochs'.format(epoch))
            print('----------------------------------------------------------------------------------')
            return None
        else:
            return 'reset'


def configure_loaders(train_files, val_files, prep_obj, FLAGS, model_cfg):
    '''
    Configures training and validation Spektral data loaders
    :param train_files: list of 'train' scan file paths
    :param val_files: list of 'valid' scan file paths
    :param prep_obj: preprocessor object
    :return: training and validation Spektral data loaders
    '''

    tr_ds = prep_dataset(train_files, prep_obj, verbose=True, augment=FLAGS.augment)
    tr_loader = prep_loader(tr_ds, model_cfg)

    va_ds = prep_dataset(val_files, prep_obj, verbose=True)
    va_loader = prep_loader(va_ds, model_cfg)

    return tr_loader, va_loader


def train(FLAGS):
    '''
    Umbrella training loop --> Prepares data-loaders and performs batch-wise training.
    Model weights are saved every time best validation mIoU is achieved.
    Necessary scalars are written to Tensorboard logs.

    :param FLAGS: Argument parser flags provided as input
    :return: None
    '''

    model_cfg = get_cfg_params(base_dir=FLAGS.dataset, train_cfg=FLAGS.trconfig, dataset_cfg=FLAGS.datacfg)

    model = network(model_cfg)
    save_summary(model)
    prep = Preprocess(model_cfg)

    tr_start_time = datetime.datetime.now().strftime("%Y-%m-%d--%H.%M.%S")
    train_log_dir = 'TB_logs/' + tr_start_time + '/train'
    valid_log_dir = 'TB_logs/' + tr_start_time + '/valid'
    gen_log_dir = 'TB_logs/' + tr_start_time + '/general'
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)
    valid_summary_writer = tf.summary.create_file_writer(valid_log_dir)
    summary_writer = tf.summary.create_file_writer(gen_log_dir)

    num_classes = model_cfg['num_classes']
    class_ignore = model_cfg["class_ignore"]
    train_miou_obj = iouEval(num_classes, class_ignore)
    val_miou_obj = iouEval(num_classes, class_ignore)

    curr_batch = epoch = tr_loss = tr_miou = 0
    list_mious = [0.0]
    # epoch at which to switch to lovàsz loss
    loss_switch_ep = model_cfg['loss_switch_ep']
    # Default loss function
    loss_func = 'sparse_ce'

    if FLAGS.dynamic:

        # Validation dataset size, training dataset will be 10 times this
        dyn_count = 200
        # Dynamic loading 'step' count
        dyn_idx = 0
        ep_int = 20

        train_files, val_files, _ = get_split_files(dataset_path=FLAGS.dataset, cfg=model_cfg, shuffle=True)
        val_files = val_files[:200]

        # Training data loader initialized
        tr_loader = update_loader(train_files, prep, dyn_idx, epoch, model_cfg,
                                  FLAGS, block_len=dyn_count*10)

        print('     Preparing validation data loader..')
        va_ds = prep_dataset(val_files, prep, FLAGS, verbose=True)
        va_loader = prep_loader(va_ds, model_cfg)

    else:

        train_files, val_files, _ = get_split_files(dataset_path=FLAGS.dataset, cfg=model_cfg, count=200, shuffle=True)

        tr_loader, va_loader = configure_loaders(train_files, val_files, prep, FLAGS, model_cfg)

    print('----------------------------------------------------------------------------------')
    print('     TRAINING START...')
    print('----------------------------------------------------------------------------------')

    # lr_schedule = CyclicalLearningRate(
    #     initial_learning_rate=model_cfg['learning_rate'],
    #     maximal_learning_rate=0.1,
    #     step_size=tr_loader.steps_per_epoch,
    #     scale_fn=lambda x: 1.,
    # )
    lr_schedule = ExponentialDecay(
        model_cfg['learning_rate'],
        decay_steps=model_cfg['lr_decay'],
        decay_rate=0.96,
        staircase=True)
    opt = Adam(learning_rate=lr_schedule)
    ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimizer=opt)
    manager = tf.train.CheckpointManager(ckpt, './tf_ckpts', max_to_keep=2)

    if FLAGS.ckpt:
        ckpt.restore(manager.latest_checkpoint)
        if manager.latest_checkpoint:
            print("     Restored from checkpoint: {}".format(manager.latest_checkpoint))

    for batch in tr_loader:

        outs = train_step(*batch, model=model, optimizer=opt, miou_obj=train_miou_obj, cfg=model_cfg, loss_fn=loss_func)

        tr_loss += outs[0]
        tr_miou += outs[1]
        curr_batch += 1

        if curr_batch == tr_loader.steps_per_epoch:

            tr_loss /= tr_loader.steps_per_epoch
            tr_miou /= tr_loader.steps_per_epoch
            epoch += 1

            va_loss, va_miou = evaluate(va_loader, model=model, miou_obj=val_miou_obj, cfg=model_cfg, loss_fn=loss_func)

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

            print('Epoch: {} ||| '
                  'Train loss: {:.4f}, Train MeanIoU: {:.4f} | '
                  'Valid loss: {:.4f}, Valid MeanIoU: {:.4f} ||| '
                  .format(epoch,
                          tr_loss, tr_miou * 100,
                          va_loss, va_miou * 100))


            ckpt.step.assign_add(1)

            if (int(ckpt.step) % 10 == 0):

                print('----------------------------------------------------------------------------------')

                # save_path = manager.save()
                weights_path = './ckpt_weights/' + tr_start_time
                # print("Saved checkpoint for step {}: {}".format(int(ckpt.step), save_path))

                if (va_miou > np.max(list_mious)) and FLAGS.save:

                    print("Saved weights for step {}: {}".format(int(ckpt.step), weights_path))
                    model.save_weights(weights_path, overwrite=True)

                print('----------------------------------------------------------------------------------')

            list_mious.append(va_miou)

            tr_loss = 0
            tr_miou = 0
            curr_batch = 0

            # Training loader updated every 20 epochs
            if FLAGS.dynamic and (int(epoch) % ep_int == 0):
                dyn_idx += 1
                tr_loader = update_loader(train_files, prep, dyn_idx, epoch,
                                          model_cfg, FLAGS, verbose=True)
                if tr_loader is None:
                    break
                elif tr_loader == 'reset':
                    dyn_idx = 0
                    tr_loader = update_loader(train_files, prep, dyn_idx, epoch,
                                              model_cfg, FLAGS, verbose=True)

            # Switch loss function to Lovàsz-Softmax if epoch reaches threshold (based on model cfg)
            if epoch == loss_switch_ep:

                loss_func = 'lovasz'
                print('//////////////////////////////////////////////////////////////////////////////////')
                print('Switching loss function to Lovàsz-Softmax..')
                print('//////////////////////////////////////////////////////////////////////////////////')

    print('----------------------------------------------------------------------------------')
    print('     TRAINING END...')
    print('----------------------------------------------------------------------------------')

    if FLAGS.save:
        save_path = 'models/infer_v3_7_resgcn7_xyzirgb_200_bs8_cce_lov_aug'
        model.save(save_path)
        print('     Model saved to {}'.format(save_path))
        print('==================================================================================')



