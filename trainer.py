import datetime
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import pymsteams

import tensorflow as tf
from tensorflow.keras.optimizers import Adam

from train_utils import loss_metrics
from train_utils.eval_metrics import iouEval
from train_utils.tf_utils import CyclicalLR

from preprocess import *
from models import Dense_GCN as network

np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)


def assign_loss_func(name):
    '''
    Updates loss function corresponding to dictionary key-value pairs
    :param name: loss function name
    :return: loss function object
    '''

    loss_dict = {'lovasz': loss_metrics.lovasz_softmax_flat,
                 'focal_tv': loss_metrics.focal_tversky_loss,
                 'tversky': loss_metrics.tversky_loss,
                 'sparse_ce': loss_metrics.sparse_cross_entropy,
                 'sparse_focal': loss_metrics.sparse_categorical_focal_loss}

    return loss_dict.get(str(name))


def train_step(inputs, model, optimizer, miou_obj, cfg, loss_fn='dice_loss'):
    '''
    Training step on single batch
    :param inputs: point cloud array, adjacency matrix
    :param target: point-wise ground truth label array
    :param model: network to perform with
    :param cfg: model configuration dictionary
    :param loss_fn: 'str' choice of loss function, updated within main training loop
    :return: training Loss and mIoU metrics
    '''

    X, A, Y, = inputs
    # experimental class imbalancing solution
    # class_weights = map_content(cfg)
    class_weights = None

    loss_obj = assign_loss_func(loss_fn)

    with tf.GradientTape() as tape:
        predictions = model([X, A], training=True)
        tr_loss = loss_obj(Y, predictions, class_weights=class_weights)

    gradients = tape.gradient(tr_loss, model.trainable_variables)

    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    pred_labels = np.argmax(predictions, axis=-1)
    miou_obj.addBatch(pred_labels, Y)
    tr_miou, _ = miou_obj.getIoU()
    tr_loss = np.mean(tr_loss)

    return tr_loss, tr_miou


def evaluate(inputs, model, cfg, miou_obj, loss_fn='dice_loss'):
    '''
    Evaluation step every epoch, on validation dataset.
    :param loader: validation dataset loader
    :param model: network to perform with
    :param cfg: model configuration dictionary
    :param loss_fn: 'str' choice of loss function, updated within main training loop
    :return: validation Loss and mIoU metrics
    '''

    va_output = []

    # experimental class imbalancing solution
    # class_weights = map_content(cfg)
    class_weights = None

    loss_obj = assign_loss_func(loss_fn)

    X, A, Y = inputs

    predictions = model([X, A], training=False)
    pred_labels = np.argmax(predictions, axis=-1)

    va_loss = loss_obj(Y, predictions, class_weights=class_weights)

    miou_obj.addBatch(pred_labels, Y)
    va_miou, _ = miou_obj.getIoU()
    va_loss = np.mean(va_loss)

    va_output.append([va_loss, va_miou])

    outs_avg = np.mean(va_output, 0)
    outs_arr = np.array(outs_avg).flatten()

    return outs_arr[0], outs_arr[1]


def train(FLAGS):
    '''
    Umbrella training loop --> Prepares data-loaders and performs batch-wise training.
    Model weights are saved every time best validation mIoU is achieved.
    Necessary scalars are written to Tensorboard logs.

    :param FLAGS: Argument parser flags provided as input
    :return: None
    '''

    model_cfg = get_cfg_params(cfg_file=FLAGS.config)

    model = network(model_cfg)

    # ## Pre-trained
    # latest_checkpoint = tf.train.latest_checkpoint('./ckpt_weights')
    # load_status = model.load_weights(latest_checkpoint)
    # load_status.assert_consumed()

    # save_summary(model)
    prep = Preprocess(model_cfg)

    tr_start_time = datetime.datetime.now().strftime("%Y-%m-%d--%H.%M.%S")
    train_log_dir = 'TB_logs/' + tr_start_time + '/train'
    valid_log_dir = 'TB_logs/' + tr_start_time + '/valid'
    gen_log_dir = 'TB_logs/' + tr_start_time + '/general'
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)
    valid_summary_writer = tf.summary.create_file_writer(valid_log_dir)
    summary_writer = tf.summary.create_file_writer(gen_log_dir)

    teamshook = pymsteams.connectorcard(FLAGS.teamshook)

    num_classes = model_cfg['num_classes']
    class_ignore = model_cfg["class_ignore"]
    train_miou_obj = iouEval(num_classes, class_ignore)
    val_miou_obj = iouEval(num_classes, class_ignore)

    tr_loss = tr_miou = va_loss = va_miou = 0
    list_mious = [0.0]
    # epoch at which to switch to lovàsz loss
    loss_switch_ep = model_cfg['loss_switch_ep']
    # Default loss function
    loss_func = 'sparse_focal'

    train_files, val_files, _ = get_split_files(cfg=model_cfg, shuffle=True)

    if FLAGS.trial:
        train_files = train_files[:50]
        val_files = val_files[:5]

    print('----------------------------------------------------------------------------------')
    print('     TRAINING START...')
    print('----------------------------------------------------------------------------------')

    lr_schedule = CyclicalLR(base_lr=0.01, max_lr=0.1)

    opt = Adam(learning_rate=lr_schedule)

    ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimizer=opt)
    manager = tf.train.CheckpointManager(ckpt, './tf_ckpts', max_to_keep=2)

    if FLAGS.ckpt:
        ckpt.restore(manager.latest_checkpoint)
        if manager.latest_checkpoint:
            print("     Restored from checkpoint: {}".format(manager.latest_checkpoint))

    for epoch in range(model_cfg['epochs']):

        for tr_file in train_files:
            tr_inputs = tr_batch_gen(prep, tr_file, 4)
            tr_loss, tr_miou = train_step(tr_inputs, model=model, optimizer=opt,
                                          miou_obj=train_miou_obj, cfg=model_cfg, loss_fn=loss_func)

        for va_file in val_files:
            va_inputs = va_batch_gen(prep, va_file)
            va_loss, va_miou = evaluate(va_inputs, model=model, miou_obj=val_miou_obj,
                                        loss_fn=loss_func, cfg=model_cfg)

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

        if (int(ckpt.step) % 10 == 0):

            teamshook.text(curr_stats)
            teamshook.send()

            print('----------------------------------------------------------------------------------')

            # save_path = manager.save()
            weights_path = './ckpt_weights/' + tr_start_time
            # print("Saved checkpoint for step {}: {}".format(int(ckpt.step), save_path))

            if (va_miou > np.max(list_mious)) and FLAGS.save:
                print("Saved weights for step {}: {}".format(int(ckpt.step), weights_path))
                model.save_weights(weights_path, overwrite=True)

            print('----------------------------------------------------------------------------------')

        list_mious.append(va_miou)

        # Switch loss function to Lovàsz-Softmax if epoch reaches threshold (based on model cfg)
        if epoch == loss_switch_ep:
            loss_func = 'lovasz'
            print('//////////////////////////////////////////////////////////////////////////////////')
            print('Switching loss function to Lovàsz-Softmax..')
            print('//////////////////////////////////////////////////////////////////////////////////')

    print('----------------------------------------------------------------------------------')
    print('     TRAINING END...')
    print('----------------------------------------------------------------------------------')

    # if FLAGS.save:
    #     save_path = 'models/infer_v4_1_DeepGCNv2_xyzrgb_nn10_200_vkitti_pretrained'
    #     model.save(save_path)
    #     print('     Model saved to {}'.format(save_path))
    #     print('==================================================================================')
