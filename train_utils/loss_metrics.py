from __future__ import print_function, division
import tensorflow as tf
import numpy as np
import tensorflow_addons as tfa

""" Title: Lovasz-Softmax and Jaccard hinge loss in Tensorflow
Author: Maxim Berman
Date: 2018
Availability: https://github.com/bermanmaxim/LovaszSoftmax """


def lovasz_grad(gt_sorted):
    """
    Computes gradient of the Lovasz extension w.r.t sorted errors
    See Alg. 1 in paper
    """
    gts = tf.reduce_sum(gt_sorted)
    intersection = gts - tf.cumsum(gt_sorted)
    union = gts + tf.cumsum(1. - gt_sorted)
    jaccard = 1. - intersection / union
    jaccard = tf.concat((jaccard[0:1], jaccard[1:] - jaccard[:-1]), 0)
    return jaccard


def lovasz_softmax(probas, labels, classes='present', per_image=False, ignore=None, order='BHWC'):
    """
    Multi-class Lovasz-Softmax loss
      probas: [B, H, W, C] or [B, C, H, W] Variable, class probabilities at each prediction (between 0 and 1)
              Interpreted as binary (sigmoid) output with outputs of size [B, H, W].
      labels: [B, H, W] Tensor, ground truth labels (between 0 and C - 1)
      classes: 'all' for all, 'present' for classes present in labels, or a list of classes to average.
      per_image: compute the loss per image instead of per batch
      ignore: void class labels
      order: use BHWC or BCHW
    """
    if per_image:
        def treat_image(prob_lab):
            prob, lab = prob_lab
            prob, lab = tf.expand_dims(prob, 0), tf.expand_dims(lab, 0)
            prob, lab = flatten_probas(prob, lab, ignore, order)
            return lovasz_softmax_flat(prob, lab, classes=classes)

        losses = tf.map_fn(treat_image, (probas, labels), dtype=tf.float32)
        loss = tf.reduce_mean(losses)
    else:
        loss = lovasz_softmax_flat(*flatten_probas(probas, labels, ignore, order), classes=classes)
    return loss


def lovasz_softmax_flat(labels, probas, classes='present', class_weights=None):
    """
    Multi-class Lovasz-Softmax loss
      probas: [P, C] Variable, class probabilities at each prediction (between 0 and 1)
      labels: [P] Tensor, ground truth labels (between 0 and C - 1)
      classes: 'all' for all, 'present' for classes present in labels, or a list of classes to average.
    """
    C = probas.shape[1]
    losses = []
    present = []
    class_to_sum = list(range(C)) if classes in ['all', 'present'] else classes
    for c in class_to_sum:
        fg = tf.cast(tf.equal(labels, c), probas.dtype)  # foreground for class c
        if classes == 'present':
            present.append(tf.reduce_sum(fg) > 0)
        if C == 1:
            if len(classes) > 1:
                raise ValueError('Sigmoid output possible only with 1 class')
            class_pred = probas[:, 0]
        else:
            class_pred = probas[:, c]
        errors = tf.abs(fg - class_pred)
        errors_sorted, perm = tf.nn.top_k(errors, k=tf.shape(errors)[0], name="descending_sort_{}".format(c))
        fg_sorted = tf.gather(fg, perm)
        grad = lovasz_grad(fg_sorted)
        losses.append(
            tf.tensordot(errors_sorted, tf.stop_gradient(grad), 1, name="loss_class_{}".format(c))
        )
    if len(class_to_sum) == 1:  # short-circuit mean when only one class
        return losses[0]
    losses_tensor = tf.stack(losses)
    if classes == 'present':
        present = tf.stack(present)
        losses_tensor = tf.boolean_mask(losses_tensor, present)

    if class_weights is not None:
        class_weights = tf.cast(class_weights, tf.float32)
        if classes == 'present':
            class_weights = tf.boolean_mask(class_weights, present)
        weights = class_weights
        losses_tensor = losses_tensor * weights

    loss = tf.reduce_mean(losses_tensor)
    return loss


def flatten_probas(probas, labels, ignore=None, order='BHWC'):
    """
    Flattens predictions in the batch
    """
    if len(probas.shape) == 3:
        probas, order = tf.expand_dims(probas, 3), 'BHWC'
    if order == 'BCHW':
        probas = tf.transpose(probas, (0, 2, 3, 1), name="BCHW_to_BHWC")
        order = 'BHWC'
    if order != 'BHWC':
        raise NotImplementedError('Order {} unknown'.format(order))
    C = probas.shape[3]
    probas = tf.reshape(probas, (-1, C))
    labels = tf.reshape(labels, (-1,))
    if ignore is None:
        return probas, labels
    valid = tf.not_equal(labels, ignore)
    vprobas = tf.boolean_mask(probas, valid, name='valid_probas')
    vlabels = tf.boolean_mask(labels, valid, name='valid_labels')
    return vprobas, vlabels


'''
dice_cross_entropy() obtained (partially) from:
https://lars76.github.io/2018/09/27/loss-functions-for-segmentation.html
'''


def dice_loss(y_true, y_pred):
    y_pred = tf.math.sigmoid(y_pred)
    numerator = 2 * tf.reduce_sum(y_true * y_pred)
    denominator = tf.reduce_sum(y_true + y_pred)

    return 1 - numerator / denominator

def dice_cross_entropy(y_true, logits, class_weights=None):

    y_true = tf.cast(y_true, tf.float32)

    o = tf.nn.softmax_cross_entropy_with_logits(y_true, logits) + dice_loss(y_true, logits)

    if class_weights is not None:
        class_weights = tf.cast(class_weights, tf.float32)
        weights = tf.reduce_sum(class_weights * y_true, axis=1)
        o = o * weights

    return tf.reduce_mean(o)


def dice_cross_entropy_weighted(y_true, predictions):

    y_true = tf.cast(y_true, tf.float32)
    o = tf.nn.sigmoid_cross_entropy_with_logits(y_true, predictions) + dice_loss(y_true, predictions)

    return tf.reduce_mean(o)


def one_hot_encoding(y_true):
    '''
    Does one-hot encoding on 1-dimensional sparse label array
    :param y_true: 1-dimensional point-wise ground truth label array
    :param cfg: model configuration dictionary
    :return: 2D one-hot encoded label array
    '''

    N = y_true.shape[0]
    one_hot = np.zeros((N, 20))

    for row in range(one_hot.shape[0]):
        one_hot[row, y_true[row]] = 1

    return one_hot

def tversky(y_true, y_pred, smooth=1, alpha=0.7):
    y_pred = tf.argmax(y_pred, axis=-1)
    y_pred = tf.cast(y_pred, tf.float32)
    y_true = tf.cast(y_true, tf.float32)

    true_pos = tf.reduce_sum(y_true * y_pred)
    false_neg = tf.reduce_sum(y_true * (1 - y_pred))
    false_pos = tf.reduce_sum((1 - y_true) * y_pred)

    return (true_pos + smooth) / (true_pos + (alpha * false_neg) + ((1 - alpha) * false_pos) + smooth)


def tversky_loss(y_true, y_pred):
    return 1 - tversky(y_true, y_pred)


def focal_tversky_loss(y_true, y_pred, gamma=0.75, class_weights=None):
    tv = tversky(y_true, y_pred)
    return tf.pow(tf.abs(1 - tv), gamma)

def sigmoid_focal_xentropy(y_true, y_pred):
    y_pred = tf.argmax(y_pred, axis=-1)
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    return tfa.losses.sigmoid_focal_crossentropy(y_true, y_pred)

if __name__ == '__main__':

    y_pred = np.array([1, 2, 3, 1, 4, 3], dtype=np.float32)
    y_true = np.array([1, 2, 3, 1, 2, 3], dtype=np.float32)

    fl = tfa.losses.SigmoidFocalCrossEntropy()
    loss = tfa.losses.sigmoid_focal_crossentropy(y_true, y_pred)

    print(loss)