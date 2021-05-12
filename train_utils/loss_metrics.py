from __future__ import print_function, division
import tensorflow as tf
import numpy as np
from tensorflow.keras import backend as K
from tensorflow.keras.losses import SparseCategoricalCrossentropy

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

def tversky_loss(y_true, y_pred, class_weights=None):
    alpha = 0.3
    beta = 1 - alpha

    if y_true.ndim < 2:
        y_true = K.cast(K.one_hot(y_true, num_classes=20), 'float32')

    y_pred = K.cast(y_pred, 'float32')

    ones = K.ones(K.shape(y_true))
    p0 = y_pred  # proba that voxels are class i
    p1 = ones - y_pred  # proba that voxels are not class i
    g0 = y_true
    g1 = ones - y_true

    num = K.sum(p0 * g0, (0, 1))
    den = num + alpha * K.sum(p0 * g1, (0, 1)) + beta * K.sum(p1 * g0, (0, 1))

    T = K.sum(num / den)  # when summing over classes, T has dynamic range [0 Ncl]
    # Ncl = K.cast(K.shape(y_true)[-1], 'float32')
    return 1 - T


def focal_tversky_loss(y_true, y_pred, gamma=1.1, class_weights=None):
    tv = tversky_loss(y_true, y_pred, class_weights)
    return K.pow(K.abs(tv), gamma)


def focal_tversky_weighted(y_true, logits, class_weights=None):
    y_true = tf.cast(tf.one_hot(y_true, 20), tf.float32)
    o = tf.nn.softmax_cross_entropy_with_logits(y_true, logits) + focal_tversky_loss(y_true, logits)

    if class_weights is not None:
        class_weights = tf.cast(class_weights, tf.float32)
        weights = tf.reduce_sum(class_weights * y_true, axis=-1)
        o = o * weights

    return tf.reduce_mean(o)


def generalized_dice_coeff(y_true, y_pred):
    y_true = K.cast(K.one_hot(y_true, num_classes=20), 'float32')
    y_pred = K.cast(y_pred, 'float32')
    w = K.sum(y_true)
    w = 1/(w**2+0.000001)
    # Compute gen dice coef:
    numerator = y_true*y_pred
    numerator = w*K.sum(numerator)
    numerator = K.sum(numerator)

    denominator = y_true+y_pred
    denominator = w*K.sum(denominator)
    denominator = K.sum(denominator)

    gen_dice_coef = 2*numerator/denominator

    return gen_dice_coef


def generalized_dice_loss(y_true, y_pred, class_weights=None):
    return 1 - generalized_dice_coeff(y_true, y_pred)


def dice_loss(y_true, y_pred, class_weights=None):
    if y_true.ndim < 2:
        y_true = tf.cast(tf.one_hot(y_true, 20), tf.float32)
    y_pred = tf.math.sigmoid(y_pred)
    numerator = 2 * tf.reduce_sum(y_true * y_pred)
    denominator = tf.reduce_sum(y_true + y_pred)

    return 1 - numerator / denominator


def dice_cross_entropy(y_true, logits, class_weights=None):

    y_true = tf.cast(tf.one_hot(y_true, 20), tf.float32)
    o = tf.nn.softmax_cross_entropy_with_logits(y_true, logits) + dice_loss(y_true, logits)

    if class_weights is not None:
        class_weights = tf.cast(class_weights, tf.float32)
        weights = tf.reduce_sum(class_weights * y_true, axis=1)
        o = o * weights

    return tf.reduce_mean(o)

def weighted_cross_entropy(y_true, logits, class_weights=None):

    y_true = tf.cast(tf.one_hot(y_true, 20), tf.float32)
    o = tf.nn.weighted_cross_entropy_with_logits(y_true, logits, pos_weight=1.2)

    if class_weights is not None:
        class_weights = tf.cast(class_weights, tf.float32)
        weights = tf.reduce_sum(class_weights * y_true, axis=1)
        o = o * weights

    return tf.reduce_mean(o)

def cross_entropy(y_true, logits, class_weights=None):

    y_true = tf.cast(tf.one_hot(y_true, 20), tf.float32)
    o = tf.nn.softmax_cross_entropy_with_logits(y_true, logits)
    if class_weights is not None:
        class_weights = tf.cast(class_weights, tf.float32)
        weights = tf.reduce_sum(class_weights * y_true, axis=-1)
        print(weights)
        o = o * weights

    return tf.reduce_mean(o)

def sparse_cross_entropy(y_true, y_pred, class_weights=None):

    loss_fn = SparseCategoricalCrossentropy(from_logits=True)
    o = loss_fn(y_true, y_pred)
    print(o.shape)
    if class_weights is not None:
        y_true = tf.cast(tf.one_hot(y_true, 20), y_pred.dtype)
        class_weights = tf.cast(class_weights, tf.float32)
        weights = tf.reduce_sum(class_weights * y_true, axis=-1)
        print(weights.shape)
        o = o * weights

    return tf.reduce_mean(o)

if __name__ == '__main__':

    y_true = np.array([1, 1, 3, 1, 2, 3], dtype=np.float32)
    y_pred = np.array([[0.7, 0.9, 0.2, 0.0, 0.0],
                       [0.0, 0.48, 0.23, 0.3, 0.0],
                       [0.5, 0.0, 0.2, 0.6, 0.0],
                       [0.4, 0.7, 0.6, 0.3, 0.0],
                       [0.0, 0.15, 0.05, 0.55, 0.7],
                       [0.2, 0.0, 0.11, 0.5, 0.8]], dtype=np.float32)

    classes = np.eye(5)
    y_true = classes[y_true.astype(int).reshape(-1)]
    pred_labels = np.argmax(y_pred, axis=-1)
    print(pred_labels)

    ones = np.ones_like(y_true)

    p0 = y_pred
    p1 = ones - y_pred
    g0 = y_true
    g1 = ones - y_true

    alpha = 0.7
    beta = 1 - alpha

    TP = np.sum(y_true * y_pred, (0, 1))
    FN = np.sum(p0 * g1, (0, 1))
    FP = np.sum(p1 * g0, (0, 1))

    TI = TP / (TP + alpha*FN + beta*FP)
    TL = 1 - TI

    print(TL)

