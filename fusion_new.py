import numpy as np
from functools import wraps
from timeit import Timer
import matplotlib.pyplot as plt
from PIL import Image

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf

TF_FLT = tf.float32


def scatter(array):
    plt.scatter(array[0, :], array[1, :])
    plt.show()


def timing(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        result = f(*args, **kwargs)
        t = Timer(lambda: result).timeit()
        print('Elapsed time for "{}" : {} ms'.format(f.__name__, t * 1e03))
        return result

    return wrapper


@timing
def tf_proc(calib, pc, img):
    extr = tf.cast(tf.reshape(calib['Ext'], (3, 4)), TF_FLT)  # extrinsic matrix
    intr = tf.cast(tf.reshape(calib['Int'], (3, 4)), TF_FLT)  # intrinsic matrix
    pc = tf.cast(tf.transpose(pc), TF_FLT)
    img = tf.cast(img, TF_FLT)
    h, w, _ = img.shape

    scaler = tf.constant([0, 0, 0, 1], TF_FLT)  # perspective scaler
    transf = tf.concat([extr, tf.expand_dims(scaler, 0)], axis=0)

    proj_mat = tf.matmul(intr, transf)  # projection matrix Velo->Cam

    pts = tf.concat([pc, tf.ones([1, pc.shape[1]])], axis=0)
    pts_2d = tf.matmul(proj_mat, pts)
    pts_2d = tf.divide(pts_2d[:2, :], pts_2d[2, :])[:2, :]  # Velo->Cam projected points

    # pc points constrained to img dims
    inds = tf.where(
        (pts_2d[0, :] < w) & (pts_2d[0, :] >= 0) &
        (pts_2d[1, :] < h) & (pts_2d[1, :] >= 0) &
        (pc[0, :] > 0)
    )
    inds = tf.squeeze(inds, axis=1)
    pc_img = tf.transpose(tf.gather(pts_2d, inds, axis=1))

    # map coordinates to rgb values
    rgb_coords = tf.transpose(
        tf.concat([tf.expand_dims(pc_img[:, 1], 0),
                   tf.expand_dims(pc_img[:, 0], 0)], axis=0))

    rgb_coords = tf.cast(tf.round(rgb_coords), dtype=tf.int32)
    img = tf.gather_nd(img, rgb_coords)

    # mapped rgb values for projected velo points
    pts_2d_rgb = tf.divide(img, tf.constant([255.]))

    return inds, pts_2d_rgb


@timing
def np_proc(calib, pc, img):
    extr = calib['Ext'].reshape(3, 4)
    intr = calib['Int'].reshape(3, 4)
    pc = np.array(pc.transpose(), dtype=np.float32)
    img = img.astype(np.float32)
    h, w, _ = img.shape

    transf = np.vstack((extr, np.array([0., 0., 0., 1.], dtype=np.float32)))
    proj_mat = intr @ transf

    pts = np.vstack((pc, np.ones((1, pc.shape[1]))))
    pts_2d = proj_mat @ pts
    pts_2d = (pts_2d[:2, :] / pts_2d[2, :])[:2, :]

    inds = np.where((pts_2d[0, :] < w) & (pts_2d[0, :] >= 0) &
                    (pts_2d[1, :] < h) & (pts_2d[1, :] >= 0) &
                    (pc[0, :] > 0)
                    )[0]
    pc_img = np.array(pts_2d[:, inds].transpose(), dtype=np.float32)

    rgb_coords = np.vstack((pc_img[:, 1], pc_img[:, 0])).transpose()

    rgb_coords = (rgb_coords.round() - 1).astype(np.int32)
    pts_2d_rgb = img[rgb_coords[:, 0], rgb_coords[:, 1]] / 255

    return inds, pts_2d_rgb


def read_calib_file(filepath):
    data = {}
    with open(filepath, 'r') as f:
        for line in f.readlines():
            line = line.rstrip()
            if len(line) == 0: continue
            key, value = line.split(':', 1)
            # The only non-float values in these files are dates, which
            # we don't care about anyway
            try:
                data[key] = np.array([float(x) for x in value.split()], dtype=np.float32)
            except ValueError:
                pass
    return data


data = read_calib_file('calib.txt')
pc = np.load('sample.npy')[:, :3]
img = np.asarray(Image.open('sample.png'))

arr = np_proc(data, pc, img)

tensor = tf_proc(data, pc, img)

# assert np.allclose(arr, tensor, atol=0)
