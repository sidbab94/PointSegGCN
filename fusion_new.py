import numpy as np
from functools import wraps
from timeit import Timer
import matplotlib.pyplot as plt
from PIL import Image
from utils.graph_gen import flann_search
from scipy import sparse as sp

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf

TF_FLT = tf.float32
TF_INT = tf.int32


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
def fusion(calib, pc, img):
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

    rgb_coords = tf.cast(tf.round(rgb_coords), TF_INT)
    img = tf.gather_nd(img, rgb_coords)

    # mapped rgb values for projected velo points, normalized
    pts_2d_rgb = tf.divide(img, tf.constant([255.]))

    mod_pc = tf.transpose(tf.gather(pc, inds, axis=1))
    mod_pc = tf.concat([mod_pc, pts_2d_rgb], axis=1)

    return mod_pc

@timing
def tf_adjacency(mod_pc):

    dist, idx = flann_search(mod_pc.numpy(), 10)

    dist = tf.cast(dist, TF_FLT)
    idx = tf.cast(idx, TF_INT)

    M, k = dist.shape

    assert M, k == idx.shape
    assert tf.greater_equal(tf.reduce_min(dist), 0)

    std = tf.math.reduce_std(dist, 0)
    mu = tf.reduce_mean(dist, 0)
    dist = tf.divide(tf.subtract(dist, mu), std)

    I = tf.repeat(tf.range(0, M), k)
    J = tf.reshape(idx, M * k)
    a_inds = tf.cast(tf.stack([I, J], axis=1), tf.int64)
    a_vals = tf.reshape(dist, M * k)
    a_shape = tf.constant([M, M], dtype=tf.int64)

    # A = tf.sparse.SparseTensor(a_inds, a_vals, a_shape)
    A = sp.csr_matrix((a_vals.numpy(), (I.numpy(), J.numpy())), shape=(M, M))
    return A

@timing
def tf_reduce(pc, red_dist=8.0):

    eucl = tf.math.reduce_euclidean_norm(pc[:, :3], axis=1)
    valid_idx = tf.squeeze(tf.where(tf.less(eucl, red_dist)))

    return tf.gather(pc, valid_idx, axis=0)

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

    mod_pc = pc.T[inds, :]
    mod_pc = np.hstack((mod_pc, pts_2d_rgb))

    return mod_pc


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
pc = np.load('sample.npy')
img = np.asarray(Image.open('sample.png'))

# arr = np_proc(data, pc, img)
# tensor = fusion(data, pc, img)

pc_red = tf_reduce(pc)
adj = tf_adjacency(pc_red)


from utils.visualization import PC_Vis

PC_Vis.draw_graph(pc[:, :3], adj)