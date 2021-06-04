import numpy as np
import struct
from functools import wraps
from timeit import Timer
import matplotlib.pyplot as plt
from PIL import Image
from utils.graph_gen import flann_search
from utils.visualization import PC_Vis

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf


#####
'''
TO DO:

- Encapsulate reader functions into single pipeline
- Wrap tf modular functions into a single function (inputs = X, Y, Img, Calib as np arrays)

'''
#####


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
    extr = tf.cast(tf.reshape(calib['Tr'], (3, 4)), TF_FLT)  # extrinsic matrix
    intr = tf.cast(tf.reshape(calib['P2'], (3, 4)), TF_FLT)  # intrinsic matrix
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

    A = tf.sparse.SparseTensor(a_inds, a_vals, a_shape)
    # A = sp.csr_matrix((a_vals.numpy(), (I.numpy(), J.numpy())), shape=(M, M))
    return A

@timing
def tf_reduce(fused_pc, red_dist=35.0):

    eucl = tf.math.reduce_euclidean_norm(fused_pc[:, :3], axis=1)
    valid_idx = tf.squeeze(tf.where(tf.less(eucl, red_dist)))

    return tf.gather(fused_pc, valid_idx, axis=0)


def tf_sparse_to_numpy(sp_input):

    return tf.sparse.to_dense(tf.sparse.reorder(sp_input)).numpy()

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


def read_bin_velodyne(pc_path, include_intensity=False):
    '''
    Reads velodyne binary file and converts it to a numpy.nd.array
    :param pc_path: SemanticKITTI binary scan file path
    :return: point cloud array
    '''
    pc_list = []
    with open(pc_path, 'rb') as f:
        content = f.read()
        pc_iter = struct.iter_unpack('ffff', content)
        for idx, point in enumerate(pc_iter):
            if include_intensity:
                pc_list.append([point[0], point[1], point[2], point[3]])
            else:
                pc_list.append([point[0], point[1], point[2]])
    return np.asarray(pc_list, dtype=np.float32)



BASE_DIR = '/media/baburaj/Seagate Backup Plus Drive/SemanticKITTI/dataset/sequences'
seq_path = os.path.join(BASE_DIR, '00')
pc = read_bin_velodyne(os.path.join(seq_path, 'velodyne', '000000.bin'))
img_path = os.path.join(seq_path, 'image_2', '000000.png')
# img = img_to_array(load_img(img_path))
img = np.asarray(Image.open(img_path))
data = read_calib_file(os.path.join(seq_path, 'calib.txt'))


fused_pc = fusion(data, pc, img)

# arr = np_proc(data, pc, img)

pc_red = tf_reduce(fused_pc)

# PC_Vis.draw_pc(pc_red, True)

adj = tf_adjacency(pc_red)

adj_array = tf.sparse.to_dense(tf.sparse.reorder(adj)).numpy()

PC_Vis.draw_graph(pc_red[:, :3], adj_array)