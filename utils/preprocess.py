import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import numpy as np
import scipy.sparse as sp
from pyflann import FLANN

from . import readers as io


def preprocess(file_path, cfg, train=False, test_run=False):
    pc, y, calib, img = io.read_scan_attr(file_path, cfg, test_run)
    pc, y = lidar_rgb_fusion(pc, y, calib, img)
    pc, y = constrain_pc(pc, y)
    a = adjacency(pc, 10)

    if cfg['augment'] and train:
        pc, a, y = augment_scan([pc, a, y], cfg['batch_size'])

    a = csr_to_tensor(a)

    return pc, a, y


def lidar_rgb_fusion(pc, y, calib, img):
    velo2cam = np.vstack((calib['Tr'].reshape(3, 4),
                          np.array([0., 0., 0., 1.], dtype=np.float32)))
    cam_intr = calib['P2'].reshape((3, 4))

    proj_mat = cam_intr @ velo2cam

    h, w, _ = img.shape
    N = pc.shape[0]

    pts_2d = np.array(pc[:, :3].transpose(), dtype=np.float32)
    pts_2d = np.vstack((pts_2d, np.ones((1, N), dtype=np.float32)))
    pts_2d = proj_mat @ pts_2d
    pts_2d[:2, :] /= pts_2d[2, :]

    inds = np.where((pts_2d[0, :] < w) & (pts_2d[0, :] >= 0) &
                    (pts_2d[1, :] < h) & (pts_2d[1, :] >= 0) &
                    (pc[:, 0] > 0)
                    )[0]

    pts_2d = pts_2d[:, inds].transpose()

    rgb_coords = np.vstack((pts_2d[:, 1], pts_2d[:, 0])).transpose()
    rgb_coords = (np.round(rgb_coords) - 1).astype(np.int32)
    rgb_array = img[rgb_coords[:, 0], rgb_coords[:, 1]] / 255

    pc = np.hstack((pc[inds, :], rgb_array))
    y = y[inds]

    return pc, y


def constrain_pc(pc, y, dist=35.0):
    valid_idx = np.where(np.linalg.norm(pc[:, :3], axis=1) < dist)
    pc = pc[valid_idx[0], :]
    y = y[valid_idx[0]]

    return pc, y


def nn_search(points, nn):
    flann = FLANN()
    params = flann.build_index(points, algorithm="kdtree_simple",
                               target_precision=0.8, log_level="info")
    idx, dist = flann.nn_index(points, nn, checks=params["checks"])
    return dist, idx


def normalize_A(A):
    n, m = A.shape
    diags = A.sum(axis=1).flatten()
    D = sp.spdiags(diags, [0], m, n, format="csr")
    D = D.power(-0.5)
    A = D.dot(A).dot(D)

    return A


def adjacency(pc, nn=10):
    dist, idx = nn_search(pc[:, :3], nn)

    M, k = dist.shape
    assert M, k == idx.shape

    # Edge weights computed based on variance
    sigma2 = np.mean(dist[:, -1]) ** 2
    dist = np.exp(- dist ** 2 / sigma2)

    # Weight matrix construction
    I = np.arange(0, M).repeat(k)
    J = idx.reshape(M * k)
    V = dist.reshape(M * k)

    A = sp.csr_matrix((V, (I, J)), shape=(M, M))

    return normalize_A(A)


def augment_scan(inputs, bs, angle=90, axis='z'):

    x, a, y = inputs

    aug_x = []
    rot_mask = np.linspace(0.0, 3.0, bs)
    rads = np.array(np.radians(rot_mask * angle))
    for i in range(len(rot_mask)):
        rot = get_rot_matrix(axis, rads[i])
        rotated = rot @ x[:, :3].T
        rot_pc = np.hstack((rotated.T, x[:, 3:]))
        if (i % 2 == 0):
            rot_pc[:, 4:] += np.random.normal(0, 0.1, size=(x.shape[0], 3))
        aug_x.append(rot_pc)

    x = np.vstack(aug_x)
    a = sp.block_diag([a] * bs)
    y = np.tile(y, bs)

    return x, a, y


def get_rot_matrix(axis, rads):

    cosa = np.cos(rads)
    sina = np.sin(rads)

    if axis == 'x':
        return np.array([[1, 0, 0], [0, cosa, sina], [0, -sina, cosa]])
    elif axis == 'y':
        return np.array([[cosa, 0, -sina], [0, 1, 0], [sina, 0, cosa]])
    elif axis == 'z':
        return np.array([[cosa, sina, 0], [-sina, cosa, 0], [0, 0, 1]])
    else:
        raise Exception('Invalid axis provided')


def csr_to_tensor(A):
    row, col, values = sp.find(A)
    out = tf.sparse.SparseTensor(indices=np.array([row, col]).T,
                                 values=values,
                                 dense_shape=A.shape)
    A = tf.sparse.reorder(out)

    return A


if __name__ == '__main__':
    scan_file = 'D:/SemanticKITTI/dataset/sequences/08/velodyne/002989.bin'
    cfg = io.get_cfg_params()

    x, a, y = preprocess(scan_file, cfg, train=True)
