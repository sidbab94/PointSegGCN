import os
import numpy as np
import tensorflow as tf
import scipy.sparse as sp
from pyflann import FLANN

from . import readers as io

np.seterr(divide='ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)


def preprocess(file_path, cfg, train=False):
    '''
    Pre-processing pipeline which:
        - reads scan attributes such as the LiDAR point cloud, RGB image, calibration parameters
          and the labels
        - Performs projection-based fusion of RGB data with geometrical LiDAR data
        - Constructs a sparse distance-based adjacency graph from the fused point cloud
    :param file_path: LiDAR scan file path
    :param cfg: model config dictionary
    :param train: Boolean flag corresponding to inputs must be preprocessed for training, or otherwise.
    :return: tuple of preprocessed inputs, ready for Forward Pass
    '''
    pc, y, calib, img = io.read_scan_attr(file_path, cfg)
    pc, y = lidar_rgb_fusion(pc, y, calib, img)
    # Restrict point cloud z-axis range, allowing only a limited, but relevant subset of points
    pc, y = constrain_pc(pc, y)
    a = construct_graph(pc, cfg['k_value'])

    if cfg['augment'] and train:
        # Augment preprocessed inputs by introducing random perturbations
        pc, a, y = augment_scan([pc, a, y], cfg['batch_size'])

    # Convert sparse CSR adjacency matrix to a TF sparse tensor
    a = csr_to_tensor(a)

    return pc, a, y


def lidar_rgb_fusion(pc, y, calib, img):
    '''
    Carries out fusion of LiDAR geometrical coordinates with their corresponding RGB vectors,
    extracted from a single camera capture.
    :param pc: LiDAR point cloud --> NumPy array [N x 3]
    :param y: Point-wise labels --> NumPy array [N, ]
    :param calib: Parsed calibration parameters
    :param img: RGB image corresponding to scan --> NumPy array [H x W x 3]
    :return: tuple of RGB-infused LiDAR point cloud and corresponding label vector
    '''
    # Extrinsic calibration parameters
    velo2cam = np.vstack((calib['Tr'].reshape(3, 4),
                          np.array([0., 0., 0., 1.], dtype=np.float32)))
    # Intrinsic calibration parameters
    cam_intr = calib['P2'].reshape((3, 4))

    # Projection matrix
    proj_mat = cam_intr @ velo2cam

    h, w, _ = img.shape
    N = pc.shape[0]

    # Project LiDAR points into camera's 2D space
    pts_2d = np.array(pc[:, :3].transpose(), dtype=np.float32)
    pts_2d = np.vstack((pts_2d, np.ones((1, N), dtype=np.float32)))
    pts_2d = proj_mat @ pts_2d
    # Normalize result wrt. depth
    pts_2d[:2, :] /= pts_2d[2, :]

    # Constrain 2D-projected point cloud to camera FOV
    inds = np.where((pts_2d[0, :] < w) & (pts_2d[0, :] >= 0) &
                    (pts_2d[1, :] < h) & (pts_2d[1, :] >= 0) &
                    (pc[:, 0] > 0)
                    )[0]
    pts_2d = pts_2d[:, inds].transpose()

    # Retrieve pixel RGB values for projected point coordinates
    rgb_coords = np.vstack((pts_2d[:, 1], pts_2d[:, 0])).transpose()
    rgb_coords = (np.round(rgb_coords) - 1).astype(np.int32)
    rgb_array = img[rgb_coords[:, 0], rgb_coords[:, 1]] / 255

    # Concatenate extracted RGB array to original LiDAR point cloud (in camera FOV)
    pc = np.hstack((pc[inds, :], rgb_array))
    y = y[inds]

    return pc, y


def constrain_pc(pc, y, dist=35.0):
    '''
    Reduces point cloud range to a limited depth
    :param pc: LiDAR point cloud --> NumPy array [N x 3]
    :param y: Point-wise labels --> NumPy array [N, ]
    :param dist: Threshold distance in meters, beyond which points are ignored
    :return: Z-reduced point cloud
    '''
    valid_idx = np.where(np.linalg.norm(pc[:, :3], axis=1) < dist)
    pc = pc[valid_idx[0], :]
    y = y[valid_idx[0]]

    return pc, y


def nn_search(points, nn):
    '''
    Performs Nearest-Neighbour search for the input point cloud, using a FLANN-based KD-Tree algorithm.
    :param points: LiDAR point cloud --> NumPy array [N x 3]
    :param nn: Number of nearest neighbours to index
    :return: Distances and indices of 'nn' nearest neighbours of every point in the input point cloud
    '''
    flann = FLANN()
    # Build KD-Tree from the LiDAR point cloud
    params = flann.build_index(points, algorithm="kdtree_simple",
                               target_precision=0.8, log_level="info")
    # Nearest neigbour queries
    idx, dist = flann.nn_index(points, nn, checks=params["checks"])
    return dist, idx


def normalize_A(A):
    '''
    Carries out the 'Renormalization trick' for as a preprocessing step for GCNs.
    :param A: Sparse adjacency matrix --> CSR matrix
    :return: Normalized sparse adjacency matrix
    '''
    n, m = A.shape
    diags = A.sum(axis=1).flatten()
    D = sp.spdiags(diags, [0], m, n, format="csr")
    D = D.power(-0.5)
    A = D.dot(A).dot(D)

    return A


def construct_graph(pc, nn=10):
    '''
    Construct a distance-based sparse adjacency graph from the input LiDAR point cloud.
    :param pc: LiDAR point cloud --> NumPy array [N x 3]
    :param nn: Number of nearest neighbours to index
    :return: Normalized distance-based sparse adjacency matrix
    '''
    dist, idx = nn_search(pc[:, :3], nn)
    assert dist.shape == (pc.shape[0], nn)
    M, k = dist.shape
    assert M, k == idx.shape

    # Edge weights computed based on variance
    sigma2 = np.mean(dist[:, -1]) ** 2
    dist = np.exp(- dist ** 2 / sigma2)

    # Weight matrix construction
    I = np.arange(0, M).repeat(k)
    J = idx.reshape(M * k)
    V = dist.reshape(M * k)

    # Insert distances and indices into a CSR matrix
    A = sp.csr_matrix((V, (I, J)), shape=(M, M))
    A += sp.eye(A.shape[0])

    return normalize_A(A)


def jitter_pc(pc, sigma=0.01, clip=0.05):
    '''
    Adds random geometric jitter to the input LiDAR point cloud
    :param pc: LiDAR point cloud --> NumPy array [N x 3]
    :param sigma: Noise factor
    :param clip: Threshold for clipping matrix
    :return: LiDAR point cloud with jitter
    '''
    N, C = pc.shape
    assert (clip > 0)
    jittered_pc = np.clip(sigma * np.random.randn(N, C), -1 * clip, clip)
    jittered_pc += pc

    return jittered_pc


def scale_pc(pc, scale_low=0.9, scale_high=1.1):
    '''
    Multiplies the point cloud array by a uniform distribution
    :param pc: LiDAR point cloud --> NumPy array [N x 3]
    :param scale_low: Lower threshold of uniform distribution
    :param scale_high: Higher threshold of uniform distribution
    :return: Scaled LiDAR point cloud
    '''
    N, C = pc.shape
    scales = np.random.uniform(scale_low, scale_high, C)
    pc *= scales
    return pc


def augment_scan(inputs, bs, angle=90, axis='z'):
    '''
    Augments input training sample with perturbed versions
    :param inputs: Tuple of training inputs
    :param angle: Rotational offset for each augmented sample wrt the original
    :param axis: Axis wrt which rotational augmentation is performed
    :return: Augmented training batch
    '''
    x, a, y = inputs

    aug_x = []
    rot_mask = np.linspace(0.0, 3.0, bs)
    rads = np.array(np.radians(rot_mask * angle))
    for i in range(len(rot_mask)):
        # Get rotational transformation matrix from angle and axis
        rot = get_rot_matrix(axis, rads[i])
        rotated = rot @ x[:, :3].T
        rot_pc = np.hstack((rotated.T, x[:, 3:]))
        if (i != 0):
            if (i % 2 == 0):
                rot_pc = jitter_pc(rot_pc)
            if (i % 3 == 0):
                rot_pc = scale_pc(rot_pc)
        aug_x.append(rot_pc)

    x = np.vstack(aug_x)
    # Adjacency matrix duplicated along block diagonals
    a = sp.block_diag([a] * bs)
    y = np.tile(y, bs)

    return x, a, y


def get_rot_matrix(axis, rads):
    '''
    Obtains rotational transformation matrix from input axis and angle
    :param axis: Axis wrt which rotational augmentation is performed
    :param rads: Rotational offset for each augmented sample wrt the original, in radians
    :return: Rotational transformation matrix
    '''
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
    '''
    Converts SciPy's CSR matrix into a TF sparse tensor
    :param A: Input sparse CSR adjacency matrix (SciPy)
    :return: TF sparse tensor
    '''
    row, col, values = sp.find(A)
    out = tf.sparse.SparseTensor(indices=np.array([row, col]).T,
                                 values=values,
                                 dense_shape=A.shape)
    A = tf.sparse.reorder(out)

    return A
