from pathlib import PurePath
import cv2
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from tensorflow import sparse, SparseTensor
from scipy import sparse as sp

from preproc_utils.readers import *
from preproc_utils.sensor_fusion import SensorFusion
from preproc_utils.graph_gen import compute_adjacency as compute_graph, normalize_A


class Preprocess:
    """
    Pre-processing pipeline for raw LiDAR scan data, as follows:

    get_scan_data() ----> get_modality() ----> get_graph()

    Arguments:
        cfg_dict: model configuration dictionary, passed from training loop

    """

    def __init__(self, cfg_dict):

        self.cfg = cfg_dict
        self.invalid_scan = False

        self.features = self.cfg['feature_spec'].split('_')
        self.feat_len = sum(c for c in [len(i) for i in self.features])

    def assess_scan(self, scan_path, testds=False):
        '''
        Scan path is processed from start to end, all necessary training inputs obtained
        :param scan_path: file path containing velodyne point cloud (.bin)
        :param testds: boolean flag indicating "test" scenario (no ground truth)
        :return: pc_rgb, A, labels_rgb
        '''

        self.outs = []
        self.testds = testds
        self.scan_path = scan_path
        self.get_scan_data()
        if self.cfg['name'] == 'semantickitti':
            # since semantickitti doesn't have inherent colour modality
            self.get_modality()
        if 'd' in self.features:
            self.get_depth()
        self.reduce_data()
        self.get_graph()

        self.outs.append(self.pc)
        self.outs.append(self.A)
        if self.testds is False:
            self.outs.append(self.labels)

        return self.outs

    def get_scan_data(self):
        '''
        Obtain scan attributes from the file path provided
        Scan attributes are:
            pc (point cloud array),
            labels (point-wise labels)
            calib (calibration parameters from the current scan sequence)
            img (image corresponding to current scan id)
        '''
        path_parts = PurePath(self.scan_path).parts
        scan_no = (path_parts[-1]).split('.')[0]
        seq_path = list(path_parts)[:-2]
        seq_path = join(*seq_path)

        if self.cfg['name'] == 'semantickitti':
            self.pc = read_bin_velodyne(self.scan_path, include_intensity='i' in self.features)
            label_path = join('labels', scan_no + '.label')
            if self.testds is False:
                self.labels = get_labels(join(seq_path, label_path), self.cfg)
            calib_path = join(seq_path, 'calib.txt')
            self.calib = read_calib_file(calib_path)

            self.img_path = join(seq_path, 'image_2', scan_no + '.png')
            self.img = cv2.cvtColor(cv2.imread(self.img_path), cv2.COLOR_BGR2RGB)

        elif self.cfg['name'] == 'vkitti':
            x = np.load(self.scan_path)
            self.pc = np.empty((x.shape[0], 6))
            self.pc[:, :3] = x[:, :3]
            self.pc[:, 3:] = x[:, 3:-1]
            self.labels = x[:, -1].astype(int)


    def get_modality(self):
        '''
        Run the SensorFusion pipeline on scan attributes, to obtain X,Y augmented with RGB information
        '''
        proj = SensorFusion(self.pc, self.calib, self.img, get_rgb='rgb' in self.features)
        fused_outputs = proj.render_lidar_rgb()
        cam_inds = fused_outputs[0]
        imgfov_pc_velo = self.pc[cam_inds, :]
        if self.testds is False:
            self.labels = self.labels[cam_inds]
        if 'rgb' in self.features:
            self.pc = np.hstack((imgfov_pc_velo, fused_outputs[1]))
        else:
            self.pc = imgfov_pc_velo


    def get_graph(self):
        '''
        Compute sparse adjacency matrix from the modified point cloud array, based on nearest neighbour search
        A is also preprocessed (normalization) for Graph Convolution operations
        '''
        self.A = compute_graph(self.pc, nn=10)
        self.A = normalize_A(self.A)


    def get_depth(self):

        depth = np.linalg.norm(self.pc[:, :3], axis=1)
        if 'xyz' in self.features:
            self.pc = np.hstack((self.pc, depth))
        else:
            self.pc = np.delete(self.pc, np.s_[:3], axis=1)
            self.pc = np.insert(self.pc, 0, depth, axis=1)


    def reduce_data(self):
        '''
        Extracts depth information from the point cloud by means of Euclidean Norm,
        also filters the sample wrt sensor-point distance, set to 35 metres
        :return: None
        '''
        if 'xyz' in self.features:
            valid_idx = np.where(np.linalg.norm(self.pc[:, :3], axis=1) < 35.0)
        else:
            raise Exception('No XYZ data found while processing point cloud.')
        self.pc = self.pc[valid_idx[0], :]
        if self.testds is False:
            self.labels = self.labels[valid_idx[0]]


def rot_augment_pc(x, bs, angle=90, axis='z'):
    '''
    Rotational augmentation of input point cloud with respect to a provided axis
    :param x: input point cloud
    :param bs: augmented batch size
    :param angle: rotational angle in degrees
    :param axis: rotational axis
    :return: stack of augmented point clouds
    '''

    aug_x = []
    rot_mask = np.linspace(0.0, 3.0, bs)
    rads = np.array(np.radians(rot_mask * angle))
    for i in range(len(rot_mask)):
        rot = get_rot_matrix(axis, rads[i])
        rotated = rot @ x[:, :3].T
        rot_pc = np.hstack((rotated.T, x[:, 3:]))
        aug_x.append(rot_pc)
    return np.vstack(aug_x)


def get_rot_matrix(axis, rads):
    '''
    Helper function for rotational augmentation -> provides a rotational transformation matrix depending on inputs
    :param axis: rotational axis
    :param rads: rotational angle in radians
    :return: rotational transformation matrix
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


def csr_to_tensor(a):
    '''
    Converts a Scipy-CSR matrix into a Sparse Tensor for NN forward pass
    :param a: input Scipy-CSR matrix
    :return: Sparse Tensor
    '''

    row, col, values = sp.find(a)
    out = SparseTensor(
        indices=np.array([row, col]).T, values=values, dense_shape=a.shape
    )
    a = sparse.reorder(out)
    return a


def tr_batch_gen(prep, file, bs=4):
    '''
    Prepares a batch of training samples from an input scan file
    :param prep: Preprocess() instance
    :param file: input scan file
    :param bs: batch size
    :return: tuple of inputs to training loop
    '''

    x, a, y = prep.assess_scan(file)
    x = rot_augment_pc(x, bs)
    a = csr_to_tensor(sp.block_diag([a] * bs))
    y = np.tile(y, bs)

    return x, a, y


def va_batch_gen(prep, file, test=False):
    '''
    Prepares a validation/test batch from an input scan file
    :param prep: Preprocess() instance
    :param file: input scan file
    :param test: Test dataset?
    :return:
    '''

    ins = prep.assess_scan(file, test)
    outs = [ins[0], csr_to_tensor(ins[1])]
    if test is False:
        outs.append(ins[2])
    return outs


if __name__ == '__main__':
    from time import time
    from visualization import PC_Vis

    BASE_DIR = safe_load(open('config/tr_config.yml', 'r'))['dataset']['base_dir']

    model_cfg = get_cfg_params()

    train_files, val_files, _ = get_split_files(cfg=model_cfg, shuffle=False)
    print(len(train_files), len(val_files))
    file_list = train_files[:3]

    prep = Preprocess(model_cfg)

    # start = time()
    x, a, y = prep.assess_scan(train_files[6])
    # print(time() - start)
    PC_Vis.draw_pc(x, True)
    PC_Vis.draw_pc_labels(x, y, model_cfg, True)
    # PC_Vis.draw_graph(x, a)
