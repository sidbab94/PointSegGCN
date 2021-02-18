from pathlib import PurePath
import cv2
from spektral.layers import GCNConv
from spektral.data import Dataset, Graph, DisjointLoader

from preproc_utils.readers import *
from preproc_utils.sensor_fusion import SensorFusion
from preproc_utils.graph_gen import compute_adjacency as compute_graph


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

    def assess_scan(self, scan_path, aug_flag=False):
        '''
        Scan path is processed from start to end, all necessary training inputs obtained
        :param scan_path: file path containing velodyne point cloud (.bin)
        :return: pc_rgb, A, labels_rgb
        '''

        self.scan_path = scan_path

        self.get_scan_data()
        if 'rgb' in self.features:
            self.get_modality()
        if 'd' in self.features:
            self.get_depth()
            self.reduce_data()
        self.prune_points()
        if aug_flag is False:
            self.get_graph()
            return self.pc, self.A, self.labels
        else:
            return self.pc, self.labels

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

        self.pc = read_bin_velodyne(self.scan_path, include_intensity='i' in self.features)
        label_path = join('labels', scan_no + '.label')
        self.labels = get_labels(join(seq_path, label_path), self.cfg)

        if 'rgb' in self.features:
            calib_path = join(seq_path, 'calib.txt')
            self.calib = read_calib_file(calib_path)

            self.img_path = join(seq_path, 'image_2', scan_no + '.png')
            self.img = cv2.cvtColor(cv2.imread(self.img_path), cv2.COLOR_BGR2RGB)

    def get_modality(self):
        '''
        Run the SensorFusion pipeline on scan attributes, to obtain X,Y augmented with RGB information
        '''
        proj = SensorFusion(self.pc, self.labels, self.calib, self.img)
        cam_inds, rgb_array = proj.render_lidar_rgb()
        imgfov_pc_velo = self.pc[cam_inds, :]
        self.pc = np.hstack((imgfov_pc_velo, rgb_array))
        self.labels = self.labels[cam_inds]

        if not isinstance(cam_inds, np.ndarray):
            # boolean flag to recognize memory exception error, while computing projection matrices
            self.invalid_scan = True

    def get_graph(self):
        '''
        Compute sparse adjacency matrix from the modified point cloud array, based on nearest neighbour search
        A is also preprocessed (normalization) for Graph Convolution operations
        '''
        if not self.invalid_scan:
            self.A = compute_graph(self.pc)
            self.A = GCNConv.preprocess(self.A)
            # self.A = sp_matrix_to_sp_tensor(self.A)
        else:
            self.A = None

    def get_depth(self):

        depth = np.linalg.norm(self.pc[:, :3], axis=1)
        if 'xyz' in self.features:
            self.pc = np.hstack((self.pc, depth))
        else:
            self.pc = np.delete(self.pc, np.s_[:3], axis=1)
            self.pc = np.insert(self.pc, 0, depth, axis=1)

        # assert self.pc.shape[1] == self.feat_len

    def reduce_data(self):
        '''
        Extracts depth information from the point cloud by means of Euclidean Norm,
        also filters the sample wrt sensor-point distance, set to 35 metres
        :return: None
        '''
        if 'xyz' in self.features:
            valid_idx = np.where(self.pc[:, -1] < 35.0)
        else:
            valid_idx = np.where(self.pc[:, 0] < 35.0)
        self.pc = self.pc[valid_idx[0], :]
        self.labels = self.labels[valid_idx[0]]

    def prune_points(self):

        N = self.pc.shape[0]
        if N % 4 != 0:
            prune_idx = np.random.choice(N, N % 4, replace=False)
            self.pc = np.delete(self.pc, prune_idx, 0)
            self.labels = np.delete(self.labels, prune_idx, 0)

    def rgb_to_scalar(self):
        rgb = self.pc[:, -3:]
        scalars = np.zeros((rgb.shape[0],), dtype='float32')
        for (kp_idx, kp_c) in enumerate(rgb):
            r, g, b = kp_c[0], kp_c[1], kp_c[2]
            scalars[kp_idx] = 256 ** 2 * r + 256 * g + b
        self.pc = np.column_stack((self.pc[:, 0], scalars))



class BatchData(Dataset):
    """
    Generates a Spektral dataset from file-list provided, using the Preprocess() pipeline
    More info: https://graphneural.network/creating-dataset/

    Arguments:
        file_list: list of velodyne scan paths to process
        prep_obj: preprocessor object
    """

    def __init__(self, file_list, prep_obj, augment=False, verbose=False):

        self.file_list = file_list
        self.prep = prep_obj
        self.augment = augment
        self.vv = verbose
        super().__init__()

    def read(self):

        output = []
        for file in self.file_list:

            if self.vv:
                print('     Processing : ', file)

            if self.augment:
                x, y = self.prep.assess_scan(file, aug_flag=True)
                aug_obj = AugmentPC((x, y), rot=True, downsample=False)
                for i in aug_obj.augmented:
                    X, Y = i
                    A = compute_graph(X)
                    A = GCNConv.preprocess(A)
                    output.append(Graph(x=X, a=A, y=Y))
            else:
                x, a, y = self.prep.assess_scan(file)
                if a is None:
                    print('Numpy Memory Error, skipping current scan: ', file)
                    continue
                output.append(Graph(x=x, a=a, y=y))

        return output


class AugmentPC:
    """
    Augmentation of point cloud by means of rotation about a provided axis,
    as well as optional down-sampling at different ratios.

    Arguments:
        inputs: processed scan inputs --> x,y ('a' computed separately for each augmented sample)
        rot: boolean flag to enable rotation
        angle: rotation angle in degrees
        axis: axis to rotate about
        downsample: adds down-sampled versions of point cloud to augmented list
        ds_ratio: number of required output samples from down-sampling process

    """
    def __init__(self, inputs, rot=False, angle=45,
                 axis='z', downsample=False, ds_ratio=5):

        self.pc, self.labels = inputs
        self.augmented = []
        if rot:
            self.rotate_pc(axis, angle)
        if downsample:
            self.downsample(ds_ratio)

    def rotate_pc(self, axis, angle):
        rot_mask = np.arange(-1.0, 1.4, 0.4)
        for i in range(len(rot_mask)):
            x = self.pc

            rads = np.radians(rot_mask[i] * angle)
            rot = self.get_rot_matrix(axis, rads)

            rotated = rot @ x[:, :3].T
            new_pc = np.hstack((rotated.T, x[:, 3:]))
            self.augmented.append([new_pc, self.labels])

    def get_rot_matrix(self, axis, rads):

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

    def downsample(self, ds_ratio):

        # initialize random generator w/ seed
        rng = np.random.default_rng(seed=123)

        for j in range(len(self.augmented)):

            pc, _, _, = self.augmented[j]
            N = pc.shape[0]

            # sample size (floored)
            valid_ss = N // ds_ratio
            # final sample size with surplus
            rem_ss = N - (valid_ss * ds_ratio)
            # point indices of original point cloud
            glo_ind = np.arange(N)

            for i in range(ds_ratio):

                if i == ds_ratio - 1:
                    valid_ss += rem_ss

                curr_ind = rng.choice(glo_ind, valid_ss, replace=True)
                # prune point indices wrt indices already sampled --> mutual exclusion of samples
                del_ind = np.nonzero(np.isin(glo_ind, curr_ind))
                glo_ind = np.delete(glo_ind, del_ind)

                self.augmented.append([pc[curr_ind, :], self.a[curr_ind][:, curr_ind], self.labels[curr_ind]])


def prep_dataset(file_list, prep_obj, augment=False, verbose=False):
    '''
    Run BatchData pipeline on list of scan files
    :param file_list: list of velodyne scan paths to process
    :param prep_obj: preprocessor object
    :param verbose: print out progress in terminal
    :return: Spektral dataset
    '''

    dataset = BatchData(file_list, prep_obj, augment, verbose)

    return dataset


def prep_loader(dataset, model_cfg):
    '''
    Configure a Spektral dataset Disjoint Loader for batch-wise data feed into train/eval loop.
    A Disjoint loader produces a batch of graphs from the dataset via their disjoint union.
    More info: https://graphneural.network/loaders/#disjointloader

    :param dataset: Spektral dataset to process
    :param model_cfg: model configuration dictionary
    :return: Spektral Disjoint
    '''

    batch_size = model_cfg['batch_size']
    epochs = model_cfg['ep']

    return DisjointLoader(dataset, batch_size=batch_size, epochs=epochs, shuffle=True)




if __name__ == '__main__':
    from time import time
    from visualization import PC_Vis

    BASE_DIR = 'D:/SemanticKITTI/dataset/sequences'

    model_cfg = get_cfg_params(BASE_DIR)

    train_files, val_files, _ = get_split_files(dataset_path=BASE_DIR, cfg=model_cfg, shuffle=False)
    file_list = train_files[:3]

    prep = Preprocess(model_cfg)

    x, y = prep.assess_scan(train_files[0], aug_flag=True)
    aug = AugmentPC(inputs=(x, y), rot=True)
    for i in aug.augmented:
        X, Y = i
        print(X[56:60, :])
        A = compute_graph(X)
        A = GCNConv.preprocess(A)
        PC_Vis.draw_graph(X, A)
        # PC_Vis.draw_pc(X, vis_test=True)

    # dataset = prep_dataset(file_list, prep, True)
    # loader = prep_loader(dataset, model_cfg)
    # i = 0
    # for batch in loader:
    #     inputs, target = batch
    #     X, A, _ = inputs
    #     print(X[56:60, :])
    #     Y = target
    #     i += 1
    #     # PC_Vis.draw_pc(X, vis_test=True)
    #     if i == 5:
    #         break

