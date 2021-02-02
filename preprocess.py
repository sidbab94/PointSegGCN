from pathlib import PurePath
import cv2
from spektral.layers import GCNConv
from spektral.data import Dataset, Graph, DisjointLoader

from preproc_utils.readers import *
from preproc_utils.sensor_fusion import SensorFusion
from preproc_utils.graph_gen import compute_adjacency as compute_graph
from preproc_utils.voxelization import voxelize


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

    def assess_scan(self, scan_path):
        '''
        Scan path is processed from start to end, all necessary training inputs obtained
        :param scan_path: file path containing velodyne point cloud (.bin)
        :return: pc_rgb, A, labels_rgb
        '''

        self.scan_path = scan_path

        self.get_scan_data()
        self.get_modality()
        self.get_depth()
        self.reduce_data()
        self.prune_points()
        self.get_graph()

        return self.pc_xyzirgbd, self.A, self.labels_xyzirgbd

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

        self.pc = read_bin_velodyne(self.scan_path)

        label_path = join('labels', scan_no + '.label')
        self.labels = get_labels(join(seq_path, label_path), self.cfg)

        calib_path = join(seq_path, 'calib.txt')
        self.calib = read_calib_file(calib_path)

        self.img_path = join(seq_path, 'image_2', scan_no + '.png')
        self.img = cv2.cvtColor(cv2.imread(self.img_path), cv2.COLOR_BGR2RGB)

    def get_modality(self):
        '''
        Run the SensorFusion pipeline on scan attributes, to obtain X,Y augmented with RGB information
        '''
        proj = SensorFusion(self.pc, self.labels, self.calib, self.img)
        self.pc_xyzirgb, self.labels_xyzirgbd = proj.render_lidar_rgb()
        if not isinstance(self.pc_xyzirgb, np.ndarray):
            # boolean flag to recognize memory exception error, while computing projection matrices
            self.invalid_scan = True

    def get_graph(self):
        '''
        Compute sparse adjacency matrix from the modified point cloud array, based on nearest neighbour search
        A is also preprocessed (normalization) for Graph Convolution operations
        '''
        if not self.invalid_scan:
            self.A = compute_graph(self.pc_xyzirgbd)
            self.A = GCNConv.preprocess(self.A)
            # self.A = sp_matrix_to_sp_tensor(self.A)
        else:
            self.A = None

    def get_depth(self):

        depth = np.linalg.norm(self.pc_xyzirgb[:, :3], axis=1)
        self.pc_xyzirgbd = np.zeros((self.pc_xyzirgb.shape[0], 8))
        self.pc_xyzirgbd[:, :7] = self.pc_xyzirgb
        self.pc_xyzirgbd[:, 7] = depth

    def reduce_data(self):
        '''
        Extracts depth information from the point cloud by means of Euclidean Norm,
        also filters the sample wrt sensor-point distance, set to 35 metres
        :return: None
        '''

        valid_idx = np.where(self.pc_xyzirgbd[:, 7] < 35.0)
        self.pc_xyzirgbd = self.pc_xyzirgbd[valid_idx[0], :]
        self.labels_xyzirgbd = self.labels_xyzirgbd[valid_idx[0]]

    def prune_points(self):

        N = self.pc_xyzirgbd.shape[0]
        if N % 4 != 0:
            prune_idx = np.random.choice(N, N % 4, replace=False)
            self.pc_xyzirgbd = np.delete(self.pc_xyzirgbd, prune_idx, 0)
            self.labels_xyzirgbd = np.delete(self.labels_xyzirgbd, prune_idx, 0)


    def assess_voxel(self, vox_id):
        '''
        Run pipeline through a single voxel
        :param vox_id: index corresponding to voxel (from list) to be processed
        :return: pc_rgb, A, labels_rgb
        '''

        vox_pc = self.vox_pc_map[vox_id]
        vox_pts_ids = vox_pc[:, -1].astype('int')
        vox_y = self.labels_rgb[vox_pts_ids]
        vox_a = compute_graph(vox_pc)
        vox_a = GCNConv.preprocess(vox_a)

        return vox_pc, vox_a, vox_y

    def voxelize_scan(self):
        '''
        'Voxelizes' the point cloud, obtains list of voxels and corresponding points
        '''

        pc_w_ids = np.insert(self.pc_rgb, self.pc_rgb.shape[1],
                             np.arange(start=0, stop=self.pc_rgb.shape[0]), axis=1)
        self.vox_pc_map = voxelize(pc_w_ids).voxel_points


class BatchData(Dataset):
    """
    Generates a Spektral dataset from file-list provided, using the Preprocess() pipeline
    More info: https://graphneural.network/creating-dataset/

    Arguments:
        file_list: list of velodyne scan paths to process
        prep_obj: preprocessor object
    """

    def __init__(self, file_list, prep_obj, verbose=False, vox=False, sampling=False, augment=False):

        self.file_list = file_list
        self.prep = prep_obj
        self.vv = verbose
        self.vox = vox
        self.ss = sampling
        self.ss_ratio = 20
        self.augment = augment
        super().__init__()

    def read(self):

        output = []
        for file in self.file_list:

            if self.vv:
                print('     Processing : ', file)

            if self.vox:
                self.prep.assess_scan(file)
                self.prep.voxelize_scan()
                for id in range(len(self.prep.vox_pc_map)):
                    x, a, y = self.prep.assess_voxel(id)
                    if a is None:
                        print('Numpy Memory Error, skipping current scan: ', file)
                        continue
                    output.append(Graph(x=x, a=a, y=y))

            elif self.ss:
                print('     Down-sampling : ', file)
                X, A, Y = self.prep.assess_scan(file)
                slice_gen = gen_sample_indices(X, self.ss_ratio)
                for i in range(self.ss_ratio):
                    indices = next(slice_gen)
                    x, a, y = slice_scan_attr((X, A, Y), indices)
                    output.append(Graph(x=x, a=a, y=y))

            else:
                x, a, y = self.prep.assess_scan(file)
                if a is None:
                    print('Numpy Memory Error, skipping current scan: ', file)
                    continue
                if self.augment:
                    aug_obj = AugmentPC((x, a, y), rot=True, downsample=False)
                    for i in aug_obj.augmented:
                        X, A, Y = i
                        output.append(Graph(x=X, a=A, y=Y))
                else:
                    output.append(Graph(x=x, a=a, y=y))

        return output


class AugmentPC:

    def __init__(self, inputs, rot=False, rot_count=3, angle=60,
                 axis='z', downsample=False, ds_ratio=5):

        self.pc, self.a, self.labels = inputs
        self.augmented = [[self.pc, self.a, self.labels]]
        if rot:
            self.rotate_pc(axis, angle, rot_count)
        if downsample:
            self.downsample(ds_ratio)

    def rotate_pc(self, axis, angle, rot_count):
        rot_mask = np.arange(-1, 1, 0.25)
        for i in range(rot_count):
            x = self.pc
            rads = np.radians(rot_mask[i] * angle)
            rot = self.get_rot_matrix(axis, rads)
            rotated = rot @ x[:, :3].T
            x[:, :3] = rotated.T
            self.augmented.append([x, self.a, self.labels])

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


def prep_dataset(file_list, prep_obj, verbose=False, vox=False, downsampling=False, augment=False):
    '''
    Run BatchData pipeline on list of scan files
    :param file_list: list of velodyne scan paths to process
    :param prep_obj: preprocessor object
    :param verbose: print out progress in terminal
    :return: Spektral dataset
    '''

    dataset = BatchData(file_list, prep_obj, verbose, vox, downsampling, augment)

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

    return DisjointLoader(dataset, batch_size=batch_size, epochs=epochs, shuffle=False)


def slice_scan_attr(inputs, sampled_ind):
    '''
    Slices scan attributes (X, A, Y) into smaller sub-samples,
    with slicing indices generated by gen_sample_indices()

    :param inputs: original 'assessed' scan inputs
    :param sampled_ind: generated slicing indices
    :return: sub-sample of original input tuple
    '''

    x, a, y = inputs

    x = x[sampled_ind, :]
    a = a[sampled_ind][:, sampled_ind]
    y = y[sampled_ind]

    return x, a, y


def gen_sample_indices(x, ss_ratio=10):
    '''
    Generates sub-sampling indices based on a ratio of total PC size,
    uses NumPy's default bit generator PCG64 for random sampling

    :param x: original point cloud
    :param ss_ratio: sub-sampling ratio
    :return: sub-sampling indices
    '''

    # initialize random generator w/ seed
    rng = np.random.default_rng(seed=123)

    N = x.shape[0]

    # sample size (floored)
    valid_ss = N // ss_ratio
    # final sample size with surplus
    rem_ss = N - (valid_ss * ss_ratio)
    # point indices of original point cloud
    glo_ind = np.arange(N)

    stop = 1
    while stop <= ss_ratio:

        if stop == ss_ratio:
            valid_ss += rem_ss

        curr_ind = rng.choice(glo_ind, valid_ss, replace=False)
        # prune point indices wrt indices already sampled --> mutual exclusion of samples
        del_ind = np.nonzero(np.isin(glo_ind, curr_ind))
        glo_ind = np.delete(glo_ind, del_ind)
        stop += 1

        yield curr_ind




if __name__ == '__main__':
    from time import time
    from visualization import PC_Vis

    BASE_DIR = 'D:/SemanticKITTI/dataset/sequences'

    model_cfg = get_cfg_params(BASE_DIR)

    train_files, val_files, _ = get_split_files(dataset_path=BASE_DIR, cfg=model_cfg, shuffle=False)

    prep = Preprocess(model_cfg)

    dataset = prep_dataset(train_files[:10], prep, augment=True)
    print(dataset)

