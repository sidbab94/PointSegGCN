import cv2
from spektral.layers import GCNConv
from spektral.data import Dataset, Graph, DisjointLoader

from preproc_utils.readers import *
from preproc_utils.sensor_fusion import SensorFusion
from preproc_utils.graph_gen import sklearn_graph as compute_graph
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
        self.get_graph()

        return self.pc_rgb, self.A, self.labels_rgb

    def get_scan_data(self):
        '''
        Obtain scan attributes from the file path provided
        Scan attributes are:
            pc (point cloud array),
            labels (point-wise labels)
            calib (calibration parameters from the current scan sequence)
            img (image corresponding to current scan id)
        '''

        path_split = self.scan_path.split('\\')
        scan_no = (path_split[-1]).split('.')[0]
        seq_path = join(path_split[0], path_split[1])

        self.pc = read_bin_velodyne(self.scan_path)

        label_path = join('labels', scan_no + '.label')
        self.labels = get_labels(join(seq_path, label_path), self.cfg)

        calib_path = join(seq_path, 'calib.txt')
        self.calib = read_calib_file(calib_path)

        img_path = join(seq_path, 'image_2', scan_no + '.png')
        self.img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)

    def get_modality(self):
        '''
        Run the SensorFusion pipeline on scan attributes, to obtain X,Y augmented with RGB information
        '''
        proj = SensorFusion(self.pc, self.labels, self.calib, self.img)
        self.pc_rgb, self.labels_rgb = proj.render_lidar_rgb()
        if not isinstance(self.pc_rgb, np.ndarray):
            # boolean flag to recognize memory exception error, while computing projection matrices
            self.invalid_scan = True

    def get_graph(self):
        '''
        Compute sparse adjacency matrix from the modified point cloud array, based on nearest neighbour search
        A is also preprocessed (normalization) for Graph Convolution operations
        '''
        if not self.invalid_scan:
            self.A = compute_graph(self.pc_rgb)
            self.A = GCNConv.preprocess(self.A)
            # self.A = sp_matrix_to_sp_tensor(self.A)
        else:
            self.A = None

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

    def __init__(self, file_list, prep_obj, verbose=False, vox=False):

        self.file_list = file_list
        self.prep = prep_obj
        self.vv = verbose
        self.vox = vox
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

            else:
                x, a, y = self.prep.assess_scan(file)
                if a is None:
                    print('Numpy Memory Error, skipping current scan: ', file)
                    continue
                output.append(Graph(x=x, a=a, y=y))

        return output


def prep_dataset(file_list, prep_obj, verbose=False, vox=False):
    '''
    Run BatchData pipeline on list of scan files
    :param file_list: list of velodyne scan paths to process
    :param prep_obj: preprocessor object
    :param verbose: print out progress in terminal
    :return: Spektral dataset
    '''

    dataset = BatchData(file_list, prep_obj, verbose, vox)

    return dataset


def prep_datasets(dataset_path, prep_obj, model_cfg, file_count=10, verbose=False, vox=False):
    '''
    Get file list, then run BatchData pipeline to obtain training and validation datasets
    :param dataset_path: dataset base directory path
    :param prep_obj: preprocessor object
    :param model_cfg: model configuration dictionary
    :param file_count: stop index for each sequence
    :param verbose: print out progress in terminal
    :return: training and validation (Spektral) datasets
    '''

    train_files, val_files, _ = get_split_files(dataset_path=dataset_path, cfg=model_cfg,
                                                count=file_count, shuffle=True)

    tr_dataset = BatchData(train_files, prep_obj, verbose, vox)
    va_dataset = BatchData(val_files, prep_obj, verbose, vox)

    return tr_dataset, va_dataset


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

    return DisjointLoader(dataset, batch_size=batch_size, epochs=epochs)


if __name__ == '__main__':
    from time import time

    BASE_DIR = 'D:/SemanticKITTI/dataset/sequences'

    model_cfg = get_cfg_params(BASE_DIR)

    train_files, val_files, _ = get_split_files(dataset_path=BASE_DIR, cfg=model_cfg, shuffle=False)

    prep = Preprocess(model_cfg)

    va_dataset = BatchData(val_files, prep, verbose=True)

    print(va_dataset)
