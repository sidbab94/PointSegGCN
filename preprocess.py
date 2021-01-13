import cv2
from spektral.layers import GCNConv
from spektral.data import Dataset, Graph, DisjointLoader

from preproc_utils.readers import *
from preproc_utils.readers import get_split_files, get_cfg_params
from preproc_utils.sensor_fusion import SensorFusion
from preproc_utils.graph_gen import sklearn_graph
from preproc_utils.voxelization import voxelize


class Preprocess:

    def __init__(self, cfg_file):

        self.cfg = cfg_file
        self.invalid_scan = False

    def get_scan_data(self):

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

        proj = SensorFusion(self.pc, self.labels, self.calib, self.img)
        self.pc_rgb, self.labels_rgb = proj.render_lidar_rgb()
        if not isinstance(self.pc_rgb, np.ndarray):
            self.invalid_scan = True

    def get_graph(self):

        if not self.invalid_scan:
            self.A = sklearn_graph(self.pc_rgb)
            self.A = GCNConv.preprocess(self.A)
            # self.A = sp_matrix_to_sp_tensor(self.A)
        else:
            self.A = None

    def assess_scan(self, scan_path):

        self.scan_path = scan_path

        self.get_scan_data()
        self.get_modality()
        self.get_graph()

        return self.pc_rgb, self.A, self.labels_rgb

    def assess_voxel(self, vox_id):

        self.pc_rgb = self.vox_pc_map[vox_id]
        vox_pts_ids = self.pc_rgb[:, -1].astype('int')
        self.labels_rgb = self.labels_rgb[vox_pts_ids]

        self.get_graph()

        return self.pc_rgb, self.A, self.labels_rgb

    def voxelize_scan(self):

        pc_w_ids = np.insert(self.pc_rgb, self.pc_rgb.shape[1],
                             np.arange(start=0, stop=self.pc_rgb.shape[0]), axis=1)
        grid = voxelize(pc_w_ids)
        grid.get_voxels()
        self.vox_pc_map = grid.voxel_points


class BatchData(Dataset):

    def __init__(self, file_list, prep_obj, verbose=False):

        self.file_list = file_list
        self.prep = prep_obj
        self.vv = verbose
        super().__init__()

    def read(self):

        output = []
        for file in self.file_list:

            if self.vv:
                print('     Processing : ', file)

            x, a, y = self.prep.assess_scan(file)

            if a is None:
                print('Numpy Memory Error, skipping current scan: ', file)
                continue

            output.append(Graph(x=x, a=a, y=y))

        return output


def prep_datasets(dataset_path, prep_obj, model_cfg, file_count=10, verbose=False):
    train_files, val_files, _ = get_split_files(dataset_path=dataset_path, cfg=model_cfg,
                                                count=file_count, shuffle=True)

    tr_dataset = BatchData(train_files, prep_obj, verbose)
    va_dataset = BatchData(val_files, prep_obj, verbose)

    return tr_dataset, va_dataset


def prep_loader(dataset, model_cfg):
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
