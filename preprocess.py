import cv2
from os.path import join
from os import listdir
from yaml import safe_load
from spektral.layers import GCNConv
from spektral.layers.ops import sp_matrix_to_sp_tensor

from preproc_utils.readers import *
from preproc_utils.sensor_fusion import SensorFusion
from preproc_utils.graph_gen import compute_adjacency, sklearn_graph
from preproc_utils.voxelization import voxelize


def get_split_files(dataset_path, cfg, count=-1, shuffle=False):
    train_seqs = cfg["tr_seq"]
    val_seqs = cfg["va_seq"]
    test_seqs = cfg["te_seq"]

    train_file_list = []
    test_file_list = []
    val_file_list = []
    seq_list = np.sort(listdir(dataset_path))

    for seq_id in seq_list:
        seq_path = join(dataset_path, seq_id)
        pc_path = join(seq_path, 'velodyne')
        if seq_id in train_seqs:
            train_file_list.append([join(pc_path, f) for f in np.sort(listdir(pc_path))[:count]])
        elif seq_id in val_seqs:
            val_file_list.append([join(pc_path, f) for f in np.sort(listdir(pc_path))[:round(count*1.0)]])
        elif seq_id in test_seqs:
            test_file_list.append([join(pc_path, f) for f in np.sort(listdir(pc_path)[:count])])

    train_file_list = np.concatenate(train_file_list, axis=0)
    val_file_list = np.concatenate(val_file_list, axis=0)
    test_file_list = np.concatenate(test_file_list, axis=0)

    if shuffle:
        np.random.shuffle(train_file_list)
        np.random.shuffle(val_file_list)
        np.random.shuffle(test_file_list)

    return train_file_list, val_file_list, test_file_list


def get_cfg_params(base_dir, dataset_cfg='config/semantic-kitti.yaml', train_cfg='config/tr_config.yml'):
    semkitti_cfg = safe_load(open(dataset_cfg, 'r'))
    tr_params = safe_load(open(train_cfg, 'r'))['training_params']

    split_params = semkitti_cfg['split']

    seq_list = np.sort(listdir(base_dir))

    tr_dict = {'ep': tr_params['epochs'],
               'num_classes': tr_params['num_classes'],
               'patience': tr_params['es_patience'],
               'l2_reg': tr_params['l2_reg'],
               'n_node_features': tr_params['n_node_features'],
               'learning_rate': tr_params['learning_rate'],
               'lr_decay': round(tr_params['lr_decay_steps'] * tr_params['epochs']),
               'loss_switch_ep': round(tr_params['lovasz_switch_ratio'] * tr_params['epochs']),
               'tr_seq': list(seq_list[split_params['train']]),
               'va_seq': list(seq_list[split_params['valid']]),
               'te_seq': list(seq_list[split_params['test']]),
               'class_ignore': semkitti_cfg["learning_ignore"],
               'learning_map': semkitti_cfg["learning_map"],
               'color_map': np.array(list(semkitti_cfg['color_map'].values()))/255}

    return tr_dict

class Preprocess:

    def __init__(self, cfg_file):

        self.cfg = cfg_file

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

        self.get_scan_data()

        proj = SensorFusion(self.pc, self.labels, self.calib, self.img)
        self.pc_rgb, self.labels_rgb = proj.render_lidar_rgb()

    def get_graph(self):

        self.get_modality()

        self.A = sklearn_graph(self.pc_rgb)
        self.A = GCNConv.preprocess(self.A)
        self.A = sp_matrix_to_sp_tensor(self.A)

    def assess_scan(self, scan_path):

        self.scan_path = scan_path

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




if __name__ == '__main__':

    from time import time

    BASE_DIR = 'D:/SemanticKITTI/dataset/sequences'

    model_cfg = get_cfg_params(BASE_DIR)

    train_files, val_files, _ = get_split_files(dataset_path=BASE_DIR, cfg=model_cfg, shuffle=True)

    prep = Preprocess(model_cfg)

    tic = time()
    tr_inputs = prep.assess_scan(train_files[0])
    print(time() - tic)

    print(tr_inputs)

