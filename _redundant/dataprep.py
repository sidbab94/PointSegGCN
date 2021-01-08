import numpy as np
from yaml import safe_load
from os import listdir
from os.path import join
import os
os.chdir(os.getcwd())

import struct
from spektral.data import Dataset, Graph
from preproc_utils.graph_gen import compute_adjacency
from preproc_utils.voxelization import voxelize


class PCGraph(Dataset):
    def __init__(self, base_dir, seq_no=None, seq_list=None, stop_idx=None, test=False, vox=False, **kwargs):
        assert (seq_no is not None) or (seq_list is not None), 'Provide either a specific seq id or list of seqs'
        self.stop_idx = stop_idx
        self.seq = seq_no
        self.seq_list = seq_list
        self.base_dir = base_dir
        self.test = test
        self.vox = vox
        if self.vox:
            print('     Voxelization enabled.')
        super().__init__(**kwargs)

    def read(self):
        self.list_of_graphs = []
        if self.seq is not None:
            velo_dir = join(self.base_dir, self.seq, 'velodyne')
            label_dir = join(self.base_dir, self.seq, 'labels')
            velo_files = listdir(velo_dir)
            label_files = listdir(label_dir)
            for i in range(self.stop_idx):
                curr_velo_path = join(velo_dir, velo_files[i])
                curr_label_path = join(label_dir, label_files[i])
                x = read_bin_velodyne(curr_velo_path)[:, :3]
                y = get_labels(curr_label_path)
                print('     Graph Construction -- Seq {} | Scan {} -- complete.'.format(self.seq, i))
                self.get_adj(x, y)

        if self.seq_list is not None:
            for id in self.seq_list:
                velo_dir = join(self.base_dir, id, 'velodyne')
                velo_files = listdir(velo_dir)

                if self.stop_idx is not None:
                    iter_range = range(self.stop_idx)
                else:
                    iter_range = range(len(velo_files))

                if self.test:
                    for i in iter_range:
                        curr_velo_path = join(velo_dir, velo_files[i])
                        x = read_bin_velodyne(curr_velo_path)[:, :3]
                        print('     Graph Construction -- Seq {} | Scan {} -- complete.'.format(id, i))
                        self.get_adj(x, y)
                else:
                    label_dir = join(self.base_dir, id, 'labels')
                    label_files = listdir(label_dir)
                    for i in iter_range:
                        curr_velo_path = join(velo_dir, velo_files[i])
                        curr_label_path = join(label_dir, label_files[i])
                        x = read_bin_velodyne(curr_velo_path)[:, :3]
                        y = get_labels(curr_label_path)
                        print('     Graph Construction -- Seq {} | Scan {} -- complete.'.format(id, i))
                        self.get_adj(x, y)
        print('     Preprocessing..')
        # print(self.list_of_graphs)
        return self.list_of_graphs

    def get_adj(self, x, y):
        if self.vox:
            print('     Voxelizing..')
            x = np.insert(x, 3, np.arange(start=0, stop=x.shape[0]), axis=1)
            subsampling = voxelize(x)
            subsampling.get_voxels()
            vox_pc_map = subsampling.voxel_points
            for vox_id in range(len(vox_pc_map)):
                vox_pts = vox_pc_map[vox_id]
                vox_pts_ids = vox_pts[:, -1].astype('int')
                vox_labels = y[vox_pts_ids]
                print(np.unique(vox_labels))
                a = compute_adjacency(vox_pts)
                self.list_of_graphs.append(Graph(x=vox_pts, a=a, y=vox_labels))
        else:
            a = compute_adjacency(x)
            self.list_of_graphs.append(Graph(x=x, a=a, y=y))

np.random.seed(19)


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
               'learning_map': semkitti_cfg["learning_map"]}

    return tr_dict


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


def read_bin_velodyne(path):
    pc_list = []
    with open(path, 'rb') as f:
        content = f.read()
        pc_iter = struct.iter_unpack('ffff', content)
        for idx, point in enumerate(pc_iter):
            pc_list.append([point[0], point[1], point[2]])
    return np.asarray(pc_list, dtype=np.float32)


def load_label_kitti(label_path, remap_lut):
    label = np.fromfile(label_path, dtype=np.uint32)
    label = label.reshape((-1))
    sem_label = label & 0xFFFF  # semantic label in lower half
    inst_label = label >> 16  # instance id in upper half
    assert ((sem_label + (inst_label << 16) == label).all())
    sem_label = remap_lut[sem_label]
    return sem_label.astype(np.int32)


def get_labels(label_path, config):
    remap_dict_val = config["learning_map"]
    max_key = max(remap_dict_val.keys())
    remap_lut_val = np.zeros((max_key + 100), dtype=np.int32)
    remap_lut_val[list(remap_dict_val.keys())] = list(remap_dict_val.values())
    labels = load_label_kitti(label_path, remap_lut=remap_lut_val)
    return labels

def to_labels(label_array, store_path):
    upper_half = label_array >> 16  # get upper half for instances
    lower_half = label_array & 0xFFFF  # get lower half for semantics

    DATA = safe_load(open('config/semantic-kitti.yaml', 'r'))
    remap_dict_val = DATA["learning_map_inv"]
    max_key = max(remap_dict_val.keys())
    remap_lut = np.zeros((max_key + 100), dtype=np.int32)

    lower_half = remap_lut[lower_half]  # do the remapping of semantics
    pred = (upper_half << 16) + lower_half  # reconstruct full label
    pred = pred.astype(np.uint32)

    pred.tofile(store_path)

if __name__ == '__main__':
    BASE_DIR = 'D:/SemanticKITTI/dataset/sequences'
    label1_array = get_labels('samples/testpc.label')
    unique_labels_orig = np.unique(label1_array)

    label2_array = get_labels('samples/recon.label')
    unique_labels_recon = np.unique(label2_array)

    print(unique_labels_orig)
    print(unique_labels_recon)
