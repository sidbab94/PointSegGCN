import numpy as np
import os
import yaml
from os.path import join
import struct

# BASE_DIR = 'D:/SemanticKITTI/dataset/sequences'

def get_split_files(dataset_path, config_file='semantic-kitti.yaml'):

    assert os.path.isfile(config_file)

    CFG = yaml.safe_load(open(config_file, 'r'))
    train_seqs = CFG["split"]["train"]
    val_seqs = CFG["split"]["valid"]
    test_seqs = CFG["split"]["test"]

    train_file_list = []
    test_file_list = []
    val_file_list = []
    seq_list = np.sort(os.listdir(dataset_path))

    for seq_id in seq_list:
        seq_path = join(dataset_path, seq_id)
        pc_path = join(seq_path, 'velodyne')

        if int(seq_id) in train_seqs:
            train_file_list.append([join(pc_path, f) for f in np.sort(os.listdir(pc_path))])
        elif int(seq_id) in val_seqs:
            val_file_list.append([join(pc_path, f) for f in np.sort(os.listdir(pc_path))])
        elif int(seq_id) in test_seqs:
            test_file_list.append([join(pc_path, f) for f in np.sort(os.listdir(pc_path))])

    train_file_list = np.concatenate(train_file_list, axis=0)
    val_file_list = np.concatenate(val_file_list, axis=0)
    test_file_list = np.concatenate(test_file_list, axis=0)

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

def get_labels(label_path):
    assert os.path.isfile('semantic-kitti.yaml')
    # label mapping with semantic-kitti config file
    DATA = yaml.safe_load(open('semantic-kitti.yaml', 'r'))
    remap_dict_val = DATA["learning_map"]
    max_key = max(remap_dict_val.keys())
    remap_lut_val = np.zeros((max_key + 100), dtype=np.int32)
    remap_lut_val[list(remap_dict_val.keys())] = list(remap_dict_val.values())
    labels = load_label_kitti(label_path, remap_lut=remap_lut_val)
    return labels

