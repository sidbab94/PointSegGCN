import numpy as np
import os
import yaml
from os.path import join

BASE_DIR = 'D:/SemanticKITTI/dataset/sequences'

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



