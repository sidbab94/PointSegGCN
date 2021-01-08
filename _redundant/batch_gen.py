import numpy as np
import os
import cv2

from preproc_utils.voxelization import voxelize
from _redundant.dataprep import read_bin_velodyne, get_labels
from projection import lidar_rgb_render, read_calib_file

from spektral.layers import GCNConv
from spektral.layers.ops import sp_matrix_to_sp_tensor
from spektral.utils.convolution import normalized_adjacency

class Graph:
    def __init__(self, x=None, a=None, y=None):
        self.x = x
        self.a = a
        self.y = y

    def numpy(self):
        return tuple(ret for ret in [self.x, self.a, self.y]
                     if ret is not None)

    def __repr__(self):
        return 'Graph(n_nodes={}, n_node_features={}, n_labels={})'\
               .format(self.x.shape[0], self.x.shape[1], self.y.shape[0])


def process_single(file_path, config):
    pc, labels = get_xy_data(file_path, config)

    A = compute_adjacency(pc)
    A = GCNConv.preprocess(A)
    A = sp_matrix_to_sp_tensor(A)

    return (pc, A, labels)

def process_single_rgb(file_path, config):
    pc, labels, calib, img = get_all_data(file_path, config)

    fov_pc, glob_pc_inds = lidar_rgb_render(pc, calib, img)
    fov_labels = labels[glob_pc_inds]

    A = compute_adjacency(fov_pc)
    A = GCNConv.preprocess(A)
    A = sp_matrix_to_sp_tensor(A)

    return (fov_pc, A, fov_labels)

def voxel(pc):
    pc = np.insert(pc, 3, np.arange(start=0, stop=pc.shape[0]), axis=1)
    grid = voxelize(pc)
    grid.get_voxels()
    vox_pc_map = grid.voxel_points
    return vox_pc_map

def process_voxel(vox_pc_map, vox_id, labels):
    vox_pts = vox_pc_map[vox_id]
    vox_pts_ids = vox_pts[:, -1].astype('int')
    vox_labels = labels[vox_pts_ids]
    assert vox_labels.shape[0] == vox_pts.shape[0]
    A = compute_adjacency(vox_pts)
    # A = GCNConv.preprocess(A)
    A = normalized_adjacency(A)
    A = sp_matrix_to_sp_tensor(A)
    return (vox_pts[:, :3], A, vox_labels)


def get_xy_data(pc_path, config):
    pc = read_bin_velodyne(pc_path)

    path_split = pc_path.split('\\')
    scan_no = (path_split[-1]).split('.')[0]
    seq_dir = os.path.join(path_split[0], path_split[1])

    label_path = os.path.join('labels', scan_no + '.label')
    labels = get_labels(os.path.join(seq_dir, label_path), config)

    return pc, labels

def get_all_data(pc_path, config):
    pc = read_bin_velodyne(pc_path)

    path_split = pc_path.split('\\')
    scan_no = (path_split[-1]).split('.')[0]
    seq_dir = os.path.join(path_split[0], path_split[1])

    label_path = os.path.join('labels', scan_no + '.label')
    labels = get_labels(os.path.join(seq_dir, label_path), config)

    calib_path = os.path.join(seq_dir, 'calib.txt')
    calib = read_calib_file(calib_path)

    img_path = os.path.join(seq_dir, 'image_2', scan_no + '.png')
    img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)

    return pc, labels, calib, img

def get_scan_data(base_dir, seq_path, id=0):
     velo_dir = os.path.join(base_dir, seq_path, 'velodyne')
     label_dir = os.path.join(base_dir, seq_path, 'labels')
     velo_path = os.path.join(velo_dir, os.listdir(velo_dir)[id])
     label_path = os.path.join(label_dir, os.listdir(label_dir)[id])
     pc = read_bin_velodyne(velo_path)
     labels = get_labels(label_path)
     return pc, labels


if __name__ == '__main__':
    base_dir = 'D:/SemanticKITTI/dataset/sequences'
    # pc, labels = get_scan_data(base_dir, '00')
    # batch = preprocess(pc, labels)
    from yaml import safe_load
    from _redundant.dataprep import get_split_files
    semkitti_cfg = safe_load(open('../config/semantic-kitti.yaml', 'r'))
    class_ignore = semkitti_cfg["learning_ignore"]

    train_files, val_files, test_files = get_split_files(dataset_path=base_dir, cfg=semkitti_cfg["split"], count=2)

    batch = process(train_files)

    print(len(batch))