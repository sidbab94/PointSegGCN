import numpy as np
import os
from os.path import join
import yaml
import struct
import open3d as o3d
from numpy import random
import colorsys

def load_label_kitti(label_path, remap_lut):
    label = np.fromfile(label_path, dtype=np.uint32)
    label = label.reshape((-1))
    sem_label = label & 0xFFFF  # semantic label in lower half
    inst_label = label >> 16  # instance id in upper half
    assert ((sem_label + (inst_label << 16) == label).all())
    sem_label = remap_lut[sem_label]
    return sem_label.astype(np.int32)

def get_file_list(dataset_path, test_scan_num='14'):
    seq_list = np.sort(os.listdir(dataset_path))

    train_file_list = []
    test_file_list = []
    val_file_list = []
    for seq_id in seq_list:
        seq_path = join(dataset_path, seq_id)
        pc_path = join(seq_path, 'velodyne')
        if seq_id == '08':
            val_file_list.append([join(pc_path, f) for f in np.sort(os.listdir(pc_path))])
            if seq_id == test_scan_num:
                test_file_list.append([join(pc_path, f) for f in np.sort(os.listdir(pc_path))])
        elif int(seq_id) >= 11 and seq_id == test_scan_num:
            test_file_list.append([join(pc_path, f) for f in np.sort(os.listdir(pc_path))])
        elif seq_id in ['00', '01', '02', '03', '04', '05', '06', '07', '09', '10']:
            train_file_list.append([join(pc_path, f) for f in np.sort(os.listdir(pc_path))])

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

class Plot:
    @staticmethod
    def random_colors(N, bright=True, seed=0):
        brightness = 1.0 if bright else 0.7
        hsv = [(0.15 + i / float(N), 1, brightness) for i in range(N)]
        colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
        random.seed(seed)
        random.shuffle(colors)
        return colors

    @staticmethod
    def draw_pc(pc_xyzrgb):
        pc = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(pc_xyzrgb[:, 0:3])
        if pc_xyzrgb.shape[1] == 3:
            o3d.visualization.draw_geometries([pc])
            return 0
        if np.max(pc_xyzrgb[:, 3:6]) > 20:  ## 0-255
            pc.colors = o3d.utility.Vector3dVector(pc_xyzrgb[:, 3:6] / 255.)
        else:
            pc.colors = o3d.utility.Vector3dVector(pc_xyzrgb[:, 3:6])
        o3d.visualization.draw_geometries([pc])
        return 0

    @staticmethod
    def draw_pc_sem_ins(pc_xyz, pc_sem_ins, plot_colors=None):
        """
        pc_xyz: 3D coordinates of point clouds
        pc_sem_ins: semantic or instance labels
        plot_colors: custom color list
        """
        if plot_colors is not None:
            ins_colors = plot_colors
        else:
            ins_colors = Plot.random_colors(len(np.unique(pc_sem_ins)) + 1, seed=2)
        ##############################
        sem_ins_labels = np.unique(pc_sem_ins)
        sem_ins_bbox = []
        Y_colors = np.zeros((pc_sem_ins.shape[0], 3))
        for id, semins in enumerate(sem_ins_labels):
            valid_ind = np.argwhere(pc_sem_ins == semins)[:, 0]
            if semins <= -1:
                tp = [0, 0, 0]
            else:
                if plot_colors is not None:
                    tp = ins_colors[semins]
                else:
                    tp = ins_colors[id]

            Y_colors[valid_ind] = tp

            ### bbox
            valid_xyz = pc_xyz[valid_ind]

            xmin = np.min(valid_xyz[:, 0])
            xmax = np.max(valid_xyz[:, 0])
            ymin = np.min(valid_xyz[:, 1])
            ymax = np.max(valid_xyz[:, 1])
            zmin = np.min(valid_xyz[:, 2])
            zmax = np.max(valid_xyz[:, 2])
            sem_ins_bbox.append(
                [[xmin, ymin, zmin], [xmax, ymax, zmax], [min(tp[0], 1.), min(tp[1], 1.), min(tp[2], 1.)]])

        Y_semins = np.concatenate([pc_xyz[:, 0:3], Y_colors], axis=-1)
        Plot.draw_pc(Y_semins)
        return Y_semins


if __name__ == "__main__":
    DATA = yaml.safe_load(open('semantic-kitti.yaml', 'r'))
    remap_dict_val = DATA["learning_map"]
    # print(remap_dict_val)
    max_key = max(remap_dict_val.keys())
    # print(max_key)
    remap_lut_val = np.zeros((max_key + 100), dtype=np.int32)
    # print(remap_lut_val)
    remap_lut_val[list(remap_dict_val.keys())] = list(remap_dict_val.values())
    # print(remap_lut_val)
    cmap = list(DATA['color_map'].values())
    # base_path = 'D:/SemanticKITTI/dataset/sequences/'
    # label_path = 'D:/SemanticKITTI/dataset/sequences/00/labels/000000.label'
    # pc_path = 'D:/SemanticKITTI/dataset/sequences/00/velodyne/000000.bin'
    base_path = '/media/baburaj/Seagate Backup Plus Drive/SemanticKITTI/dataset/sequences/'
    label_path = '/media/baburaj/Seagate Backup Plus Drive/SemanticKITTI/dataset/sequences/00/labels/000000.label'
    pc_path = '/media/baburaj/Seagate Backup Plus Drive/SemanticKITTI/dataset/sequences/00/velodyne/000000.bin'

    cloud = read_bin_velodyne(pc_path)
    labels = load_label_kitti(label_path, remap_lut=remap_lut_val)
    cmap_labels = list(cmap[i] for i in np.unique(labels))


    # print(labels.shape)
    # print(cloud.shape)

    train, val, test = get_file_list(base_path)
    print(len(train), len(val), len(test))

    # Plot.draw_pc_sem_ins(cloud, labels, plot_colors=cmap_labels)