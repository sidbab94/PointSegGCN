import os
import sys
import numpy as np
from time import time
import networkx as nx
from optparse import OptionParser
from mayavi import mlab

from utils import graph_gen as graph, voxelization as vox
from utils.dataprep import get_labels, read_bin_velodyne
from utils.visualization import show_voxel_wlabels, ShowPC

########
BASE_DIR = 'D:/SemanticKITTI/dataset/sequences'
########

parser = OptionParser()
parser.add_option('--input', dest='scanpath',
                  help='Provide input LiDAR scan path in the following format s{sequence#}_{scanid}, e.g. s04_000000 ')
parser.add_option('--NN', dest='nearestN', default=5, type='int',
                  help='Provide number of nearest neighbours to process each point with. '
                       ' ** DEFAULT: 2')
parser.add_option('--sample_size', dest='ss', default=100.0, type='float',
                  help='Provide proportion of points (in %) to be randomly sampled from input. '
                       ' ** DEFAULT: 100.0')
parser.add_option('--vox', dest='voxel', action='store_true',
                  help='Voxelize point cloud before graph construction for better speed, possibly lower resolution')
parser.add_option('--vp', dest='visualize_cloud', action='store_true',
                  help='Enable visualization of down-sampled point cloud')
parser.add_option('--vg', dest='visualize_graphs', action='store_true',
                  help='Enable visualization of graph constructed from input point cloud')


options, _ = parser.parse_args()
np.random.seed(89)

def construct_whole_graph(pc, labels, vis_scale=1, vis_pc=False, vis_graph=False):
    start = time()
    A = graph.adjacency(pc, options.nearestN, labels)
    assert A.shape == (pc.shape[0], pc.shape[0])
    elapsed = time() - start
    print('     Graph construction done.')
    print('     Time elapsed : {} seconds'.format(round(elapsed, 5)))
    if vis_graph:
        G = nx.from_scipy_sparse_matrix(A)
        print('     Visualizing graph...')
        show_voxel_wlabels(pc[:, :3], labels, G, vis_scale)
        mlab.show()
    if vis_pc:
        print('     Visualizing point cloud...')
        ShowPC.draw_pc_sem_ins(pc_xyz=pc[:, :3], pc_sem_ins=labels)
    return A


def construct_vox_graph(vox_pc_map, labels, vis_scale, vis_pc=False, vis_graph=False, vis_voxels=False):
    elapsed = 0.0
    all_pts = np.empty((1, 3))
    all_lbls = np.empty((1,))
    for vox_id in range(len(vox_pc_map)):
        vox_start = time()

        vox_pts = vox_pc_map[vox_id]
        vox_pts_ids = vox_pts[:, -1].astype('int')
        vox_labels = labels[vox_pts_ids]

        A = graph.adjacency(vox_pts, options.nearestN, vox_labels)

        elapsed = (time() - vox_start) + elapsed
        if vox_id == len(vox_pc_map) - 1:
            print('     Graph construction done.')
        if vis_graph:
            G = nx.from_scipy_sparse_matrix(A)
            show_voxel_wlabels(vox_pts[:, :3], vox_labels, G, vis_scale)
        if vis_pc:
            all_pts = np.concatenate((all_pts, vox_pts[:, :3]))
            all_lbls = np.concatenate((all_lbls, vox_labels))

    if vis_graph:
        print('     Visualizing graphs...')
        mlab.show()
    if vis_pc:
        print('     Visualizing downsampled point cloud...')
        ShowPC.draw_pc_sem_ins(pc_xyz=all_pts, pc_sem_ins=all_lbls)
    print('----------------------------------------------------------------------------------')
    print('Time elapsed for {} voxel(s): {} seconds'.format(len(vox_pc_map), round(elapsed, 5)))


def main():

    # parse input scan path for attributes (binary, label files)
    seq_no = options.scanpath[1:3]
    velo_file = options.scanpath.split('_')[1] + '.bin'
    label_file = options.scanpath.split('_')[1] + '.label'
    velo_path = os.path.join(BASE_DIR + os.altsep, seq_no, 'velodyne', velo_file)
    label_path = os.path.join(BASE_DIR + os.altsep, seq_no, 'labels', label_file)

    # read point cloud and labels
    pc = read_bin_velodyne(velo_path)
    labels = get_labels(label_path)

    # extract XYZ vectors fron original cloud
    pc = pc[:, :3]
    # add 'index' dimension as last column
    pc = np.insert(pc, 3, np.arange(start=0, stop=pc.shape[0]), axis=1)

    N = pc.shape[0]
    # if point cloud sampling rate specified as argument, proceed with down-sampling point cloud
    if options.ss != 100.0:
        sample_size = round(options.ss * 0.01 * N)
        pc = pc[np.random.choice(N, sample_size, replace=False), :]
    vis_pc = options.visualize_cloud
    vis_graph = options.visualize_graphs

    print('==================================================================================')

    if options.voxel:
        print('----------------------------------------------------------------------------------')
        print('1.   Performing voxelization...')
        grid = vox.voxelize(pc)
        grid.get_voxels()
        vis_scale = grid.v_scaleup
        pointcount = grid.pcount
        act_res = round((pointcount/pc.shape[0])*100, 2)
        print('     No. of points downsampled from {} to {}.'.format(pc.shape[0], pointcount))
        print('     Reduced to {}% of original density.'.format(act_res))
        print('     Voxelization done.')
        vox_pc_map = grid.voxel_points
        print('----------------------------------------------------------------------------------')
        print('2.   Constructing graphs for voxels...')
        construct_vox_graph(vox_pc_map, labels=labels, vis_scale=vis_scale,
                            vis_pc=vis_pc, vis_graph=vis_graph)
        print('----------------------------------------------------------------------------------')
    else:
        print('1.   Constructing graph for {} points...'.format(pc.shape[0]))
        A = construct_whole_graph(pc, labels=labels, vis_scale=1, vis_pc=vis_pc, vis_graph=vis_graph)

    print('==================================================================================')

    sys.exit()

if __name__ == "__main__":
    main()
