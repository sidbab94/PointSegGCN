import numpy as np
from numpy import genfromtxt
from time import time
import graph_gen as graph
import voxelization as vox
from visualization import show_voxel
from optparse import OptionParser
from mayavi import mlab
import yaml
import sys
from preprocess import Plot, load_label_kitti

parser = OptionParser()
parser.add_option('--input', dest='inputpc', default='samples/testpc.csv',
                  help='Provide input point cloud file in supported format (.pts, .csv, .npy) '
                       '** DEFAULT: samples/testpc.csv')
parser.add_option('--NN', dest='nearestN', default=2, type='int',
                  help='Provide number of nearest neighbours to process each point with. '
                       ' ** DEFAULT: 2')
parser.add_option('--sample_size', dest='ss', default=100.0, type='float',
                  help='Provide proportion of points (in %) to be randomly sampled from input. '
                       ' ** DEFAULT: 100.0')
parser.add_option('--nns_method', dest='searchmethod', default='kdtree',
                  help='Specify method to implement nearest neighbour search -- knn or kdtree? '
                       ' ** DEFAULT: kdtree')
parser.add_option('--vox_factor', dest='div_factor', default=15, type='int',
                  help='Specify voxelization factor (number of voxels along principal axis) '
                       ' ** DEFAULT: 10')
parser.add_option('--vox_thresh', dest='occ_thresh', default=5e-3, type='float',
                  help='Specify threshold as factor of original point cloud size, below which voxels would be ignored'
                       ' ** DEFAULT: 9e-3')
parser.add_option('--vp', dest='visualize_pc', action='store_true',
                  help='Enable visualization of down-sampled point cloud')
parser.add_option('--vg', dest='visualize_graphs', action='store_true',
                  help='Enable visualization of graph constructed from input point cloud')
parser.add_option('-o', dest='optimal', action='store_true',
                  help='Enable optimization of nearest neighbour search algorithm, by pre-sorting points')

options, _ = parser.parse_args()
np.random.seed(89)

label_path = '/media/baburaj/Seagate Backup Plus Drive/SemanticKITTI/dataset/sequences/00/labels/000000.label'
# label_path = 'D:/SemanticKITTI/dataset/sequences/00/labels/000000.label'
DATA = yaml.safe_load(open('semantic-kitti.yaml', 'r'))
remap_dict_val = DATA["learning_map"]
max_key = max(remap_dict_val.keys())
remap_lut_val = np.zeros((max_key + 100), dtype=np.int32)
remap_lut_val[list(remap_dict_val.keys())] = list(remap_dict_val.values())
labels = load_label_kitti(label_path, remap_lut=remap_lut_val)

def construct_vox_graph(vox_pc_map, vis_scale, vis_pc=False, vis_graph=False):
    elapsed = 0.0
    all_pts = np.empty((1, 3))
    all_lbls = np.empty((1,))
    for vox_id in range(len(vox_pc_map)):
        vox_start = time()

        vox_pts = vox_pc_map[vox_id]
        # vox_pts_ids = vox_pts[:, -1].astype('int')
        # vox_labels = labels[vox_pts_ids]
        # all_pts = np.concatenate((all_pts, vox_pts[:, :3]))
        # all_lbls = np.concatenate((all_lbls, vox_labels))
        # nns = graph.NearestNodeSearch(pointcloud=vox_pts, options=options)
        # nns.get_neighbours()
        # G = nns.graph
        # Plot.draw_pc_sem_ins(pc_xyz=vox_pts, pc_sem_ins=vox_labels)

        G = graph.adjacency(vox_pts, nn=options.nearestN)

        elapsed = (time() - vox_start) + elapsed
        if vox_id == len(vox_pc_map) - 1:
            print('     Graph construction done.')
        if vis_graph:
            show_voxel(vox_pts[:, :3], G, vis_scale)
    if vis_graph:
        print('     Visualizing graphs...')
        mlab.show()
    if vis_pc:
        print('     Visualizing downsampled point cloud...')
        Plot.draw_pc_sem_ins(pc_xyz=all_pts, pc_sem_ins=all_lbls)
    print('----------------------------------------------------------------------------------')
    print('Time elapsed for {} voxel(s): {} seconds'.format(len(vox_pc_map), round(elapsed, 5)))




def main():
    input_cloud_fmt = options.inputpc.split('.')[1]
    if input_cloud_fmt == 'pts':
        pc = genfromtxt(options.inputpc, delimiter='')
    elif input_cloud_fmt == 'csv':
        pc = genfromtxt(options.inputpc, delimiter=',')
    elif input_cloud_fmt == 'npy':
        pc = np.load(options.inputpc)
    else:
        raise Exception('Provide a supported point cloud format')

    pc = pc[:, :3]
    pc = np.insert(pc, 3, np.arange(start=0, stop=pc.shape[0]), axis=1)

    N = pc.shape[0]
    if options.ss != 100.0:
        sample_size = round(options.ss * 0.01 * N)
        pc = pc[np.random.choice(N, sample_size, replace=False), :]
    if options.optimal:
        pc = graph.sortpoints(pc)

    print('==================================================================================')
    print('----------------------------------------------------------------------------------')
    print('1.   Performing voxelization...')
    grid = vox.voxelize(pc, options.div_factor)
    grid.get_voxels()
    vis_scale = grid.v_scaleup
    pointcount = grid.pcount
    act_res = round((pointcount/pc.shape[0])*100, 2)
    print('     No. of points downsampled from {} to {}.'.format(pc.shape[0], pointcount))
    print('     Reduced to {}% of original density.'.format(act_res))
    print('     Voxelization done.')
    vox_pc_map = grid.voxel_points
    vis_pc = options.visualize_pc
    vis_graph = options.visualize_graphs
    print('----------------------------------------------------------------------------------')
    print('2.   Constructing graphs for voxels...')
    construct_vox_graph(vox_pc_map, vis_scale, vis_pc=vis_pc, vis_graph=vis_graph)
    print('----------------------------------------------------------------------------------')
    print('==================================================================================')

    sys.exit()

if __name__ == "__main__":
    main()
