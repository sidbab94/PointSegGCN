import numpy as np
from numpy import genfromtxt
from time import time
import graph_gen as graph
import voxelization as vox
from visualization import show_voxel
from optparse import OptionParser
from mayavi import mlab
import sys

parser = OptionParser()
parser.add_option('--input', dest='inputpc', default='samples/vkitti3d_01.npy',
                  help='Provide input point cloud file in supported format (.pts, .csv, .npy) '
                       '** DEFAULT: samples/vkitti3d_01.npy')
parser.add_option('--NN', dest='nearestN', default=2, type='int',
                  help='Provide number of nearest neighbours to process each point with. '
                       ' ** DEFAULT: 2')
parser.add_option('--sample_size', dest='ss', default=100.0, type='float',
                  help='Provide proportion of points (in %) to be randomly sampled from input. '
                       ' ** DEFAULT: 50.0')
parser.add_option('--nns_method', dest='searchmethod', default='kdtree',
                  help='Specify method to implement nearest neighbour search -- knn or kdtree? '
                       ' ** DEFAULT: kdtree')
parser.add_option('--vox_factor', dest='div_factor', default=15, type='int',
                  help='Specify voxelization factor (number of voxels along principal axis) '
                       ' ** DEFAULT: 10')
parser.add_option('--vox_thresh', dest='occ_thresh', default=5e-3, type='float',
                  help='Specify threshold as factor of original point cloud size, below which voxels would be ignored'
                       ' ** DEFAULT: 9e-3')
parser.add_option('-v', dest='visualize', action='store_true',
                  help='Enable visualization of graph constructed from input point cloud')
parser.add_option('-o', dest='optimal', action='store_true',
                  help='Enable optimization of nearest neighbour search algorithm, by pre-sorting points')

options, _ = parser.parse_args()
np.random.seed(89)


def construct_vox_graph(vox_pc_map, N, vis_scale, visual=False):
    elapsed = 0.0
    for vox_id in range(len(vox_pc_map)):
        vox_start = time()
        vox_pts = vox_pc_map[vox_id]
        nns = graph.NearestNodeSearch(pointcloud=vox_pts, options=options)
        nns.get_neighbours()
        G = nns.graph
        elapsed = (time() - vox_start) + elapsed
        if vox_id == len(vox_pc_map) - 1:
            print('     Graph construction done.')
        if visual:
            show_voxel(vox_pts, G, vis_scale)
    if visual:
        print('     Visualizing...')
        mlab.show()
    print('----------------------------------------------------------------------------------')
    print('Time elapsed for {} voxel(s) and {} points: {} seconds'.format(len(vox_pc_map), N, round(elapsed, 5)))


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
    print('     Voxelization done.')
    vox_pc_map = grid.voxel_points
    visual = options.visualize
    print('----------------------------------------------------------------------------------')
    print('2.   Constructing graphs for voxels...')
    construct_vox_graph(vox_pc_map, pc.shape[0], vis_scale, visual=visual)
    print('----------------------------------------------------------------------------------')
    print('==================================================================================')

    sys.exit()

if __name__ == "__main__":
    main()
