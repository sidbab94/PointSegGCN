import numpy as np
from numpy import genfromtxt
from time import time
import graph_gen as graph
import voxelization as vox
from visualization import show_voxel
from optparse import OptionParser
from mayavi import mlab


parser = OptionParser()
parser.add_option('--input', dest='inputpc', default='samples/vkitti3d_01.npy',
                  help='Provide input point cloud file in .pts format')
parser.add_option('--NN', dest='nearestN', default=2, type='int',
                  help='Provide number of nearest neighbours to process each point with')
parser.add_option('--sample_size', dest='ss', default=50.0, type='float',
                  help='Provide proportion of points (in %) to be randomly sampled from input distribution')
parser.add_option('--nns_method', dest='searchmethod', default='kdtree',
                  help='Specify method to implement nearest neighbour search -- knn or kdtree?')
parser.add_option('-v', dest='visualize', action='store_true',
                  help='Enable visualization of graph constructed from input point cloud')
parser.add_option('-o', dest='optimal', action='store_true',
                  help='Enable optimization of nearest neighbour search algorithm, by pre-sorting points')
parser.add_option('--vox_factor', dest='div_factor', default=10, type='int',
                  help='Specify voxelization factor (number of voxels along principal axis)')
parser.add_option('--vox_thresh', dest='occ_thresh', default=9e-3, type='float',
                  help='Specify threshold as factor of original point cloud size, below which voxels would be ignored')

options, _ = parser.parse_args()
np.random.seed(89)



def construct_vox_graph(vox_pc_map, visual=False):
    for vox_id in range(len(vox_pc_map)):

        vox_pts = vox_pc_map[vox_id]
        nns = graph.NearestNodeSearch(pointcloud=vox_pts, options=options)
        nns.get_neighbours()
        G = nns.graph

        if visual:
            show_voxel(vox_pts, G)
    if visual:
        mlab.show()


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
    print('     Voxelization done.')
    vox_pc_map = grid.voxel_points
    visual = options.visualize
    print('----------------------------------------------------------------------------------')
    print('2.   Constructing graphs for voxels...')
    vox_start = time()
    construct_vox_graph(vox_pc_map, visual=visual)
    print('     Graph construction done.')
    print('----------------------------------------------------------------------------------')
    print('Time elapsed for {} voxel(s) and {} points: {} s'.format(len(vox_pc_map)+1, pc.shape[0], time() - vox_start))
    print('----------------------------------------------------------------------------------')
    print('==================================================================================')


if __name__ == "__main__":
    main()