import numpy as np
from numpy import genfromtxt
from scipy.spatial.distance import cdist
from sklearn.neighbors import KDTree
from time import time
import networkx as nx
from mayavi import mlab
from optparse import OptionParser

parser = OptionParser()
parser.add_option('--input', dest='inputpc', default='sample.npy',
                  help='Provide input point cloud file in .pts format')
parser.add_option('--NN', dest='nearestN', default=1, type='int',
                  help='Provide number of nearest neighbours to process each point with')
parser.add_option('--sample_size', dest='ss', default=100.0, type='float',
                  help='Provide proportion of points (in %) to be randomly sampled from input distribution')
parser.add_option('--nns_method', dest='searchmethod', default='knn',
                  help='Specify method to implement nearest neighbour search -- knn or kdtree?')
parser.add_option('-v', dest='visualize', action='store_true',
                  help='Enable visualization of graph constructed from input point cloud')
parser.add_option('-o', dest='optimal', action='store_true',
                  help='Enable optimization of nearest neighbour search algorithm, in favour of performance')

options, _ = parser.parse_args()
np.random.seed(89)


def sortpoints(points):
    x_sorted = points[np.argsort(points[:, 0])]
    x_sorted = x_sorted[:-round(0.05 * x_sorted.shape[0]), :]  # exclude last 5% of points along x-axis
    if points.shape[0] > 50000:
        Nsort = x_sorted.shape[0]
        points = x_sorted[np.random.choice(Nsort, round(0.05 * Nsort), replace=False), :]
    else:
        points = x_sorted
    return points

class NearestNodeSearch:
    def __init__(self, pointcloud, options):
        self.points_alldims = pointcloud
        self.optimal = options.optimal
        if self.optimal:
            self.points_alldims = sortpoints(self.points_alldims)
        self.points = self.points_alldims[:, :3]
        self.N = self.points.shape[0]
        self.nn = options.nearestN
        self.nnsmode = options.searchmethod
        self.adjacencyMatrix = np.zeros((self.N, self.N), dtype='uint8')

    def kdtree(self):
        tree = KDTree(self.points)
        _, nearest_ind = tree.query(self.points, k=self.nn + 1)
        return nearest_ind[:, 1:].tolist()

    def get_neighbours(self):
        indices = np.arange(self.points.shape[0])

        if self.nnsmode == 'kdtree':
            endpoint_indices = self.kdtree()
            nn_indices = [list(i) for i in zip(indices, endpoint_indices)]
            for pair in nn_indices:
                i = pair[0]
                for j in pair[1]:
                    self.addEdges(i, j)
        else:
            list_of_points = list(self.points.tolist())
            for point in enumerate(self.points):
                self.curr_point_indx = point[0]
                self.curr_point = np.array([point[1]])
                # if self.roisearch:
                #     targets = self.searchROI(radius=1000)
                #     distances = cdist(targets, self.curr_point, metric='euclidean').flatten()
                #     nn_indices = np.argsort(distances)[1:self.nn + 1]
                #     nearest_points = list(targets.tolist()[i] for i in nn_indices)
                #     global_nn_indices = [list_of_points.index(nearest_points[i]) for i in range(len(nearest_points))]
                #     for i in range(len(global_nn_indices)):
                #         self.addEdges(self.curr_point_indx, global_nn_indices[i])
                # else:
                targets = self.points
                distances = cdist(targets, self.curr_point, metric='euclidean').flatten()
                nn_indices = np.argsort(distances)[1:self.nn + 1]
                for i in range(len(nn_indices)):
                    self.addEdges(self.curr_point_indx, nn_indices[i])


    def searchROI(self, radius=100):
        last_point_indx = self.points.shape[0] - 1
        n_bwd = self.curr_point_indx
        n_fwd = last_point_indx - self.curr_point_indx
        roi_fwd = roi_bwd = radius
        if n_bwd < radius:
            roi_bwd = n_bwd
        if n_fwd < radius:
            roi_fwd = n_fwd
        target_points = self.points[self.curr_point_indx - roi_bwd:self.curr_point_indx + roi_fwd + 1, :]
        return target_points

    def addEdges(self, start, end):
        self.adjacencyMatrix[start, end] = 1
        self.adjacencyMatrix[end, start] = 1

    def containsEdge(self, start, end):
        if self.adjacencyMatrix[start, end] > 0:
            return True
        else:
            return False

    def removeEdge(self, start, end):
        if self.adjacencyMatrix[start, end] == 0:
            print("There is no edge between %d and %d" % (start, end))
        else:
            self.adjacencyMatrix[start, end] = 0


def create_8bit_rgb_lut():
    """
    :return: Look-Up Table as 256**3 x 4 array
    """
    xl = np.mgrid[0:256, 0:256, 0:256]
    lut = np.vstack((xl[0].reshape(1, 256 ** 3),
                     xl[1].reshape(1, 256 ** 3),
                     xl[2].reshape(1, 256 ** 3),
                     255 * np.ones((1, 256 ** 3)))).T
    return lut.astype('int32')

def rgb_2_scalar_idx(r, g, b):
    """

    :param r: Red value
    :param g: Green value
    :param b: Blue value
    :return: scalar index of input colour
    """
    return 256 ** 2 * r + 256 * g + b

def visualize_graph_rgb(graph, array):
    """

    :param graph: Input graph constructed from adj matrix
    :param array: Input point cloud array (coarsened)
    :return: Visualization of point cloud with graph network overlay
    """
    mlab.figure(size=(1200, 800), bgcolor=(1, 1, 1))
    G = nx.convert_node_labels_to_integers(graph)
    xyz = array[:, :3]
    rgb = array[:, 3:]
    scalars = np.zeros((rgb.shape[0],))

    for (kp_idx, kp_c) in enumerate(rgb):
        scalars[kp_idx] = rgb_2_scalar_idx(kp_c[0], kp_c[1], kp_c[2])

    rgb_lut = create_8bit_rgb_lut()

    pts = mlab.points3d(
        xyz[:, 0],
        xyz[:, 1],
        xyz[:, 2],
        scalars,
        scale_factor=0.2,
        scale_mode="none",
        resolution=20,
    )

    pts.mlab_source.dataset.lines = np.array(list(G.edges()))
    tube = mlab.pipeline.tube(pts, tube_radius=0.01)
    mlab.pipeline.surface(tube, color=(0.8, 0.8, 0.8))

    pts.module_manager.scalar_lut_manager.lut._vtk_obj.SetTableRange(0, rgb_lut.shape[0])
    pts.module_manager.scalar_lut_manager.lut.number_of_colors = rgb_lut.shape[0]
    pts.module_manager.scalar_lut_manager.lut.table = rgb_lut
    pts.glyph.scale_mode = 'data_scaling_off'
    # pts.glyph.glyph.clamping = False

    mlab.show()

def visualize_graph_xyz(graph, array):
    mlab.figure(size=(1200, 800), bgcolor=(0, 0, 0))
    G = nx.convert_node_labels_to_integers(graph)
    xyz = array
    scalars = np.array(list(G.nodes())) + 5
    pts = mlab.points3d(
        xyz[:, 0],
        xyz[:, 1],
        xyz[:, 2],
        scalars,
        scale_factor=0.3,
        scale_mode="none",
        colormap="Reds",
        resolution=20,
    )

    pts.mlab_source.dataset.lines = np.array(list(G.edges()))
    tube = mlab.pipeline.tube(pts, tube_radius=0.08)
    mlab.pipeline.surface(tube, color=(0.8, 0.8, 0.8))
    pts.glyph.glyph.clamping = False
    pts.glyph.scale_mode = 'data_scaling_off'
    mlab.show()


def main():
    input_cloud_fmt = options.inputpc.split('.')[1]
    if input_cloud_fmt == 'pts':
        pc_array = genfromtxt(options.inputpc, delimiter='')
    elif input_cloud_fmt == 'csv':
        pc_array = genfromtxt(options.inputpc, delimiter=',')
    elif input_cloud_fmt == 'npy':
        pc_array = np.load(options.inputpc)
    else:
        raise Exception('Provide a supported point cloud format')

    N = pc_array.shape[0]
    if options.ss != 100.0:
        sample_size = round(options.ss * 0.01 * N)
        pc_array = pc_array[np.random.choice(N, sample_size, replace=False), :]
    nns = NearestNodeSearch(pointcloud=pc_array, options=options)
    nns_sample = nns.points_alldims

    start = time()
    nns.get_neighbours()
    A = nns.adjacencyMatrix
    print('NNS Graph Construction: Time elapsed in seconds for {} points: {}'.format(nns.N, time() - start))
    assert (A.transpose() == A).all()

    X = nx.Graph(A)

    if options.visualize:
        if pc_array.shape[1] > 4:
            visualize_graph_rgb(X, nns_sample)
        else:
            visualize_graph_xyz(X, nns_sample)


if __name__ == "__main__":
    main()
