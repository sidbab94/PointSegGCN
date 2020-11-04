import numpy as np
from numpy import genfromtxt
from scipy.spatial.distance import cdist
from time import time
import networkx as nx
from mayavi import mlab
from optparse import OptionParser
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


def eucl_dist(target_points, source_point):
    """

    :param target_points: Target points coordinates (n-1) x XYZ
    :param source_point: Source point coordinates XYZ
    :return: Euclidean (L2) norm of above, computed using tensorflow
    """
    x = source_point
    y = target_points
    l2_norm = tf.norm(x - y, ord='euclidean', axis=1)
    print('Next point.')
    return l2_norm

def get_neighbours_gpu(points, nn):
    """
    TARGET DEVICE - GPU
    :param points: complete point cloud as numpy array
    :param nn: nearest neighbours to search for
    :return: global index map of each point to corresponding neighbours
    """
    # all indices from 0 to total num_points-1
    indices = np.arange(points.shape[0])
    # list of all points
    list_of_points = list(points.tolist())
    # pair of start point and end(neighbour) points in index format
    pointpairs = []

    X = tf.placeholder(shape=[1, 3], dtype=tf.float32)
    Y = tf.placeholder(shape=[points.shape[0]-1, 3], dtype=tf.float32)
    dist_comp = eucl_dist(X, Y)
    with tf.Session() as sess:
        for point in enumerate(points):
            # current point index
            curr_point_indx = point[0]
            # current point coordinates in np array format
            curr_point = np.array([point[1]])

            # all points except current point
            targets = points[indices != curr_point_indx, :]

            # Run TF session on euclidean distance computation function; Input - Current point, Remaining points
            sess.run(tf.initializers.global_variables())
            distances = sess.run(dist_comp, feed_dict={X: curr_point, Y: targets})

            # index -> coordinates map for target points
            target_indx_dict = dict(enumerate(targets))
            # index -> distances map:       dict
            dist_indx_dict = dict(enumerate(distances))

            # sorted distances in ascending order
            sorted_distances = {k: v for k, v in sorted(dist_indx_dict.items(), key=lambda item: item[1])}
            # nearest neighbour indices:    list
            nn_indices = list(sorted_distances.keys())[:nn]
            # nearest points:       list
            nearest_points = list(target_indx_dict[i].tolist() for i in nn_indices)

            # current point
            start = point[1].flatten()
            # current point global index
            startpoint_index = list_of_points.index(start.tolist())
            # end point(s) global indices
            endpoint_indices = [list_of_points.index(nearest_points[i]) for i in range(len(nearest_points))]
            # start-end point global index pairs
            pointpairs.extend([[startpoint_index, endpoint_indices]])

    return pointpairs

def get_neighbours(points, nn):
    """
    TARGET DEVICE - CPU
    :param points: complete point cloud as numpy array
    :param nn: nearest neighbours to search for
    :return: global index map of each point to corresponding neighbours
    """
    # all indices from 0 to total num_points-1
    indices = np.arange(points.shape[0])
    # list of all points
    list_of_points = list(points.tolist())
    # pair of start point and end(neighbour) points in index format
    pointpairs = []

    for point in enumerate(points):
        # current point index
        curr_point_indx = point[0]
        # current point coordinates in np array format
        curr_point = np.array([point[1]])

        # all points except current point
        targets = points[indices != curr_point_indx, :]
        # Compute euclidean distance between points using scipy
        distances = cdist(targets, curr_point, metric='euclidean').flatten().tolist()

        # index -> coordinates map for target points
        target_indx_dict = dict(enumerate(targets))
        # index -> distances map:       dict
        dist_indx_dict = dict(enumerate(distances))

        # sorted distances in ascending order
        sorted_distances = {k: v for k, v in sorted(dist_indx_dict.items(), key=lambda item: item[1])}
        # nearest neighbour indices:    list
        nn_indices = list(sorted_distances.keys())[:nn]
        assert len(nn_indices) == nn
        # nearest points:       list
        nearest_points = list(target_indx_dict[i].tolist() for i in nn_indices)
        assert len(nearest_points) == nn
        # current point
        start = point[1].flatten()
        # current point global index
        startpoint_index = list_of_points.index(start.tolist())
        # end point(s) global indices
        endpoint_indices = [list_of_points.index(nearest_points[i]) for i in range(len(nearest_points))]
        # start-end point global index pairs
        pointpairs.extend([[startpoint_index, endpoint_indices]])

    return pointpairs

class Graph(object):
    """
    Graph constructor class
    """
    def __init__(self, numNodes):
        self.adjacencyMatrix = []  # 2D list
        for i in range(numNodes):
            self.adjacencyMatrix.append([0 for i in range(numNodes)])
        self.numNodes = numNodes

    def addEdge(self, start, end):
        """

        :param start: Start node global index
        :param end: End node global index
        :return: Modified adjacency matrix
        """
        self.adjacencyMatrix[start][end] = 1

    def removeEdge(self, start, end):
        if self.adjacencyMatrix[start][end] == 0:
            print("There is no edge between %d and %d" % (start, end))
        else:
            self.adjacencyMatrix[start][end] = 0

    def containsEdge(self, start, end):
        if self.adjacencyMatrix[start][end] > 0:
            return True
        else:
            return False

    def __len__(self):
        return self.numNodes


def create_8bit_rgb_lut():
    """
    :return: Look-Up Table as 256**3 x 4 array
    """
    xl = np.mgrid[0:256, 0:256, 0:256]
    lut = np.vstack((xl[0].reshape(1, 256**3),
                        xl[1].reshape(1, 256**3),
                        xl[2].reshape(1, 256**3),
                        255 * np.ones((1, 256**3)))).T
    return lut.astype('int32')

def rgb_2_scalar_idx(r, g, b):
    """

    :param r: Red value
    :param g: Green value
    :param b: Blue value
    :return: scalar index of input colour
    """
    return 256**2 *r + 256 * g + b


def visualize_graph(graph, array):
    """

    :param graph: Input graph constructed from adj matrix
    :param array: Input point cloud array (coarsened)
    :return: Visualization of point cloud with graph network overlay
    """
    G = nx.convert_node_labels_to_integers(graph)
    xyz = array[:, :3]
    rgb = array[:, 3:]
    scalars = np.zeros((rgb.shape[0], ))

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
    pts.glyph.glyph.clamping = False

    mlab.show()


np.random.seed(89)

parser = OptionParser()

parser.add_option('--input', dest='inputpc', default='airplane.pts',
                  help='Provide input point cloud file in .pts format')
parser.add_option('--kNN', dest='nearestN', default=3, type='int',
                  help='Provide number of nearest neighbours to process each point with')
parser.add_option('--sample_size', dest='ss', default=2.0, type='float',
                  help='Provide proportion of points (in %) to be randomly sampled from input distribution')
parser.add_option('-g', dest='gpu_en', action='store_true',
                  help='Enable GPU for euclidean distance computation (may be 10X slower!)')
parser.add_option('-v', dest='visualize', action='store_true',
                  help='Enable visualization of graph constructed from input point cloud')
options, _ = parser.parse_args()

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
sample_size = round(options.ss * 0.01 * N)
NN = options.nearestN
pc_array = pc_array[np.random.choice(N, sample_size, replace=False), :]
pc_array_XYZ = pc_array[:, :3]

points_dict = dict(enumerate(pc_array_XYZ.tolist()))
G = Graph(numNodes=pc_array.shape[0])

if options.gpu_en:
    device = 'GPU'
    start = time()
    index_pairs_cpu = get_neighbours_gpu(pc_array_XYZ, NN)
else:
    device = 'CPU'
    start = time()
    index_pairs_cpu = get_neighbours(pc_array_XYZ, NN)
print('Time elapsed in seconds for {} points: {}, using {}'.format(pc_array.shape[0], time()-start, device))

for pair in index_pairs_cpu:
    startindex = pair[0]
    for endindex in pair[1]:
        G.addEdge(startindex, endindex)

A = G.adjacencyMatrix
A = np.array(A)
X = nx.Graph(A)

if options.visualize:
    visualize_graph(X, pc_array)
