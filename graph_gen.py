import numpy as np
from numpy import genfromtxt
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D
from scipy.spatial.distance import cdist
from time import time
from time import sleep
import networkx as nx
from mayavi import mlab
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

print(tf.test.is_gpu_available())
print(tf.config.list_physical_devices())


# with tf.device('GPU:0'):
#     a = tf.constant([1.0, 2.0, 3.0], shape=[1, 3], name='a')
#     s = np.array([[4.0, 3.0, 2.0], [5.0, 3.0, 2.0]]).astype('float32')
#     distances = []
#     for i in range(s.shape[0]):
#         l2_norm = tf.norm(a - s[i, :], ord='euclidean')
#         distances.append(l2_norm)
#         # b = tf.constant(s, shape=[2, 3], name='b')
#     # c = tf.matmul(a, b)
#     # l2_norm = tf.norm(a - b, ord='euclidean')
#
# with tf.Session() as sess:
#     o = sess.run(distances)

def eucl_dist(target_points, source_point):
    x = source_point
    y = target_points
    l2_norm = tf.norm(x - y, ord='euclidean', axis=1)
    print('Next point.')
    return l2_norm

def get_neighbours_gpu(points, nn):
    """
    find k nearest neighbours of each point, draw edges
    """
    # all indices from 0 to total num_points-1:     list
    indices = np.arange(points.shape[0])
    # nearest neighbours to search for
    # list of all points:   list
    list_of_points = list(points.tolist())
    # pair of start point and end(neighbour) points in index format:    list
    pointpairs = []

    X = tf.placeholder(shape=[1, 3], dtype=tf.float32)
    Y = tf.placeholder(shape=[points.shape[0]-1, 3], dtype=tf.float32)
    dist_comp = eucl_dist(X, Y)
    with tf.Session() as sess:
        for point in enumerate(points):
            # current point index:  int
            curr_point_indx = point[0]
            # current point coordinates in np array format:     np array
            curr_point = np.array([point[1]])

            # all points except current point:      np array
            targets = points[indices != curr_point_indx, :]

            # with tf.Session() as sess:
            sess.run(tf.initializers.global_variables())
            distances = sess.run(dist_comp, feed_dict={X: curr_point, Y: targets})

            # # euclidean distances:      np array
            # distances = eucl_dist(X, Y)

            # index -> coordinates map for target points:       dict
            target_indx_dict = dict(enumerate(targets))
            # index -> distances map:       dict
            dist_indx_dict = dict(enumerate(distances))

            # sorted distances in ascending order:      dict
            sorted_distances = {k: v for k, v in sorted(dist_indx_dict.items(), key=lambda item: item[1])}
            # nearest neighbour indices:    list
            nn_indices = list(sorted_distances.keys())[:nn]
            # nearest points:       list
            nearest_points = list(target_indx_dict[i].tolist() for i in nn_indices)

            # current point:       np array
            start = point[1].flatten()
            # current point global index:   int
            startpoint_index = list_of_points.index(start.tolist())
            # end point(s) global indices:      int
            endpoint_indices = [list_of_points.index(nearest_points[i]) for i in range(len(nearest_points))]
            # start-end point global index pairs:       list
            pointpairs.extend([[startpoint_index, endpoint_indices]])
    return pointpairs

def get_neighbours(points, nn):
    """
    find k nearest neighbours of each point, draw edges
    """
    indices = np.arange(points.shape[0])
    list_of_points = list(points.tolist())
    pointpairs = []

    for point in enumerate(points):
        curr_point_indx = point[0]
        curr_point = np.array([point[1]])

        targets = points[indices != curr_point_indx, :]
        distances = cdist(targets, curr_point, metric='euclidean').flatten().tolist()

        target_indx_dict = dict(enumerate(targets))
        dist_indx_dict = dict(enumerate(distances))

        sorted_distances = {k: v for k, v in sorted(dist_indx_dict.items(), key=lambda item: item[1])}
        nn_indices = list(sorted_distances.keys())[:nn]
        nearest_points = list(target_indx_dict[i].tolist() for i in nn_indices)

        start = point[1].flatten()
        startpoint_index = list_of_points.index(start.tolist())
        endpoint_indices = [list_of_points.index(nearest_points[i]) for i in range(len(nearest_points))]
        pointpairs.extend([[startpoint_index, endpoint_indices]])
    return pointpairs

def plot3D(points):
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], marker='x', label='Original')
    plt.autoscale()

class Graph(object):
    def __init__(self, numNodes):
        self.adjacencyMatrix = []  # 2D list
        for i in range(numNodes):
            self.adjacencyMatrix.append([0 for i in range(numNodes)])
        self.numNodes = numNodes

    def addEdge(self, start, end):
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


def generate_3d_data(m, w1=0.1, w2=0.3, noise=0.1):
    angles = np.random.rand(m) * 3 * np.pi / 2 - 0.5
    data = np.empty((m, 3))
    data[:, 0] = np.cos(angles) + np.sin(angles)/2 + noise * np.random.randn(m) / 2
    data[:, 1] = np.sin(angles) * 0.7 + noise * np.random.randn(m) / 2
    data[:, 2] = data[:, 0] * w1 + data[:, 1] * w2 + noise * np.random.randn(m)
    return data

def visualize_graph(graph, array):
    G = nx.convert_node_labels_to_integers(graph)
    xyz = array
    scalars = np.array(list(G.nodes())) + 5
    pts = mlab.points3d(
        xyz[:, 0],
        xyz[:, 1],
        xyz[:, 2],
        scalars,
        scale_factor=0.025,
        scale_mode="none",
        colormap="Blues",
        resolution=20,
    )

    pts.mlab_source.dataset.lines = np.array(list(G.edges()))
    tube = mlab.pipeline.tube(pts, tube_radius=0.005)
    mlab.pipeline.surface(tube, color=(0.8, 0.8, 0.8))
    mlab.show()

np.random.seed(89)
# rand_array = np.random.rand(100, 3)
# rand_array = generate_3d_data(m=1000)
rand_array = genfromtxt('000001.pts', delimiter='')
NN = 10

N = rand_array.shape[0]
rand_array = rand_array[np.random.choice(N, round(0.04*N), replace=False), :]
# print(rand_array.shape)


# plot3D(rand_array)
points_dict = dict(enumerate(rand_array.tolist()))
G = Graph(numNodes=rand_array.shape[0])

start_cpu = time()
index_pairs_cpu = get_neighbours(rand_array, 3)
print('Time elapsed in seconds for {} points: {}, using CPU'.format(rand_array.shape[0], time()-start_cpu))
# print(type(index_pairs_cpu))
# print(len(index_pairs_cpu))
sleep(1)
start_gpu = time()
index_pairs_gpu = get_neighbours_gpu(rand_array, 3)
print('Time elapsed in seconds for {} points: {}, using GPU'.format(rand_array.shape[0], time()-start_gpu))
# print(type(index_pairs_gpu))
# print(len(index_pairs_gpu))

for pair in index_pairs_gpu:
    startindex = pair[0]
    for endindex in pair[1]:
        G.addEdge(startindex, endindex)

A = G.adjacencyMatrix
A = np.array(A)
X = nx.Graph(A)
plt.show()

visualize_graph(X, rand_array)

