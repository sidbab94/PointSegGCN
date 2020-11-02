import numpy as np
from numpy import genfromtxt
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from time import time
import networkx as nx
from mayavi import mlab
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
# print(tf.test.is_gpu_available())
# print(tf.config.list_physical_devices())


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
    with tf.device('GPU:0'):
        distances = []
        for i in range(y.shape[0]):
            l2_norm = tf.norm(x - y[i, :], ord='euclidean')
            distances.append(l2_norm)
    return distances

def get_neighbours(points):
    """
    find k nearest neighbours of each point, draw edges
    """
    indices = np.arange(points.shape[0])
    nn = 2
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


def get_neighbours_gpu(points):
    """
    find k nearest neighbours of each point, draw edges
    """
    indices = np.arange(points.shape[0])
    nn = 2
    list_of_points = list(points.tolist())
    pointpairs = []
    for point in enumerate(points):
        curr_point_indx = point[0]
        curr_point = np.array([point[1]])

        targets = points[indices != curr_point_indx, :]
        # distances = cdist(targets, curr_point, metric='euclidean').flatten().tolist()


        distances = sess.run(eucl_dist(targets, curr_point))
        print(distances)
        # print(distances == distances2)


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
    ax1 = plt.subplot(projection='3d')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.scatter(points[:, 0], points[:, 1], points[:, 2], marker='x', label='Original')
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
        scale_factor=0.1,
        scale_mode="none",
        colormap="Blues",
        resolution=20,
    )

    pts.mlab_source.dataset.lines = np.array(list(G.edges()))
    tube = mlab.pipeline.tube(pts, tube_radius=0.01)
    mlab.pipeline.surface(tube, color=(0.8, 0.8, 0.8))
    mlab.show()

np.random.seed(89)
# rand_array = np.random.rand(20, 3)
# rand_array = generate_3d_data(m=1000)
rand_array = genfromtxt('000001.pts', delimiter='')
print(rand_array.shape)


# plot3D(rand_array)
points_dict = dict(enumerate(rand_array.tolist()))
G = Graph(numNodes=rand_array.shape[0])
start = time()
with tf.Session() as sess:
    index_pairs = get_neighbours(rand_array)
print('Time elapsed in seconds for {} points: {}'.format(rand_array.shape[0], time()-start))
for pair in index_pairs:
    startindex = pair[0]
    for endindex in pair[1]:
        G.addEdge(startindex, endindex)

A = G.adjacencyMatrix
A = np.array(A)
X = nx.Graph(A)
plt.show()

# visualize_graph(X, rand_array)

