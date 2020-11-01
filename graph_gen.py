import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from time import time
import networkx as nx
plt.switch_backend('Qt5Agg')


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

def plot3D(points):
    ax1 = plt.subplot(projection='3d')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.scatter(points[:, 0], points[:, 1], points[:, 2], marker='x', label='Original')
    plt.autoscale()
    plt.show()


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

np.random.seed(89)
# rand_array = np.random.rand(20, 3)
# plot3D(rand_array)
rand_array = generate_3d_data(m=20)
G = Graph(numNodes=rand_array.shape[0])
"""
find a way to construct edges, by passing indices of start and end points to Graph()
"""
start = time()
index_pairs = get_neighbours(rand_array)
print('Time elapsed in seconds for {} points: {}'.format(rand_array.shape[0], time()-start))
for pair in index_pairs:
    startindex = pair[0]
    for endindex in pair[1]:
        G.addEdge(startindex, endindex)

A = G.adjacencyMatrix
A = np.array(A)
print(A.shape)
graph = nx.Graph(A)
nx.draw(graph)
plt.show()