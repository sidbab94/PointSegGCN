import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from time import time
import networkx as nx

def get_neighbours(points):
    """
    find k nearest neighbours of each point, draw edges
    """
    indices = np.arange(points.shape[0])
    nn = 2
    for point in enumerate(points):
        curr_point_indx = point[0]
        curr_point = np.array([point[1]])
        targets = points[indices != curr_point_indx, :]
        distances = cdist(targets, curr_point, metric='euclidean').flatten().tolist()
        target_indx_dict = dict(enumerate(targets))
        dist_indx_dict = dict(enumerate(distances))
        sorted_distances = {k: v for k, v in sorted(dist_indx_dict.items(), key=lambda item: item[1])}
        # print('---------')
        nn_indices = list(sorted_distances.keys())[:nn]
        nearest_points = list(target_indx_dict[i] for i in nn_indices)
        start, end = point[1].flatten(), nearest_points
        return start, end


def plot3D(points):
    fig = plt.figure()
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


np.random.seed(89)
rand_array = np.random.rand(10, 3)
G = Graph(numNodes=rand_array.shape[0])
"""
find a way to construct edges, by passing indices of start and end points to Graph()
"""
start = time()
get_neighbours(rand_array)
print('Time elapsed in seconds for {} points: {}'.format(rand_array.shape[0], time()-start))
