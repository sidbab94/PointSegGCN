import numpy as np
from scipy.spatial.distance import cdist
from sklearn.neighbors import NearestNeighbors
import networkx as nx

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
        self.points = self.points_alldims[:, :3]
        self.N = self.points.shape[0]
        self.nn = options.nearestN
        self.nnsmode = options.searchmethod
        self.adjacencyMatrix = np.zeros((self.N, self.N), dtype='uint8')

    def kdtree(self):
        search = NearestNeighbors(n_neighbors=self.nn + 1, algorithm='kd_tree').fit(self.points)
        _, nearest_ind = search.kneighbors(self.points)
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
            for point in enumerate(self.points):
                self.curr_point_indx = point[0]
                self.curr_point = np.array([point[1]])
                targets = self.points
                distances = cdist(targets, self.curr_point, metric='euclidean').flatten()
                nn_indices = np.argsort(distances)[1:self.nn + 1]
                for i in range(len(nn_indices)):
                    self.addEdges(self.curr_point_indx, nn_indices[i])

        assert (self.adjacencyMatrix.transpose() == self.adjacencyMatrix).all()
        self.graph = nx.Graph(self.adjacencyMatrix)

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
