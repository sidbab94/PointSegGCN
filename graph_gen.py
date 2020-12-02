import numpy as np
from scipy.spatial.distance import cdist
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KDTree
import networkx as nx
import scipy

def sortpoints(points):
    x_sorted = points[np.argsort(points[:, 0])]
    x_sorted = x_sorted[:-round(0.05 * x_sorted.shape[0]), :]  # exclude last 5% of points along x-axis
    if points.shape[0] > 50000:
        Nsort = x_sorted.shape[0]
        points = x_sorted[np.random.choice(Nsort, round(0.05 * Nsort), replace=False), :]
    else:
        points = x_sorted
    return points

def kdtree(points, nn):
    search = NearestNeighbors(n_neighbors=nn+1, algorithm='kd_tree').fit(points)
    dist, idx = search.kneighbors(points)
    return dist[:, 1:], idx[:, 1:]

def kdgraph(points, nn):
    search = NearestNeighbors(n_neighbors=nn+1, algorithm='kd_tree').fit(points[:, :3])
    adj = search.kneighbors_graph(points[:, :3], mode='distance')
    return adj

def show(pts, graph):
    point_size = 0.005
    edge_size = 0.001
    G = nx.convert_node_labels_to_integers(graph)
    scalars = np.array(list(G.nodes())) + 5
    pts = mlab.points3d(
        pts[:, 0],
        pts[:, 1],
        pts[:, 2],
        scalars,
        scale_factor=point_size,
        scale_mode="none",
        colormap="Blues",
        resolution=20)
    pts.mlab_source.dataset.lines = np.array(list(G.edges()))
    tube = mlab.pipeline.tube(pts, tube_radius=edge_size)
    mlab.pipeline.surface(tube, color=(0.8, 0.8, 0.8))
    pts.glyph.scale_mode = 'data_scaling_off'


def adjacency(points, nn):
    dist, idx = kdtree(points[:, :3], nn)
    # print(idx.shape, type(idx))
    M, k = dist.shape
    assert M, k == idx.shape
    assert dist.min() >= 0

    # Weights.
    sigma2 = np.mean(dist[:, -1])**2
    dist = np.exp(- dist**2 / sigma2)
    # Weight matrix.
    I = np.arange(0, M).repeat(k)
    J = idx.reshape(M*k)
    V = dist.reshape(M*k)
    W = scipy.sparse.coo_matrix((V, (I, J)), shape=(M, M))
    # No self-connections.
    W.setdiag(0)

    # Non-directed graph.
    bigger = W.T > W
    W = W - W.multiply(bigger) + W.T.multiply(bigger)

    assert W.nnz % 2 == 0
    assert np.abs(W - W.T).mean() < 1e-10
    assert type(W) is scipy.sparse.csr.csr_matrix
    return nx.from_scipy_sparse_matrix(W)

# from mayavi import mlab
# import voxelization as vox
# from visualization import show_voxel
# pc = np.genfromtxt('samples/airplane.pts', delimiter='')
# # g = adjacency(pc, 2)
# grid = vox.voxelize(pc, 15)
# grid.get_voxels()
# vox_pc_map = grid.voxel_points
# for vox_id in range(len(vox_pc_map)):
#     vox_pts = vox_pc_map[vox_id]
#     g = adjacency(vox_pts, 2)
#     show_voxel(vox_pts, g, vis_scale=0)
# mlab.show()


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
        print(nearest_ind[:, 1:].shape)
        return nearest_ind[:, 1:].tolist()

    def get_neighbours(self):
        indices = np.arange(self.points.shape[0])

        if self.nnsmode == 'kdtree':
            endpoint_indices = self.kdtree()
            # print(endpoint_indices.shape, type(endpoint_indices))
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
