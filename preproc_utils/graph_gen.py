import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import coo_matrix, csr_matrix

def color_mask(dist_adj, labels):
    '''
    Experimental color-masking on the distance-based adjacency matrix
    Color information (for now) derived from label-similarity among points, not viable during testing.
    :param dist_adj: distance-based weighted sparse adjacency matrix
    :param labels: point-wise ground truth label array
    :return: modified adjacency matrix
    '''
    # Get (row,col) array corresponding to non zero elements from distance-based adj matrix
    nz_arr = np.array(dist_adj.nonzero()).T
    # Vectorized computation of labels corresponding to non-zero values
    set_labels = labels[nz_arr]
    assert set_labels.shape == nz_arr.shape
    # Compute boolean vector of label overlaps between neighbours
    eq_check = np.array((set_labels[:, 0] == set_labels[:, 1]), dtype='int')
    # Modify adj data vector with overlap info
    dist_adj.data = dist_adj.data * eq_check

    return dist_adj

def sklearn_graph(points, nn=5):
    '''
    Performs nearest neighbour search and computes a sparse adjacency matrix using Scikit-learn's API
    :param points: point cloud array
    :param nn: number of nearest neighbours to search for
    :return: sparse adjacency matrix
    '''

    search = NearestNeighbors(n_neighbors=nn+1, algorithm='kd_tree').fit(points)
    graph = search.kneighbors_graph(points, mode='distance')

    return graph

def kdtree(points, nn):
    '''
    Performs nearest neighbour search based on Scikit-learn's KD-Tree approach
    :param points: point cloud array
    :param nn: number of nearest neighbours to search for
    :return: nearest neighbour distances and indices
    '''
    search = NearestNeighbors(n_neighbors=nn+1, algorithm='kd_tree').fit(points)
    dist, idx = search.kneighbors(points)
    return dist[:, 1:], idx[:, 1:]

def balltree_graph(points, radius=2.0):
    '''
    Performs radius-based nearest neighbour search and computes a sparse adjacency matrix using Scikit-learn's API
    :param points: point cloud array
    :param radius: radius of sphere to search for neighbours within
    :return: sparse adjacency matrix
    '''
    neigh = NearestNeighbors(radius=radius)
    neigh.fit(points[:, :3])

    return neigh.radius_neighbors_graph(points[:, :3])

def compute_adjacency(points, nn=5, labels=None):
    '''
    Computest a weighted and undirected distance-based adjacency matrix from point cloud array
    :param points: point cloud array
    :param nn: number of nearest neighbours to search for
    :param labels: point-wise ground truth label array
    :return: sparse adjacency matrix
    '''
    # Obtain distances and indices of nearest {nn} neighbours of all points
    dist, idx = kdtree(points[:, :3], nn)
    M, k = dist.shape
    assert M, k == idx.shape
    # Make sure every the query doesn't include the point itself, and only the neighbours
    assert dist.min() >= 0

    # Edge weights computed based on variance
    sigma2 = np.mean(dist[:, -1])**2
    dist = np.exp(- dist**2 / sigma2)

    # Weight matrix construction
    I = np.arange(0, M).repeat(k)
    J = idx.reshape(M*k)
    V = dist.reshape(M*k)
    W = coo_matrix((V, (I, J)), shape=(M, M))
    # No self-loops, remove diagonal non-zero elements
    W.setdiag(0)

    # Non-directed graph.
    bigger = W.T > W
    W = W - W.multiply(bigger) + W.T.multiply(bigger)

    # Final assertion checks
    assert W.nnz % 2 == 0
    assert np.abs(W - W.T).mean() < 1e-10
    assert type(W) is csr_matrix

    if labels != None:
        # Modify wrt node colour information
        W = color_mask(W, labels)

    return W
    # print(type(W), W.shape)

if __name__ == '__main__':
    from preproc_utils.readers import read_bin_velodyne
    from visualization import show_voxel
    from mayavi import mlab
    import os
    BASE_DIR = 'D:/SemanticKITTI/dataset/sequences'

    pc = os.path.join(BASE_DIR, '08', 'velodyne', '000000.bin')
    x = read_bin_velodyne(pc)

    A = compute_adjacency(x)
    show_voxel(x, A)
    mlab.show()
