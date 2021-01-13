import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import coo_matrix, csr_matrix

def color_mask(dist_adj, labels):
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

    search = NearestNeighbors(n_neighbors=nn+1, algorithm='kd_tree').fit(points)
    graph = search.kneighbors_graph(points, mode='connectivity')

    return graph


def kdtree(points, nn):
    search = NearestNeighbors(n_neighbors=nn+1, algorithm='kd_tree').fit(points)
    dist, idx = search.kneighbors(points)
    return dist[:, 1:], idx[:, 1:]

def compute_adjacency(points, nn=5, labels=None):
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

    # return W
    print(type(W), W.shape)

if __name__ == '__main__':
    from preproc_utils.readers import read_bin_velodyne
    import os
    BASE_DIR = 'D:/SemanticKITTI/dataset/sequences'

    pc = os.path.join(BASE_DIR, '08', 'velodyne', '000000.bin')
    x = read_bin_velodyne(pc)
