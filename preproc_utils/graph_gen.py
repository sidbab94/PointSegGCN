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

def kdtree(points, nn):
    search = NearestNeighbors(n_neighbors=nn+1, algorithm='kd_tree').fit(points)
    dist, idx = search.kneighbors(points)
    return dist[:, 1:], idx[:, 1:]

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

def adjacency(points, nn=5, labels=None):
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


if __name__ == "__main__":
    pass

