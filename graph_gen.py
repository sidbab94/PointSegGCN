import numpy as np
from sklearn.neighbors import NearestNeighbors
import scipy

def color_mask(dist_adj, labels):
    # Get (row,col) array corresponding to non zero elements from distance-based adj matrix
    nz_arr = np.array(dist_adj.nonzero()).T
    # Vectorized computation of labels corresponding to non-zero values
    set_labels = labels[nz_arr]
    assert set_labels.shape == nz_arr.shape
    # Compute boolean vector of label overlaps between neighbours
    eq_check = np.array((set_labels[:, 0] == set_labels[:, 1]), dtype='int')
    # print(set(eq_check))
    old_data = np.unique(dist_adj.data)
    # Modify adj data vector with overlap info
    dist_adj.data = dist_adj.data * eq_check
    # check if adj data has been modified
    assert old_data.all() != np.unique(dist_adj.data).all()

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

def adjacency(points, nn, labels):
    dist, idx = kdtree(points[:, :3], nn)
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

    # Modify wrt node colour information
    old = W.data
    W = color_mask(W, labels)
    assert not np.array_equiv(old, W.data)

    return W


if __name__ == "__main__":
    from time import time
    from mayavi import mlab
    import networkx as nx
    pc = np.load('vox_pts.npy')
    lbl = np.load('vox_lbl.npy')
    tic = time()
    w = adjacency(pc, 2, lbl)
    print(time() - tic)

