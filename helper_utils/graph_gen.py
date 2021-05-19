from sklearn.neighbors import NearestNeighbors, radius_neighbors_graph, kneighbors_graph
from scipy.sparse import csr_matrix, spdiags

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


def intensity_mask(dist_adj, x):

    # Get (row,col) array corresponding to non zero elements from distance-based adj matrix
    nz_arr = np.array(dist_adj.nonzero()).T

    # Vectorized computation of labels corresponding to non-zero values
    nz_intensity = x[:, 3][nz_arr]

    # compute similarity mask
    sim_mask = np.isclose(nz_intensity[:, 0], nz_intensity[:, 1], rtol=1e-02, equal_nan=False)

    dist_adj.eliminate_zeros()

    assert nz_arr.shape[0] == dist_adj.data.shape[0]

    assert dist_adj.data.shape == sim_mask.shape

    # Modify adj data vector with overlap info
    dist_adj.data = dist_adj.data * sim_mask

    return dist_adj

def sklearn_graph(points, nn=5):
    '''
    Performs nearest neighbour search and computes a sparse adjacency matrix using Scikit-learn's API
    :param points: point cloud array
    :param nn: number of nearest neighbours to search for
    :return: sparse adjacency matrix
    '''

    # search = NearestNeighbors(n_neighbors=nn+1, algorithm='kd_tree').fit(points)
    graph = kneighbors_graph(points[:, :3], n_neighbors=nn, mode='connectivity', include_self=True)

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

def balltree_graph(points, radius=0.3, mode='connectivity'):
    '''
    Performs radius-based nearest neighbour search and computes a sparse adjacency matrix using Scikit-learn's API
    :param points: point cloud array
    :param radius: radius of sphere to search for neighbours within
    :return: sparse adjacency matrix
    '''

    return radius_neighbors_graph(points[:, :3], radius, mode=mode, include_self=True)

def compute_adjacency(points, nn=5):
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
    A = csr_matrix((V, (I, J)), shape=(M, M))

    # Final assertion checks
    # assert A.nnz % 2 == 0
    assert type(A) is csr_matrix

    # A = intensity_mask(A, points)

    return A

def normalize_A(a):

    n, m = a.shape
    diags = a.sum(axis=1).flatten()
    D = spdiags(diags, [0], m, n, format="csr")
    D = D.power(-0.5)
    L = D.dot(a).dot(D)
    return L

if __name__ == '__main__':
    from helper_utils.visualization import PC_Vis

    from preprocess import *

    model_cfg = get_cfg_params(cfg_file='../config/tr_config.yml')
    prep = Preprocess(model_cfg)

    train_files, _, _ = get_split_files(cfg=model_cfg, shuffle=False)
    file_list = train_files[:3]
    sample = file_list[2]

    x, a, y = prep.assess_scan(sample)
    print(x.shape, a.shape)
    PC_Vis.draw_graph(x, a)

