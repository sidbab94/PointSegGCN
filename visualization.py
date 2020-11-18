import numpy as np
from mayavi import mlab
import networkx as nx


def create_8bit_rgb_lut():
    """
    :return: Look-Up Table as 256**3 x 4 array
    """
    xl = np.mgrid[0:256, 0:256, 0:256]
    lut = np.vstack((xl[0].reshape(1, 256**3),
                        xl[1].reshape(1, 256**3),
                        xl[2].reshape(1, 256**3),
                        255 * np.ones((1, 256**3)))).T
    return lut.astype('int32')

def rgb_2_scalar_idx(r, g, b):
    """

    :param r: Red value
    :param g: Green value
    :param b: Blue value
    :return: scalar index of input colour
    """
    return 256**2 *r + 256 * g + b

def cube_faces(xmin, xmax, ymin, ymax, zmin, zmax):
    faces = []

    x, y = np.mgrid[xmin:xmax:3j, ymin:ymax:3j]
    z = np.ones(y.shape) * zmin
    faces.append((x, y, z))

    x, y = np.mgrid[xmin:xmax:3j, ymin:ymax:3j]
    z = np.ones(y.shape) * zmax
    faces.append((x, y, z))

    x, z = np.mgrid[xmin:xmax:3j, zmin:zmax:3j]
    y = np.ones(z.shape) * ymin
    faces.append((x, y, z))

    x, z = np.mgrid[xmin:xmax:3j, zmin:zmax:3j]
    y = np.ones(z.shape) * ymax
    faces.append((x, y, z))

    y, z = np.mgrid[ymin:ymax:3j, zmin:zmax:3j]
    x = np.ones(z.shape) * xmin
    faces.append((x, y, z))

    y, z = np.mgrid[ymin:ymax:3j, zmin:zmax:3j]
    x = np.ones(z.shape) * xmax
    faces.append((x, y, z))

    return faces

def mlab_plt_cube(xmin, xmax, ymin, ymax, zmin, zmax):
    faces = cube_faces(xmin, xmax, ymin, ymax, zmin, zmax)
    for grid in faces:
        x, y, z = grid
        mlab.mesh(x, y, z, opacity=0.2, color=(0.1, 0.2, 0.3))

def show_voxel(vox_pts, graph, vis_scale):
    if vis_scale == 1:
        point_size = 0.2
        edge_size = 0.02
    else:
        point_size = 0.01
        edge_size = 0.003

    G = nx.convert_node_labels_to_integers(graph)
    if vox_pts.shape[1] > 3:
        rgb = vox_pts[:, 3:]
        scalars = np.zeros((rgb.shape[0],))
        for (kp_idx, kp_c) in enumerate(rgb):
            scalars[kp_idx] = rgb_2_scalar_idx(kp_c[0], kp_c[1], kp_c[2])
        rgb_lut = create_8bit_rgb_lut()
        pts = mlab.points3d(
            vox_pts[:, 0],
            vox_pts[:, 1],
            vox_pts[:, 2],
            scalars,
            scale_factor=point_size,
            scale_mode="none",
            resolution=20)
    else:
        scalars = np.array(list(G.nodes())) + 5
        pts = mlab.points3d(
            vox_pts[:, 0],
            vox_pts[:, 1],
            vox_pts[:, 2],
            scalars,
            scale_factor=point_size,
            scale_mode="none",
            colormap="Reds",
            resolution=20)

    pts.mlab_source.dataset.lines = np.array(list(G.edges()))
    tube = mlab.pipeline.tube(pts, tube_radius=edge_size)
    mlab.pipeline.surface(tube, color=(0.8, 0.8, 0.8))

    if vox_pts.shape[1] > 3:
        pts.module_manager.scalar_lut_manager.lut._vtk_obj.SetTableRange(0, rgb_lut.shape[0])
        pts.module_manager.scalar_lut_manager.lut.number_of_colors = rgb_lut.shape[0]
        pts.module_manager.scalar_lut_manager.lut.table = rgb_lut

    pts.glyph.scale_mode = 'data_scaling_off'

    xmin = np.min(vox_pts[:, 0])
    xmax = np.max(vox_pts[:, 0])
    ymin = np.min(vox_pts[:, 1])
    ymax = np.max(vox_pts[:, 1])
    zmin = np.min(vox_pts[:, 2])
    zmax = np.max(vox_pts[:, 2])
    pad = np.array(((xmax-xmin) * 0.01, (ymax-ymin) * 0.01, (zmax-zmin) * 0.01))
    mlab_plt_cube(xmin - pad[0], xmax + pad[0],
                  ymin - pad[1], ymax + pad[1],
                  zmin - pad[2], zmax - pad[2])


