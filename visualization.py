import numpy as np
from mayavi import mlab
import networkx as nx

# import vispy.scene
# from vispy.scene import visuals
# import sys

# def vispy_show(pointcloud_array=[], path=''):
#     canvas = vispy.scene.SceneCanvas(keys='interactive', show=True)
#     view = canvas.central_widget.add_view()
#
#     if not path:
#         points = pointcloud_array
#     else:
#         points = np.fromfile(path, dtype=np.float32).reshape(-1, 4)
#     # point cloud format --> {X, Y, Z, R, G, B, L}
#     if points.shape[1] == 7:
#         pc_colors = points[:, 3:-1] / 255
#         pc_colorsA = (np.ones((pc_colors.shape[0], 4)) * .5)
#         pc_colorsA[:, :-1] = pc_colors
#         vis_RGB = tuple(tuple(row) for row in pc_colorsA)
#     else:
#         vis_RGB = (1, 1, 1, .5)
#
#     # create scatter object and fill in the data
#     scatter = visuals.Markers()
#     scatter.set_data(points[:, :3], edge_color=None, face_color=vis_RGB, size=5)
#
#     view.add(scatter)
#     view.camera = 'turntable'  # or try 'arcball'
#
#     # add a colored 3D axis for orientation
#     axis = visuals.XYZAxis(parent=view.scene)
#
#     if sys.flags.interactive != 1:
#         vispy.app.run()

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

def show_voxel(vox_pts, graph):
    # mlab.figure(size=(1200, 800), bgcolor=(1, 1, 1))

    G = nx.convert_node_labels_to_integers(graph)
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
        scale_factor=0.2,
        scale_mode="none",
        resolution=20)

    pts.mlab_source.dataset.lines = np.array(list(G.edges()))
    tube = mlab.pipeline.tube(pts, tube_radius=0.02)
    mlab.pipeline.surface(tube, color=(0.8, 0.8, 0.8))

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

    # mlab.show()

def visualize_graph_xyz(graph, array):
    mlab.figure(size=(1200, 800), bgcolor=(0, 0, 0))
    G = nx.convert_node_labels_to_integers(graph)
    xyz = array
    scalars = np.array(list(G.nodes())) + 5
    pts = mlab.points3d(
        xyz[:, 0],
        xyz[:, 1],
        xyz[:, 2],
        scalars,
        scale_factor=0.3,
        scale_mode="none",
        colormap="Reds",
        resolution=20,
    )

    pts.mlab_source.dataset.lines = np.array(list(G.edges()))
    tube = mlab.pipeline.tube(pts, tube_radius=0.08)
    mlab.pipeline.surface(tube, color=(0.8, 0.8, 0.8))
    pts.glyph.glyph.clamping = False
    pts.glyph.scale_mode = 'data_scaling_off'
    mlab.show()

