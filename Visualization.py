import numpy as np
import vispy.scene
from vispy.scene import visuals
from mayavi import mlab
import sys
from numpy import genfromtxt

def visualize(pointcloud_array=[], path=''):
    canvas = vispy.scene.SceneCanvas(keys='interactive', show=True)
    view = canvas.central_widget.add_view()

    if not path:
        points = pointcloud_array
    else:
        points = np.fromfile(path, dtype=np.float32).reshape(-1, 4)
    # point cloud format --> {X, Y, Z, R, G, B, L}
    if points.shape[1] == 7:
        pc_colors = points[:, 3:-1] / 255
        pc_colorsA = (np.ones((pc_colors.shape[0], 4)) * .5)
        pc_colorsA[:, :-1] = pc_colors
        vis_RGB = tuple(tuple(row) for row in pc_colorsA)
    else:
        vis_RGB = (1, 1, 1, .5)

    # create scatter object and fill in the data
    scatter = visuals.Markers()
    scatter.set_data(points[:, :3], edge_color=None, face_color=vis_RGB, size=5)

    view.add(scatter)
    view.camera = 'turntable'  # or try 'arcball'

    # add a colored 3D axis for orientation
    axis = visuals.XYZAxis(parent=view.scene)

    if sys.flags.interactive != 1:
        vispy.app.run()

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

def visualize_myvi(array):
    xyz = array[:, :3]
    rgb = array[:, 3:]
    scalars = np.zeros((rgb.shape[0], ))
    for (kp_idx, kp_c) in enumerate(rgb):
        scalars[kp_idx] = rgb_2_scalar_idx(kp_c[0], kp_c[1], kp_c[2])

    rgb_lut = create_8bit_rgb_lut()
    pts = mlab.points3d(
        xyz[:, 0],
        xyz[:, 1],
        xyz[:, 2],
        scalars,
        scale_factor=0.2,
        scale_mode="none",
        resolution=20,
    )
    pts.module_manager.scalar_lut_manager.lut._vtk_obj.SetTableRange(0, rgb_lut.shape[0])
    pts.module_manager.scalar_lut_manager.lut.number_of_colors = rgb_lut.shape[0]
    pts.module_manager.scalar_lut_manager.lut.table = rgb_lut
    pts.glyph.scale_mode = 'data_scaling_off'
    mlab.show()

file = 'vkitti3d_01.npy'
# points = genfromtxt(file, delimiter='')
points = np.load(file)
visualize(points)

