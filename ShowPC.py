import numpy as np
import vispy.scene
from vispy.scene import visuals
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



file = 'vkitti3d_01.npy'
# points = genfromtxt(file, delimiter='')
points = np.load(file)
visualize(points)

