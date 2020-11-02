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

    # create scatter object and fill in the data
    scatter = visuals.Markers()
    scatter.set_data(points[:, :3], edge_color=None, face_color=(1, 1, 1, .5), size=5)

    view.add(scatter)
    view.camera = 'turntable'  # or try 'arcball'

    # add a colored 3D axis for orientation
    axis = visuals.XYZAxis(parent=view.scene)

    if sys.flags.interactive != 1:
        vispy.app.run()

file = 'data/train_data/02691156/000001.pts'
points = genfromtxt(file, delimiter='')
visualize(points)

