import numpy as np
import vispy.visuals
import vispy.scene
from vispy.scene import visuals
import sys


# N = 2000
#import open3d as o3d

# pc = o3d.io.read_point_cloud('airplane.xyz')
#
# pc.scale(1 / np.max(pc.get_max_bound() - pc.get_min_bound()),
#          center=pc.get_center())
# pc.colors = o3d.utility.Vector3dVector(np.random.uniform(0, 1, size=(N, 3)))
# # o3d.visualization.draw_geometries([pc])
#
# print('voxelization')
# voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pc,
#                                                             voxel_size=0.05)
#
# print(type(voxel_grid))
# o3d.visualization.draw_geometries([voxel_grid])

class voxelize:
    def __init__(self, pc_array):
        self.points = pc_array
        self.N = pc_array.shape[0]
        self.bound_matrix = np.empty((3, 2))
        self.bound_buffer = 0.0
        self.get_bounds()
        self.draw_volume()

    def get_bounds(self):
        self.x_bounds = [np.min(self.points[:, 0]), np.max(self.points[:, 0])]
        self.y_bounds = [np.min(self.points[:, 1]), np.max(self.points[:, 1])]
        self.z_bounds = [np.min(self.points[:, 2]), np.max(self.points[:, 2])]
        self.x_length = self.x_bounds[1] - self.x_bounds[0]
        self.y_length = self.y_bounds[1] - self.y_bounds[0]
        self.z_length = self.z_bounds[1] - self.z_bounds[0]

    def draw_volume(self):
        canvas = vispy.scene.SceneCanvas(keys='interactive', show=True)
        view = canvas.central_widget.add_view()

        volume = vispy.visuals.BoxVisual(width=self.x_length, height=self.y_length, depth=self.z_length)
        view.add(volume)
        view.camera = 'turntable'  # or try 'arcball'

        # add a colored 3D axis for orientation
        axis = visuals.XYZAxis(parent=view.scene)

        if sys.flags.interactive != 1:
            vispy.app.run()

    # def get_voxel_count(self):
    #     vox_count_x =

pc_array = np.genfromtxt('airplane.pts',  delimiter='')
vox = voxelize(pc_array)