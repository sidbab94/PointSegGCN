import numpy as np
import math
from mayavi import mlab


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
        mlab.mesh(x, y, z, opacity=0.4)


class voxelize:
    def __init__(self, pc_array):
        self.points = pc_array
        self.N = pc_array.shape[0]
        self.bound_matrix = np.empty((3, 2))
        self.x_bounds = [np.min(self.points[:, 0]), np.max(self.points[:, 0])]
        self.y_bounds = [np.min(self.points[:, 1]), np.max(self.points[:, 1])]
        self.z_bounds = [np.min(self.points[:, 2]), np.max(self.points[:, 2])]
        self.x_min = self.x_bounds[0]
        self.x_max = self.x_bounds[1]
        self.y_min = self.y_bounds[0]
        self.y_max = self.y_bounds[1]
        self.z_min = self.z_bounds[0]
        self.z_max = self.z_bounds[1]
        self.x_length = self.x_bounds[1] - self.x_bounds[0]
        self.y_length = self.y_bounds[1] - self.y_bounds[0]
        self.z_length = self.z_bounds[1] - self.z_bounds[0]
        self.avg_length = (self.x_length + self.y_length + self.z_length) / 3
        self.bound_buffer = 0.0
        self.x_start = self.x_min - self.bound_buffer
        self.x_end = self.x_max + self.bound_buffer

    def plot_volume(self):
        pts = mlab.points3d(
            self.points[:, 0],
            self.points[:, 1],
            self.points[:, 2],
            scale_factor=0.003,
            scale_mode="none",
            colormap="Blues",
            resolution=20,
        )
        faces = cube_faces(self.x_min - self.bound_buffer, self.x_max + self.bound_buffer,
                           self.y_min - self.bound_buffer, self.y_max + self.bound_buffer,
                           self.z_min - self.bound_buffer, self.z_max + self.bound_buffer)
        for grid in faces:
            x, y, z = grid
            cube = mlab.mesh(x, y, z, opacity=0.4)
        mlab.show()

    def div_volume(self):
        # self.x_length = int(round(self.x_length * 1000))
        # self.y_length = int(round(self.y_length * 1000))
        # self.z_length = int(round(self.z_length * 1000))
        # self.voxel_l = math.gcd(self.x_length, math.gcd(self.y_length, self.z_length))
        # self.x_voxels = self.x_length / self.voxel_l
        # self.y_voxels = self.y_length / self.voxel_l
        # self.z_voxels = self.z_length / self.voxel_l
        # self.n_voxels = self.x_voxels * self.y_voxels
        # self.n_voxels = self.n_voxels * self.z_voxels
        # print(self.voxel_l)
        # self.voxel_l /= 1000
        # print(self.x_voxels, self.y_voxels, self.z_voxels)

        self.vox_xn = np.linspace(start=self.x_min - self.bound_buffer,
                                  stop=self.x_max + self.bound_buffer,
                                  num=15)
        self.vox_xl = round(self.vox_xn[1] - self.vox_xn[0], 5)
        self.vox_yn = np.linspace(start=self.y_min - self.bound_buffer,
                                  stop=self.y_max + self.bound_buffer,
                                  num=15)
        self.vox_yl = round(self.vox_yn[1] - self.vox_yn[0], 5)
        self.vox_zn = np.linspace(start=self.z_min - self.bound_buffer,
                                  stop=self.z_max + self.bound_buffer,
                                  num=15)
        self.vox_zl = round(self.vox_zn[1] - self.vox_zn[0], 5)

    def get_voxels(self):

        self.voxels = np.empty((1, 6))
        for i in range(len(self.vox_xn)):
            for j in range(len(self.vox_yn)):
                for k in range(len(self.vox_zn)):
                    curr_bounds = np.array([[self.vox_xn[i], self.vox_xn[i] + self.vox_xl,
                                             self.vox_yn[j], self.vox_yn[j] + self.vox_yl,
                                             self.vox_zn[k], self.vox_zn[k] + self.vox_zl]])
                    self.voxels = np.concatenate((self.voxels, curr_bounds), axis=0)

        for voxel in self.voxels[np.random.choice(self.voxels.shape[0], 10, replace=False), :]:
            print('Current voxel bounds: ', voxel)
            for point in self.points[:, :3]:
                if ((voxel[0] <= point[0] < voxel[1]) and (voxel[2] <= point[1] < voxel[3]) and (
                        voxel[4] <= point[2] < voxel[5])):
                    faces = cube_faces(voxel[0], voxel[1], voxel[2], voxel[3], voxel[4], voxel[5])
                    print('Point(s) detected in voxel')
                    pts = mlab.points3d(
                        self.points[:, 0],
                        self.points[:, 1],
                        self.points[:, 2],
                        scale_factor=0.003,
                        scale_mode="none",
                        colormap="Blues",
                        resolution=20,
                    )
                    for grid in faces:
                        x, y, z = grid
                        mlab.mesh(x, y, z, opacity=0.4, color=(0.3, 0.2, 0.2))
                    mlab.show()


pc_array = np.genfromtxt('airplane.pts', delimiter='')
vox = voxelize(pc_array)
vox.div_volume()
vox.get_voxels()
# vox.plot_volume()
