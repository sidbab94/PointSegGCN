import numpy as np
import math
from mayavi import mlab

def cube_faces(xmin, xmax, ymin, ymax, zmin, zmax):
    faces = []

    x,y = np.mgrid[xmin:xmax:3j,ymin:ymax:3j]
    z = np.ones(y.shape)*zmin
    faces.append((x,y,z))

    x,y = np.mgrid[xmin:xmax:3j,ymin:ymax:3j]
    z = np.ones(y.shape)*zmax
    faces.append((x,y,z))

    x,z = np.mgrid[xmin:xmax:3j,zmin:zmax:3j]
    y = np.ones(z.shape)*ymin
    faces.append((x,y,z))

    x,z = np.mgrid[xmin:xmax:3j,zmin:zmax:3j]
    y = np.ones(z.shape)*ymax
    faces.append((x,y,z))

    y,z = np.mgrid[ymin:ymax:3j,zmin:zmax:3j]
    x = np.ones(z.shape)*xmin
    faces.append((x,y,z))

    y,z = np.mgrid[ymin:ymax:3j,zmin:zmax:3j]
    x = np.ones(z.shape)*xmax
    faces.append((x,y,z))

    return faces

def mlab_plt_cube(xmin,xmax,ymin,ymax,zmin,zmax):
    faces = cube_faces(xmin,xmax,ymin,ymax,zmin,zmax)
    for grid in faces:
        x,y,z = grid
        mlab.mesh(x,y,z,opacity=0.4)

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
        self.avg_length = (self.x_length + self.y_length + self.z_length)/3
        self.bound_buffer = self.avg_length * 0.05

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
        self.x_length = int(round(self.x_length * 1000))
        self.y_length = int(round(self.y_length * 1000))
        self.z_length = int(round(self.z_length * 1000))
        self.voxel_l = math.gcd(self.x_length, math.gcd(self.y_length, self.z_length))
        self.x_voxels = int(self.x_length / self.voxel_l)
        self.y_voxels = int(self.y_length / self.voxel_l)
        self.z_voxels = int(self.z_length / self.voxel_l)
        self.n_voxels = int(self.x_voxels * self.y_voxels)
        self.n_voxels = int(self.n_voxels * self.z_voxels)
        self.voxel_l /= 1000
        print(self.x_voxels, self.y_voxels, self.z_voxels)





pc_array = np.genfromtxt('airplane.pts',  delimiter='')
vox = voxelize(pc_array)
vox.div_volume()
vox.plot_volume()