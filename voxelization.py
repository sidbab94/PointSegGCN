import numpy as np
from time import time
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
        mlab.mesh(x, y, z, opacity=0.4, color=(0.3, 0.2, 0.2))


class voxelize:
    def __init__(self, pc_array, div_factor=15):
        self.xyz = pc_array[:, :3]
        self.N = pc_array.shape[0]
        print(self.N)

        self.x_bounds = [np.min(self.xyz[:, 0]), np.max(self.xyz[:, 0])]
        self.y_bounds = [np.min(self.xyz[:, 1]), np.max(self.xyz[:, 1])]
        self.z_bounds = [np.min(self.xyz[:, 2]), np.max(self.xyz[:, 2])]
        self.x_min = self.x_bounds[0]
        self.x_max = self.x_bounds[1]
        self.y_min = self.y_bounds[0]
        self.y_max = self.y_bounds[1]
        self.z_min = self.z_bounds[0]
        self.z_max = self.z_bounds[1]

        self.x_length = self.x_max - self.x_min
        self.y_length = self.y_max - self.y_min
        self.z_length = self.z_max - self.z_min
        self.xy_factor = self.x_length / self.y_length
        self.xz_factor = self.x_length / self.z_length
        self.avg_length = (self.x_length + self.y_length + self.z_length) / 3
        self.bound_buffer = 0.0  # self.avg_length * 0.01

        self.vox_xn = np.linspace(start=self.x_min - self.bound_buffer,
                                  stop=self.x_max + self.bound_buffer,
                                  num=div_factor)
        self.vox_xl = round(self.vox_xn[1] - self.vox_xn[0], 5)
        self.vox_yn = np.linspace(start=self.y_min - self.bound_buffer,
                                  stop=self.y_max + self.bound_buffer,
                                  num=round(div_factor / self.xy_factor))
        self.vox_yl = round(self.vox_yn[1] - self.vox_yn[0], 5)
        self.vox_zn = np.linspace(start=self.z_min - self.bound_buffer,
                                  stop=self.z_max + self.bound_buffer,
                                  num=round(div_factor / self.xz_factor))
        self.vox_zl = round(self.vox_zn[1] - self.vox_zn[0], 5)

    def get_voxels(self):
        self.all_voxels = np.empty((1, 6))
        for i in range(len(self.vox_xn)):
            for j in range(len(self.vox_yn)):
                for k in range(len(self.vox_zn)):
                    curr_bounds = np.array([[self.vox_xn[i], self.vox_xn[i] + self.vox_xl,
                                             self.vox_yn[j], self.vox_yn[j] + self.vox_yl,
                                             self.vox_zn[k], self.vox_zn[k] + self.vox_zl]])
                    self.all_voxels = np.concatenate((self.all_voxels, curr_bounds), axis=0)

        self.occ_voxels = np.empty((1, 6))
        for voxel in self.all_voxels:
            x_occ = self.xyz[(voxel[0] <= self.xyz[:, 0]) & (voxel[1] > self.xyz[:, 0])]
            if x_occ.size != 0:
                xy_occ = x_occ[(voxel[2] <= x_occ[:, 1]) & (voxel[3] > x_occ[:, 1])]
                if xy_occ.size != 0:
                    xyz_occ = xy_occ[(voxel[4] <= xy_occ[:, 2]) & (voxel[5] > xy_occ[:, 2])]
                    if xyz_occ.shape[0] >= round(6e-3 * self.N):
                        self.occ_voxels = np.concatenate((self.occ_voxels, np.array([voxel])), axis=0)
        self.occ_voxels = np.delete(self.occ_voxels, 0, 0)

    def vox_overlay_vis(self):
        for voxel in self.occ_voxels:  # [np.random.choice(self.occ_voxels.shape[0], 10, replace=False), :]:
            mlab_plt_cube(voxel[0], voxel[1], voxel[2], voxel[3], voxel[4], voxel[5])
        pts = mlab.points3d(
            self.xyz[:, 0],
            self.xyz[:, 1],
            self.xyz[:, 2],
            scale_factor=0.003,
            scale_mode="none",
            colormap="Blues",
            resolution=20)
        mlab.show()


pc_array = np.genfromtxt('airplane.pts', delimiter='')
start = time()
vox = voxelize(pc_array, div_factor=15)
vox.get_voxels()
print('Voxelization -- Time elapsed in seconds: ', round(time()-start, 5))
vox.vox_overlay_vis()
