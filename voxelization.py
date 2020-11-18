import numpy as np
import math

def sample(points, thresh=1000):
    return points[np.random.choice(points.shape[0], thresh, replace=False), :]

class voxelize:
    def __init__(self, pc_array, div_factor=15):
        self.pc = pc_array
        self.N = pc_array.shape[0]

        x_bounds = [np.min(self.pc[:, 0]), np.max(self.pc[:, 0])]
        y_bounds = [np.min(self.pc[:, 1]), np.max(self.pc[:, 1])]
        z_bounds = [np.min(self.pc[:, 2]), np.max(self.pc[:, 2])]
        x_min = x_bounds[0]
        x_max = x_bounds[1]
        y_min = y_bounds[0]
        y_max = y_bounds[1]
        z_min = z_bounds[0]
        z_max = z_bounds[1]

        x_length = x_max - x_min
        y_length = y_max - y_min
        z_length = z_max - z_min
        self.lengths = [x_length, y_length, z_length]
        self.v_scaleup = 0
        self.order_of_mag()

        xy_factor = x_length / y_length
        xz_factor = x_length / z_length

        self.vox_X = np.linspace(start=x_min,
                                 stop=x_max,
                                 num=div_factor)
        self.vox_xl = round(self.vox_X[1] - self.vox_X[0], 5)
        self.vox_Y = np.linspace(start=y_min,
                                 stop=y_max,
                                 num=round(div_factor / xy_factor))
        self.vox_yl = round(self.vox_Y[1] - self.vox_Y[0], 5)
        self.vox_Z = np.linspace(start=z_min,
                                 stop=z_max,
                                 num=round(div_factor / xz_factor))
        self.vox_zl = round(self.vox_Z[-1] - self.vox_Z[0], 5)

        self.occ_voxels = np.empty((1, 6))
        self.voxel_points = []

    def get_voxels(self, occ_thresh=5e-3):
        for i in range(len(self.vox_X)):
            for j in range(len(self.vox_Y)):
                for k in range(len(self.vox_Z)):
                    voxel = np.array([self.vox_X[i], self.vox_X[i] + self.vox_xl,
                                      self.vox_Y[j], self.vox_Y[j] + self.vox_yl,
                                      self.vox_Z[k], self.vox_Z[k] + self.vox_zl])
                    x_occ = self.pc[(voxel[0] < self.pc[:, 0]) & (voxel[1] > self.pc[:, 0])]
                    if x_occ.size != 0:
                        xy_occ = x_occ[(voxel[2] < x_occ[:, 1]) & (voxel[3] > x_occ[:, 1])]
                        if xy_occ.size != 0:
                            xyz_occ = xy_occ[(voxel[4] < xy_occ[:, 2]) & (voxel[5] > xy_occ[:, 2])]
                            if round(occ_thresh * self.N) <= xyz_occ.shape[0] <= 1000:
                                self.voxel_points.append(xyz_occ)
                                self.occ_voxels = np.concatenate((self.occ_voxels, np.array([voxel])), axis=0)
                            elif xyz_occ.shape[0] > 1000:
                                self.voxel_points.append(sample(xyz_occ))
                                self.occ_voxels = np.concatenate((self.occ_voxels, np.array([voxel])), axis=0)
        self.occ_voxels = np.delete(self.occ_voxels, 0, 0)

    def order_of_mag(self):
        o = np.array([math.floor(math.log10(length)) for length in self.lengths]) >= 1
        if np.any(o):
            self.v_scaleup = 1
