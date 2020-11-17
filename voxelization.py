import numpy as np

class voxelize:
    def __init__(self, pc_array, div_factor=15):
        self.pc = pc_array
        self.N = pc_array.shape[0]

        self.x_bounds = [np.min(self.pc[:, 0]), np.max(self.pc[:, 0])]
        self.y_bounds = [np.min(self.pc[:, 1]), np.max(self.pc[:, 1])]
        self.z_bounds = [np.min(self.pc[:, 2]), np.max(self.pc[:, 2])]
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
        self.vox_zl = round(self.vox_zn[-1] - self.vox_zn[0], 5)

    def get_voxels(self, occ_thresh=5e-3):
        self.all_voxels = np.empty((1, 6))
        for i in range(len(self.vox_xn)):
            for j in range(len(self.vox_yn)):
                for k in range(len(self.vox_zn)):
                    curr_bounds = np.array([[self.vox_xn[i], self.vox_xn[i] + self.vox_xl,
                                             self.vox_yn[j], self.vox_yn[j] + self.vox_yl,
                                             self.vox_zn[k], self.vox_zn[k] + self.vox_zl]])
                    self.all_voxels = np.concatenate((self.all_voxels, curr_bounds), axis=0)

        self.occ_voxels = np.empty((1, 6))
        self.voxel_points = []

        for voxel in self.all_voxels:
            x_occ = self.pc[(voxel[0] <= self.pc[:, 0]) & (voxel[1] > self.pc[:, 0])]
            if x_occ.size != 0:
                xy_occ = x_occ[(voxel[2] <= x_occ[:, 1]) & (voxel[3] > x_occ[:, 1])]
                if xy_occ.size != 0:
                    xyz_occ = xy_occ[(voxel[4] <= xy_occ[:, 2]) & (voxel[5] > xy_occ[:, 2])]
                    if xyz_occ.shape[0] >= round(occ_thresh * self.N):
                        self.voxel_points.append(xyz_occ)
                        self.occ_voxels = np.concatenate((self.occ_voxels, np.array([voxel])), axis=0)
        self.occ_voxels = np.delete(self.occ_voxels, 0, 0)

