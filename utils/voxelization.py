import numpy as np
import math

def sample(points, thresh=1000):
    '''
    randomly samples {thresh} points from the provided sample
    :param points: input sample of points
    :param thresh: no. of points to sample from input
    :return: randomly sampled point cloud array
    '''
    return points[np.random.choice(points.shape[0], thresh, replace=False), :]

class voxelize:
    def __init__(self, pc_array, div_factor=15):
        self.pc = pc_array
        self.N = pc_array.shape[0]

        # Get axis bounds of the entire point cloud
        x_bounds = [np.min(self.pc[:, 0]), np.max(self.pc[:, 0])]
        y_bounds = [np.min(self.pc[:, 1]), np.max(self.pc[:, 1])]
        z_bounds = [np.min(self.pc[:, 2]), np.max(self.pc[:, 2])]

        # min and max values of x,y,z bounds
        x_min = x_bounds[0]
        x_max = x_bounds[1]
        y_min = y_bounds[0]
        y_max = y_bounds[1]
        z_min = z_bounds[0]
        z_max = z_bounds[1]

        # lengths along each axis
        x_length = x_max - x_min
        y_length = y_max - y_min
        z_length = z_max - z_min
        self.lengths = [x_length, y_length, z_length]

        # depending upon scale of point cloud, adjust visualization settings
        self.v_scaleup = 0
        self.order_of_mag()

        # assuming x is longest axis, calculate scaling factors of x wrt other axes --> y and z
        xy_factor = x_length / y_length
        xz_factor = x_length / z_length

        # Calculate voxel markers along each axis, based on input factor with which x axis is voxelized
        self.vox_X = np.linspace(start=x_min,
                                 stop=x_max,
                                 num=div_factor)
        # Calculate voxel length based on distance between voxel markers
        self.vox_xl = round(self.vox_X[1] - self.vox_X[0], 5)

        # Above process repeated for y and z, num_voxels determined separately ("num" parameter)
        self.vox_Y = np.linspace(start=y_min,
                                 stop=y_max,
                                 num=round(div_factor / xy_factor))
        self.vox_yl = round(self.vox_Y[1] - self.vox_Y[0], 5)
        self.vox_Z = np.linspace(start=z_min,
                                 stop=z_max,
                                 num=round(div_factor / xz_factor)+1)
        self.vox_zl = round(self.vox_Z[-1] - self.vox_Z[0], 5)

        # Initialize occupied voxels as empty array os shape (1, 6), each column corresponding to x/y/z min/max bounds
        self.occ_voxels = np.empty((1, 6))
        # Initialize list of points to be included in each 'occupied' voxel
        self.voxel_points = []
        # Initialize total point count, after sampling (see below)
        self.pcount = 0

    def get_voxels(self, occ_thresh=5e-3):
        '''
        Get voxels occupied by an optimal number of points
        :param occ_thresh: percentage of total point cloud size to consider as the lowest threshold for voxel occupancy
        '''
        # Looping through each axis voxel range
        for i in range(len(self.vox_X)):
            for j in range(len(self.vox_Y)):
                for k in range(len(self.vox_Z)):
                    # Current voxel as array, with min/max bounds of each axis as each element
                    voxel = np.array([self.vox_X[i], self.vox_X[i] + self.vox_xl,
                                      self.vox_Y[j], self.vox_Y[j] + self.vox_yl,
                                      self.vox_Z[k], self.vox_Z[k] + self.vox_zl])
                    # Get list of points of input point cloud within current voxel x bounds
                    x_occ = self.pc[(voxel[0] < self.pc[:, 0]) & (voxel[1] > self.pc[:, 0])]
                    # Check if points exist
                    if x_occ.size != 0:
                        # Get list of points of x-populated point cloud within current voxel y bounds
                        xy_occ = x_occ[(voxel[2] < x_occ[:, 1]) & (voxel[3] > x_occ[:, 1])]
                        # Check if points exist
                        if xy_occ.size != 0:
                            # Get list of points of xy-populated point cloud within current voxel z bounds
                            xyz_occ = xy_occ[(voxel[4] < xy_occ[:, 2]) & (voxel[5] > xy_occ[:, 2])]
                            # check point population thresholding: greater than small input %, smaller than 20000 points
                            if round(occ_thresh * self.N) <= xyz_occ.shape[0] <= 20000:
                                # add points to list
                                self.voxel_points.append(xyz_occ)
                                # update total point count
                                self.pcount += xyz_occ.shape[0]
                                # add current voxel to array of occupied voxels
                                self.occ_voxels = np.concatenate((self.occ_voxels, np.array([voxel])), axis=0)
                            # if point population greater than 20000 (optimal count)
                            elif xyz_occ.shape[0] > 20000:
                                # random sampling
                                downsampled = sample(xyz_occ, thresh=20000)
                                self.voxel_points.append(downsampled)
                                self.pcount += downsampled.shape[0]
                                self.occ_voxels = np.concatenate((self.occ_voxels, np.array([voxel])), axis=0)
        # delete empty row of occupied voxel array
        self.occ_voxels = np.delete(self.occ_voxels, 0, 0)

    def order_of_mag(self):
        '''
        Determine graph visualization setting boolean flag
        :return: visualization scaling flag,
                if 0 (default) --> smaller point cloud (ShapeNet scale) if 1 --> large (SemanticKITTI scale)
        '''
        o = np.array([math.floor(math.log10(length)) for length in self.lengths]) >= 1
        if np.any(o):
            self.v_scaleup = 1
