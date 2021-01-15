import numpy as np

class SensorFusion:
    """
    Fuses LiDAR geometric information with RGB modality, acquired from corresponding scan images.

    Arguments:
        pc: 3D point cloud array with XYZ information
        labels: point-wise ground truth label array
        calib: calibration parameters in dictionary format
        img: RGB image array read from corresponding file path
    """

    def __init__(self, pc, labels, calib, img):

        self.pc = pc
        self.labels = labels
        self.calib = calib
        self.img = img

    def render_lidar_rgb(self):
        '''
        Renders RGB information on the velodyne point cloud, as follows:
            --> Compute projection matrix of velodyne to camera, from the camera intrinsic and extrinsic parameters
            --> Perform homogeneous transformation of point cloud with projection matrix, to obtain points
                in camera coordinate system
            --> Obtain the point cloud indices corresponding to Camera FOV
            --> Map projected points to pixel (RGB) values
            --> Crop original 3D point cloud to retain points within Camera FOV and mask label array with new indices
            --> Augment modified point cloud with RGB information

        :return: RGB-augmented point cloud and label arrays
        '''

        img_height, img_width, _ = self.img.shape

        # projection matrix (project from velo2cam2)
        self.proj_mat = self.project_velo_to_cam2()

        # apply projection
        try:
            pts_2d = self.project_to_image()

            # Filter lidar points to be within image FOV
            inds = np.where((pts_2d[0, :] < img_width) & (pts_2d[0, :] >= 0) &
                            (pts_2d[1, :] < img_height) & (pts_2d[1, :] >= 0) &
                            (self.pc[:, 0] > 0)
                            )[0]

            # get 2d points corresponding to camera FOV and camera coordinates
            imgfov_pc_pixel = pts_2d[:, inds].transpose()

            # get rgb array for projected 2d points based on nearest-pixel values on orignal image
            color_array_pts2d = np.zeros((imgfov_pc_pixel.shape[0], 3), dtype='float64')
            for i in range(imgfov_pc_pixel.shape[0]):
                y_coord, x_coord = imgfov_pc_pixel[i]
                val_x, val_y = int(round(x_coord)) - 1, int(round(y_coord)) - 1
                color_array_pts2d[i] = self.img[val_x, val_y] / 255

            # get lidar points corresponding to camera FOV and velodyne coordinates, color with mapped rgb values
            imgfov_pc_velo = self.pc[inds, :]

            pc_rgb = np.zeros((imgfov_pc_velo.shape[0], 6), dtype='float64')
            pc_rgb[:, :3] = imgfov_pc_velo
            pc_rgb[:, 3:] = color_array_pts2d

            labels_rgb = self.labels[inds]

        except MemoryError:
            # Projection computation may induce memory exceptions
            return None, None

        return pc_rgb, labels_rgb

    def project_velo_to_cam2(self):
        '''
        Computes projection matrix by means of homogenous transformations.
        Calibration parameters 'Tr' (transformation matrix from velodyne to camera) and 'P2' (intrinsic camera
        parameters) invoked to perform computation
        :return: projection matrix velodyne --> camera
        '''

        P_velo2cam_ref = np.vstack((self.calib['Tr'].reshape(3, 4), np.array([0., 0., 0., 1.])))  # velo2ref_cam
        R_ref2rect = np.eye(4)

        P_rect2cam2 = self.calib['P2'].reshape((3, 4))
        proj_mat = P_rect2cam2 @ R_ref2rect @ P_velo2cam_ref

        return proj_mat

    def project_to_image(self):
        '''
        Projects 3D points to a 2D space corresponding to the camera coordinate system.
        Uses already computed projection matrix.
        :return:
        '''

        points = self.pc.transpose()
        num_pts = points.shape[1]

        # Change to homogenous coordinate
        points = np.vstack((points, np.ones((1, num_pts))))

        points = self.proj_mat @ points
        points[:2, :] /= points[2, :]

        return points[:2, :]


if __name__ == '__main__':

    from preproc_utils.readers import *
    from preprocess import Preprocess
    from visualization import PC_Vis
    import random

    BASE_DIR = 'D:/SemanticKITTI/dataset/sequences'

    model_cfg = get_cfg_params(BASE_DIR, dataset_cfg='../config/semantic-kitti.yaml',
                               train_cfg='../config/tr_config.yml')

    train_files, val_files, _ = get_split_files(dataset_path=BASE_DIR, cfg=model_cfg, shuffle=False)

    prep = Preprocess(model_cfg)

    _, _, _ = prep.assess_scan(random.choice(train_files))

    pc, labels, calib, img = prep.pc, prep.labels, prep.calib, prep.img

    fusion = SensorFusion(pc, labels, calib, img)
    pc_rgb, labels_rgb = fusion.render_lidar_rgb()

    PC_Vis.draw_pc(pc_rgb, vis_test=True)

