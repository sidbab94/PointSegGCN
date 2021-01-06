import numpy as np
import os
import cv2
from yaml import safe_load
import matplotlib.pyplot as plt

from preproc_utils.dataprep import read_bin_velodyne
from visualization import ShowPC

def project_velo_to_cam2(calib):
    P_velo2cam_ref = np.vstack((calib['Tr'].reshape(3, 4), np.array([0., 0., 0., 1.])))  # velo2ref_cam
    R_ref2rect = np.eye(4)
    # R0_rect = calib['R0_rect'].reshape(3, 3)  # ref_cam2rect
    # R_ref2rect[:3, :3] = R0_rect
    P_rect2cam2 = calib['P2'].reshape((3, 4))
    proj_mat = P_rect2cam2 @ R_ref2rect @ P_velo2cam_ref
    return proj_mat

def project_cam2_to_velo(calib):
    # inverse rigid transformation
    velo2cam_ref = np.vstack((calib['Tr_velo_to_cam'].reshape(3, 4), np.array([0., 0., 0., 1.])))  # velo2ref_cam
    P_cam_ref2velo = np.linalg.inv(velo2cam_ref)


    R_ref2rect = np.eye(4)
    ## unavailable
    # R0_rect = calib['R0_rect'].reshape(3, 3)  # ref_cam2rect
    # R_ref2rect[:3, :3] = R0_rect
    R_ref2rect_inv = np.linalg.inv(R_ref2rect)  # rect2ref_cam

    proj_mat = R_ref2rect_inv @ P_cam_ref2velo
    return proj_mat

def project_to_image(points, proj_mat):
    """
    Apply the perspective projection
    Args:
        pts_3d:     3D points in camera coordinate [3, npoints]
        proj_mat:   Projection matrix [3, 4]
    """
    num_pts = points.shape[1]

    # Change to homogenous coordinate
    points = np.vstack((points, np.ones((1, num_pts))))
    points = proj_mat @ points
    points[:2, :] /= points[2, :]
    return points[:2, :]

def read_calib_file(filepath):
    """
    Read in a calibration file and parse into a dictionary.
    Ref: https://github.com/utiasSTARS/pykitti/blob/master/pykitti/utils.py
    """
    data = {}
    with open(filepath, 'r') as f:
        for line in f.readlines():
            line = line.rstrip()
            if len(line) == 0: continue
            key, value = line.split(':', 1)
            # The only non-float values in these files are dates, which
            # we don't care about anyway
            try:
                data[key] = np.array([float(x) for x in value.split()])
            except ValueError:
                pass

    return data

def render_lidar_on_image(pts_velo, img, calib, img_width, img_height):
    # projection matrix (project from velo2cam2)
    proj_velo2cam2 = project_velo_to_cam2(calib)

    # apply projection
    pts_2d = project_to_image(pts_velo.transpose(), proj_velo2cam2)

    # Filter lidar points to be within image FOV
    inds = np.where((pts_2d[0, :] < img_width) & (pts_2d[0, :] >= 0) &
                    (pts_2d[1, :] < img_height) & (pts_2d[1, :] >= 0) &
                    (pts_velo[:, 0] > 0)
                    )[0]

    # Filter out pixels points
    imgfov_pc_pixel = pts_2d[:, inds]

    # Retrieve depth from lidar
    imgfov_pc_velo = pts_velo[inds, :]
    imgfov_pc_velo = np.hstack((imgfov_pc_velo, np.ones((imgfov_pc_velo.shape[0], 1))))
    imgfov_pc_cam2 = proj_velo2cam2 @ imgfov_pc_velo.transpose()

    cmap = plt.cm.get_cmap('hsv', 256)
    cmap = np.array([cmap(i) for i in range(256)])[:, :3] * 255

    for i in range(imgfov_pc_pixel.shape[1]):
        depth = imgfov_pc_cam2[2, i]
        color = cmap[int(640.0 / depth), :]
        cv2.circle(img, (int(np.round(imgfov_pc_pixel[0, i])),
                         int(np.round(imgfov_pc_pixel[1, i]))),
                   2, color=tuple(color), thickness=-1)
    plt.imshow(img)
    plt.yticks([])
    plt.xticks([])
    plt.show()
    return img

def lidar_camfov_map(pc_velo, calib, img, vis=False):

    img_height, img_width, _ = img.shape

    # projection matrix (project from velo2cam2)
    proj_velo2cam2 = project_velo_to_cam2(calib)

    # apply projection
    pts_2d = project_to_image(pc_velo.transpose(), proj_velo2cam2)

    # Filter lidar points to be within image FOV
    inds = np.where((pts_2d[0, :] < img_width) & (pts_2d[0, :] >= 0) &
                    (pts_2d[1, :] < img_height) & (pts_2d[1, :] >= 0) &
                    (pc_velo[:, 0] > 0)
                    )[0]

    # get 2d points corresponding to camera FOV and camera coordinates
    imgfov_pc_pixel = pts_2d[:, inds].transpose()

    # get rgb array for projected 2d points based on nearest-pixel values on orignal image
    color_array_pts2d = np.zeros((imgfov_pc_pixel.shape[0], 3), dtype='float64')
    for i in range(imgfov_pc_pixel.shape[0]):
        y_coord, x_coord = imgfov_pc_pixel[i]
        val_x, val_y = int(round(x_coord))-1, int(round(y_coord))-1
        color_array_pts2d[i] = img[val_x, val_y]/255

    # get lidar points corresponding to camera FOV and velodyne coordinates, color with mapped rgb values
    imgfov_pc_velo = pc_velo[inds, :]
    pc_cam_rgb = np.zeros((imgfov_pc_velo.shape[0], 6), dtype='float64')
    pc_cam_rgb[:, :3] = imgfov_pc_velo
    pc_cam_rgb[:, 3:] = color_array_pts2d

    # modify original scan point cloud feature array --> XYZ + RGB
    mod_pc_velo = np.zeros((pc_velo.shape[0], 6))
    mod_pc_velo[:, :3] = pc_velo[:, :3]
    mod_pc_velo[:, 3:] = np.full((mod_pc_velo.shape[0], 3), fill_value=211/255, dtype='float64')  # unmapped -> gray
    mod_pc_velo[inds, 3:] = color_array_pts2d

    if vis:
        # o3d_vis(imgfov_pc_velo, color_array_pts2d, lidar_to_camera=True)
        ShowPC.draw_pc(mod_pc_velo)
        ShowPC.draw_pc(pc_cam_rgb)

if __name__ == '__main__':
    BASE_DIR = 'D:/SemanticKITTI/dataset/sequences'
    cfg = safe_load(open('config/semantic-kitti.yaml', 'r'))
    seq_list = np.sort(os.listdir(BASE_DIR))

    seq_no = 0

    for id in seq_list:
        if int(id) == seq_no:
            seq_no = id

    velo_file = os.path.join(BASE_DIR, seq_no, 'velodyne', '000000.bin')
    label_file = os.path.join(BASE_DIR, seq_no, 'labels', '000000.label')
    calib_file = os.path.join(BASE_DIR, seq_no, 'calib.txt')
    img_file = os.path.join(BASE_DIR, seq_no, 'image_2', '000000.png')

    pc_velo = read_bin_velodyne(velo_file)
    calib = read_calib_file(calib_file)
    rgb = cv2.cvtColor(cv2.imread(img_file), cv2.COLOR_BGR2RGB)
    img_height, img_width, img_channel = rgb.shape

    # render_lidar_on_image(pc_velo, rgb, calib, img_width, img_height)
    lidar_camfov_map(pc_velo, calib, rgb)
