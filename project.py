import numpy as np
import os
import cv2
from yaml import safe_load
import matplotlib.pyplot as plt


from preproc_utils.dataprep import get_labels, read_bin_velodyne
from vis_aux.visualize import ScanVis, o3d_vis

def inverse_transform(Tr):
    rot_Tr = Tr[0:3, 0:3]
    tra_Tr = Tr[0:3, 3]
    rot_Tr_inv = rot_Tr.transpose()
    tra_Tr_inv = rot_Tr_inv.dot(tra_Tr)
    Tr_inv = Tr
    Tr_inv[0:3, 0:3] = rot_Tr_inv
    Tr_inv[0:3, 3] = tra_Tr_inv
    return Tr_inv


def project_velo_to_cam2(calib):
    P_velo2cam_ref = np.vstack((calib['Tr'].reshape(3, 4), np.array([0., 0., 0., 1.])))  # velo2ref_cam
    P_rect2cam2 = calib['P2'].reshape((3, 4))
    proj_mat = P_rect2cam2 @ P_velo2cam_ref
    return proj_mat

def project_cam2_to_velo(calib):
    R_ref2rect = np.eye(4)
    R0_rect = calib['R0_rect'].reshape(3, 3)  # ref_cam2rect
    R_ref2rect[:3, :3] = R0_rect
    R_ref2rect_inv = np.linalg.inv(R_ref2rect)  # rect2ref_cam

    # inverse rigid transformation
    velo2cam_ref = np.vstack((calib['Tr_velo_to_cam'].reshape(3, 4), np.array([0., 0., 0., 1.])))  # velo2ref_cam
    P_cam_ref2velo = np.linalg.inv(velo2cam_ref)

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

if __name__ == '__main__':
    BASE_DIR = 'D:/SemanticKITTI/dataset/sequences'
    cfg = safe_load(open('config/semantic-kitti.yaml', 'r'))

    velo_file = os.path.join(BASE_DIR, '00', 'velodyne', '000000.bin')
    label_file = os.path.join(BASE_DIR, '00', 'labels', '000000.label')
    calib_file = os.path.join(BASE_DIR, '00', 'calib.txt')
    img_file = os.path.join(BASE_DIR, '00', 'image_2', '000000.png')

    pc_velo = read_bin_velodyne(velo_file)
    calib = read_calib_file(calib_file)
    rgb = cv2.cvtColor(cv2.imread(img_file), cv2.COLOR_BGR2RGB)
    img_height, img_width, img_channel = rgb.shape

    render_lidar_on_image(pc_velo, rgb, calib, img_width, img_height)





# P0, P1, P2, P3, Tr = np.genfromtxt(calib_file, delimiter='')[:, 1:]
#
# P0 = np.reshape(P0, (3, 4))
# Rot = np.reshape(Tr, (3, 4))
# Rot = np.vstack((Rot, np.array([0, 0, 0, 1])))
#
# # sample K
# f_x = f_y = P0[0]
# c_x, c_y = P0[2], P0[1, 2]
# K = (f_x, f_y, c_x, c_y)
#
# img_file = os.path.join(BASE_DIR, '00', 'image_2', '000000.png')
# im = Image.open(img_file, 'r')
# width, height = im.size
# pixel_values = im.getdata()
# pixel_values = np.array(pixel_values)
#
# open3d = o3d_vis(velo_file, pc_labels, Rot=Rot, imsize=(width, height), K=K)

