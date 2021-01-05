import cv2
import open3d as o3d
import matplotlib.pyplot as plt
import numpy as np
from geometry_utils import *
from preproc_utils.dataprep import get_labels, read_bin_velodyne
from yaml import safe_load
import os

# BASE_DIR = 'D:/SemanticKITTI/dataset/sequences'
# cfg = safe_load(open('../config/semantic-kitti.yaml', 'r'))
#
# velo_file = os.path.join(BASE_DIR, '00', 'velodyne', '000000.bin')
# label_file = os.path.join(BASE_DIR, '00', 'labels', '000000.label')
# pc_labels = get_labels(label_file, cfg)
# pc = read_bin_velodyne(velo_file)
#
# calib_file = os.path.join(BASE_DIR, '00', 'calib.txt')
# P0, P1, P2, P3, Tr = np.genfromtxt(calib_file, delimiter='')[:, 1:]
#
# P0 = np.reshape(P0, (3, 4))
# K = np.vstack((P0, np.array([0, 0, 0, 1])))
# Rot = np.reshape(Tr, (3, 4))
# Rot = np.vstack((Rot, np.array([0, 0, 0, 1])))
#
# print(K)
#
# # sample K
# # f_x = f_y = P0[0]
# # c_x, c_y = P0[2], P0[1, 2]
# # K = (f_x, f_y, c_x, c_y)
#
# img_file = os.path.join(BASE_DIR, '00', 'image_2', '000000.png')

# Load images
rgb = cv2.cvtColor(cv2.imread('rgb.png'), cv2.COLOR_BGR2RGB)

# Depth is stored as float32 in meters
depth = cv2.imread('depth.exr', cv2.IMREAD_ANYDEPTH)
print(depth)

# Get intrinsic parameters
height, width, _ = rgb.shape
K = intrinsic_from_fov(height, width, 90)  # +- 45 degrees
K_inv = np.linalg.inv(K)

# Get pixel coordinates
pixel_coords = pixel_coord_np(width, height)  # [3, npoints]

# Apply back-projection: K_inv @ pixels * depth
cam_coords = K_inv[:3, :3] @ pixel_coords * depth.flatten()

# Limit points to 150m in the z-direction for visualisation
cam_coords = cam_coords[:, np.where(cam_coords[2] <= 150)[0]]

# Visualize
pcd_cam = o3d.geometry.PointCloud()
pcd_cam.points = o3d.utility.Vector3dVector(cam_coords.T[:, :3])
# Flip it, otherwise the pointcloud will be upside down
pcd_cam.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
o3d.visualization.draw_geometries([pcd_cam])

def project_topview(cam_points):
    """
    Draw the topview projection
    """
    max_longitudinal = 70
    window_x = (-50, 50)
    window_y = (-3, max_longitudinal)

    x, y, z = cam_points
    # flip the y-axis to positive upwards
    y = - y

    # We sample points for points less than 70m ahead and above ground
    # Camera is mounted 1m above on an ego vehicle
    ind = np.where((z < max_longitudinal) & (y > -1.2))
    bird_eye = cam_points[:3, ind]

    # Color by radial distance
    dists = np.sqrt(np.sum(bird_eye[0:2:2, :] ** 2, axis=0))
    axes_limit = 10
    colors = np.minimum(1, dists / axes_limit / np.sqrt(2))

    # Draw Points
    fig, axes = plt.subplots(figsize=(12, 12))
    axes.scatter(bird_eye[0, :], bird_eye[2, :], c=colors, s=0.1)
    axes.set_xlim(window_x)
    axes.set_ylim(window_y)
    axes.set_title('Bird Eye View')
    plt.axis('off')

    plt.gca().set_aspect('equal')
    plt.show()

# Do top view projection
project_topview(cam_coords)