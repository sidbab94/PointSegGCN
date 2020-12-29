import os
os.environ['ETS_TOOLKIT'] = 'qt4'
import numpy as np
from numpy import random
import colorsys
from mayavi import mlab
import networkx as nx
import yaml
import open3d as o3d

os.chdir(os.getcwd())


def create_8bit_rgb_lut():
    """
    :return: Look-Up Table as 256**3 x 4 array
    """
    xl = np.mgrid[0:256, 0:256, 0:256]
    lut = np.vstack((xl[0].reshape(1, 256**3),
                        xl[1].reshape(1, 256**3),
                        xl[2].reshape(1, 256**3),
                        255 * np.ones((1, 256**3)))).T
    return lut.astype('int32')

def rgb_2_scalar_idx(r, g, b):
    """

    :param r: Red value
    :param g: Green value
    :param b: Blue value
    :return: scalar index of input colour
    """
    return 256**2 *r + 256 * g + b

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
        mlab.mesh(x, y, z, opacity=0.1, color=(0.1, 0.2, 0.3))

def show_voxel_wlabels(vox_pts, vox_lbls, graph, vis_scale):
    if vis_scale == 1:
        point_size = 0.04#0.2
        edge_size = 0.01
    else:
        point_size = 0.005
        edge_size = 0.001
    G = nx.convert_node_labels_to_integers(graph)

    rgb = rgb_map[vox_lbls]
    scalars = np.zeros((rgb.shape[0],))
    for (kp_idx, kp_c) in enumerate(rgb):
        scalars[kp_idx] = rgb_2_scalar_idx(kp_c[0], kp_c[1], kp_c[2])
    rgb_lut = create_8bit_rgb_lut()
    pts = mlab.points3d(
        vox_pts[:, 0],
        vox_pts[:, 1],
        vox_pts[:, 2],
        scalars,
        scale_factor=point_size,
        scale_mode="none",
        resolution=20)
    pts.mlab_source.dataset.lines = np.array(list(G.edges()))
    tube = mlab.pipeline.tube(pts, tube_radius=edge_size)
    mlab.pipeline.surface(tube, color=(0.8, 0.8, 0.8))

    pts.module_manager.scalar_lut_manager.lut._vtk_obj.SetTableRange(0, rgb_lut.shape[0])
    pts.module_manager.scalar_lut_manager.lut.number_of_colors = rgb_lut.shape[0]
    pts.module_manager.scalar_lut_manager.lut.table = rgb_lut

    pts.glyph.scale_mode = 'data_scaling_off'


def show_voxel(vox_pts, graph, vis_scale):
    if vis_scale == 1:
        point_size = 0.1#0.2
        edge_size = 0.02
    else:
        point_size = 0.005
        edge_size = 0.001

    G = nx.convert_node_labels_to_integers(graph)
    if vox_pts.shape[1] > 5:
        rgb = vox_pts[:, 3:]
        # print(rgb[:5, :])
        scalars = np.zeros((rgb.shape[0],))
        for (kp_idx, kp_c) in enumerate(rgb):
            scalars[kp_idx] = rgb_2_scalar_idx(kp_c[0], kp_c[1], kp_c[2])
        rgb_lut = create_8bit_rgb_lut()
        pts = mlab.points3d(
            vox_pts[:, 0],
            vox_pts[:, 1],
            vox_pts[:, 2],
            scalars,
            scale_factor=point_size,
            scale_mode="none",
            resolution=20)
    else:
        scalars = np.array(list(G.nodes())) + 5
        pts = mlab.points3d(
            vox_pts[:, 0],
            vox_pts[:, 1],
            vox_pts[:, 2],
            scalars,
            scale_factor=point_size,
            scale_mode="none",
            colormap="binary",
            resolution=20)

    pts.mlab_source.dataset.lines = np.array(list(G.edges()))
    tube = mlab.pipeline.tube(pts, tube_radius=edge_size)
    mlab.pipeline.surface(tube, color=(0.8, 0.8, 0.8))

    if vox_pts.shape[1] > 6:
        pts.module_manager.scalar_lut_manager.lut._vtk_obj.SetTableRange(0, rgb_lut.shape[0])
        pts.module_manager.scalar_lut_manager.lut.number_of_colors = rgb_lut.shape[0]
        pts.module_manager.scalar_lut_manager.lut.table = rgb_lut

    pts.glyph.scale_mode = 'data_scaling_off'

    xmin = np.min(vox_pts[:, 0])
    xmax = np.max(vox_pts[:, 0])
    ymin = np.min(vox_pts[:, 1])
    ymax = np.max(vox_pts[:, 1])
    zmin = np.min(vox_pts[:, 2])
    zmax = np.max(vox_pts[:, 2])
    pad = np.array(((xmax-xmin) * 0.01, (ymax-ymin) * 0.01, (zmax-zmin) * 0.01))
    mlab_plt_cube(xmin - pad[0], xmax + pad[0],
                  ymin - pad[1], ymax + pad[1],
                  zmin - pad[2], zmax - pad[2])


def cmap_cfg():
    semkitti = yaml.safe_load(open('./config/semantic-kitti.yaml', 'r'))
    bgr_map = np.array(list(semkitti['color_map'].values()))
    # rgb_map = np.empty_like(bgr_map)
    # rgb_map[:, 0] = bgr_map[:, 2]
    # rgb_map[:, 1] = bgr_map[:, 1]
    # rgb_map[:, 2] = bgr_map[:, 0]
    return bgr_map/255


class ShowPC:
    @staticmethod
    def random_colors(N, bright=True, seed=0):
        brightness = 1.0 if bright else 0.7
        hsv = [(0.15 + i / float(N), 1, brightness) for i in range(N)]
        colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
        random.seed(seed)
        random.shuffle(colors)
        return colors

    @staticmethod
    def draw_pc(pc_xyzrgb):
        pc = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(pc_xyzrgb[:, 0:3])
        if pc_xyzrgb.shape[1] == 3:
            o3d.visualization.draw_geometries([pc])
            return 0
        if np.max(pc_xyzrgb[:, 3:6]) > 20:  ## 0-255
            pc.colors = o3d.utility.Vector3dVector(pc_xyzrgb[:, 3:6] / 255.)
        else:
            pc.colors = o3d.utility.Vector3dVector(pc_xyzrgb[:, 3:6])
        o3d.visualization.draw_geometries([pc])
        return 0

    @staticmethod
    def draw_pc_sem_ins(pc_xyz, pc_sem_ins, plot_colors=cmap_cfg()):
        """
        pc_xyz: 3D coordinates of point clouds
        pc_sem_ins: semantic or instance labels
        plot_colors: custom color list
        """
        if plot_colors is not None:
            ins_colors = plot_colors
        else:
            ins_colors = ShowPC.random_colors(len(np.unique(pc_sem_ins)) + 1, seed=2)
        # ins_colors = cmap_cfg(pc_sem_ins)
        ##############################
        sem_ins_labels = np.unique(pc_sem_ins)
        sem_ins_bbox = []
        Y_colors = np.zeros((pc_sem_ins.shape[0], 3))
        for id, semins in enumerate(sem_ins_labels):
            valid_ind = np.argwhere(pc_sem_ins == semins)[:, 0]
            if semins <= -1:
                tp = [0, 0, 0]
            else:
                if plot_colors is not None:
                    tp = ins_colors[semins]
                else:
                    tp = ins_colors[id]

            Y_colors[valid_ind] = tp

            ### bbox
            valid_xyz = pc_xyz[valid_ind]

            xmin = np.min(valid_xyz[:, 0])
            xmax = np.max(valid_xyz[:, 0])
            ymin = np.min(valid_xyz[:, 1])
            ymax = np.max(valid_xyz[:, 1])
            zmin = np.min(valid_xyz[:, 2])
            zmax = np.max(valid_xyz[:, 2])
            sem_ins_bbox.append(
                [[xmin, ymin, zmin], [xmax, ymax, zmax], [min(tp[0], 1.), min(tp[1], 1.), min(tp[2], 1.)]])

        Y_semins = np.concatenate([pc_xyz[:, 0:3], Y_colors], axis=-1)
        ShowPC.draw_pc(Y_semins)
        return Y_semins


if __name__ == '__main__':
    from preproc_utils.dataprep import get_labels, read_bin_velodyne
    BASE_DIR = 'D:/SemanticKITTI/dataset/sequences'
    # x = np.genfromtxt('samples/testpc.csv', delimiter=',')
    # y = get_labels('../samples/testpc.label')
    pc = os.path.join(BASE_DIR, '08', 'velodyne', '000000.bin')
    x = read_bin_velodyne(pc)
    y = get_labels(os.path.join(BASE_DIR, '08', 'labels', '000000.label'))
    ShowPC.draw_pc_sem_ins(x, y)