import os
import numpy as np
import open3d as o3d
import networkx as nx
from mayavi import mlab
import matplotlib as mpl
from matplotlib import cm
import matplotlib.pyplot as plt

os.chdir(os.getcwd())

class MplColorHelper:

  def __init__(self, cmap_name, start_val, stop_val):
    self.cmap_name = cmap_name
    self.cmap = plt.get_cmap(cmap_name)
    self.norm = mpl.colors.Normalize(vmin=start_val, vmax=stop_val)
    self.scalarMap = cm.ScalarMappable(norm=self.norm, cmap=self.cmap)

  def get_rgb(self, val):
    return self.scalarMap.to_rgba(val)


def create_8bit_rgb_lut():
    """
    :return: Look-Up Table as 256**3 x 4 array
    """
    xl = np.mgrid[0:256, 0:256, 0:256]
    lut = np.vstack((xl[0].reshape(1, 256 ** 3),
                     xl[1].reshape(1, 256 ** 3),
                     xl[2].reshape(1, 256 ** 3),
                     255 * np.ones((1, 256 ** 3)))).T
    return lut.astype('int32')


def rgb_2_scalar_idx(r, g, b):
    """
    Converts an RGB array to a single scalar value
    :param r: Red value
    :param g: Green value
    :param b: Blue value
    :return: scalar index of input colour
    """
    return 256 ** 2 * r + 256 * g + b


def cube_faces(xmin, xmax, ymin, ymax, zmin, zmax):
    """
    Voxel generation helper function

    :param xmin: Min X axis bound
    :param xmax: Max X axis bound
    :param ymin: Min Y axis bound
    :param ymax: Max Y axis bound
    :param zmin: Min Z axis bound
    :param zmax: Max Z axis bound
    :return: Voxel faces to draw
    """
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
    """
    Draws voxel mesh on Mayavi window
    :param xmin: Min X axis bound
    :param xmax: Max X axis bound
    :param ymin: Min Y axis bound
    :param ymax: Max Y axis bound
    :param zmin: Min Z axis bound
    :param zmax: Max Z axis bound
    """
    faces = cube_faces(xmin, xmax, ymin, ymax, zmin, zmax)
    for grid in faces:
        x, y, z = grid
        mlab.mesh(x, y, z, opacity=0.1, color=(0.1, 0.2, 0.3))


class PC_Vis:
    """
    Visualization class using Open3D.
    Side-by-side comparative visualization, and individual PC-XYZRGB visualization possible.
    """

    @staticmethod
    def eval(pc, y_true, cfg, y_pred=None, gt_colour=False):

        if y_true is None:

            pred_pc_wlabels = PC_Vis.draw_pc_labels(pc, y_pred, cfg)
            pred_pc_wlabels_obj = PC_Vis.draw_pc(pred_pc_wlabels)

            vis2 = o3d.visualization.Visualizer()
            vis2.create_window(window_name='Predicted')
            vis2.add_geometry(pred_pc_wlabels_obj)

            while True:
                if not vis2.poll_events():
                    break
                vis2.update_renderer()

            vis2.destroy_window()

        else:

            orig_pc_wlabels = PC_Vis.draw_pc_labels(pc, y_true, cfg)
            if gt_colour:
                orig_pc_wlabels_obj = PC_Vis.draw_pc(pc)
            else:
                orig_pc_wlabels_obj = PC_Vis.draw_pc(orig_pc_wlabels)

            if y_pred is None:

                vis = o3d.visualization.Visualizer()
                vis.create_window(window_name='Ground Truth')
                vis.add_geometry(orig_pc_wlabels_obj)

                while True:
                    if not vis.poll_events():
                        break
                    vis.update_renderer()

                vis.destroy_window()

            else:

                pred_pc_wlabels = PC_Vis.draw_pc_labels(pc, y_pred, cfg)
                pred_pc_wlabels_obj = PC_Vis.draw_pc(pred_pc_wlabels)

                vis = o3d.visualization.Visualizer()
                vis.create_window(window_name='Ground Truth', width=960, height=985, left=0, top=40)
                vis.add_geometry(orig_pc_wlabels_obj)

                vis2 = o3d.visualization.Visualizer()
                vis2.create_window(window_name='Predicted', width=960, height=985, left=960, top=40)
                vis2.add_geometry(pred_pc_wlabels_obj)

                while True:
                    if not vis.poll_events():
                        break
                    vis.update_renderer()

                    if y_pred is not None:
                        if not vis2.poll_events():
                            break
                        vis2.update_renderer()

                vis.destroy_window()
                vis2.destroy_window()

    @staticmethod
    def draw_pc(pc_xyzrgbi, vis_test=False):
        pc = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(pc_xyzrgbi[:, 0:3])

        if pc_xyzrgbi.shape[1] > 4:
            if np.max(pc_xyzrgbi[:, -3:]) > 20:  ## 0-255
                pc.colors = o3d.utility.Vector3dVector(pc_xyzrgbi[:, -3:] / 255.)
            else:
                pc.colors = o3d.utility.Vector3dVector(pc_xyzrgbi[:, -3:])

        if vis_test:
            o3d.visualization.draw_geometries([pc])
            return None
        else:
            return pc

    @staticmethod
    def draw_pc_labels(pc_xyz, pc_sem_ins, cfg=None, vis_test=False):

        ins_colors = cfg['color_map']

        sem_ins_labels = np.unique(pc_sem_ins)
        sem_ins_bbox = []
        Y_colors = np.zeros((pc_sem_ins.shape[0], 3))
        for id, semins in enumerate(sem_ins_labels):
            valid_ind = np.argwhere(pc_sem_ins == semins)[:, 0]
            if semins <= -1:
                tp = [0, 0, 0]
            else:
                tp = ins_colors[semins]

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

        if vis_test:
            PC_Vis.draw_pc(Y_semins, True)

        return Y_semins

    @staticmethod
    def draw_pc_intensity(pc_xyzirgb):

        intensity = pc_xyzirgb[:, 3]
        pc = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(pc_xyzirgb[:, 0:3])

        i2rgb = np.interp(intensity, (intensity.min(), intensity.max()), (0.0, 255.0))
        i2rgb /= 255.
        COL = MplColorHelper('plasma', i2rgb.min(), i2rgb.max()).get_rgb(i2rgb)
        pc.colors = o3d.utility.Vector3dVector(COL[:, :3])

        o3d.visualization.draw_geometries([pc])

    @staticmethod
    def draw_graph(pc, graph):
        """
        Display point cloud with graph (nodes, edges)

        :param pc: point cloud array
        :param graph: sparse adjacency matrix
        :return:
        """
        point_size = 0.09  # 0.2
        edge_size = 0.025
        G = nx.from_scipy_sparse_matrix(graph)
        # G = nx.from_numpy_array(graph)
        # G = nx.convert_node_labels_to_integers(G)
        if pc.shape[1] > 5:
            rgb = pc[:, -3:] * 255
            scalars = np.zeros((rgb.shape[0],))
            for (kp_idx, kp_c) in enumerate(rgb):
                scalars[kp_idx] = rgb_2_scalar_idx(kp_c[0], kp_c[1], kp_c[2])
            rgb_lut = create_8bit_rgb_lut()
            pts = mlab.points3d(
                pc[:, 0],
                pc[:, 1],
                pc[:, 2],
                scalars,
                scale_factor=point_size,
                scale_mode="none",
                resolution=20)
        else:
            scalars = np.array(list(G.nodes())) + 5
            pts = mlab.points3d(
                pc[:, 0],
                pc[:, 1],
                pc[:, 2],
                scalars,
                scale_factor=point_size,
                scale_mode="none",
                colormap="binary",
                resolution=20)

        pts.mlab_source.dataset.lines = np.array(list(G.edges()))
        tube = mlab.pipeline.tube(pts, tube_radius=edge_size)
        mlab.pipeline.surface(tube, color=(0.8, 0.8, 0.8))

        if pc.shape[1] > 5:
            pts.module_manager.scalar_lut_manager.lut._vtk_obj.SetTableRange(0, rgb_lut.shape[0])
            pts.module_manager.scalar_lut_manager.lut.number_of_colors = rgb_lut.shape[0]
            pts.module_manager.scalar_lut_manager.lut.table = rgb_lut

        pts.glyph.scale_mode = 'data_scaling_off'

        mlab.show()
