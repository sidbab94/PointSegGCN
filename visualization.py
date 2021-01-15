import os
os.environ['ETS_TOOLKIT'] = 'qt4'
import numpy as np
from mayavi import mlab
import networkx as nx
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
    Converts an RGB array to a single scalar value
    :param r: Red value
    :param g: Green value
    :param b: Blue value
    :return: scalar index of input colour
    """
    return 256**2 *r + 256 * g + b

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

def show_voxel(vox_pts, graph, show_voxel=True):
    """
    Display voxel containing graph (nodes, edges)

    :param vox_pts: point cloud array
    :param graph: sparse adjacency matrix
    :return:
    """
    point_size = 0.04#0.2
    edge_size = 0.01
    G = nx.from_scipy_sparse_matrix(graph)
    G = nx.convert_node_labels_to_integers(G)
    if vox_pts.shape[1] > 5:
        rgb = vox_pts[:, 3:]
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

    if show_voxel:
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



class PC_Vis:
    """
    Visualization class using Open3D.
    Side-by-side comparative visualization, and individual PC-XYZRGB visualization possible.
    """

    @staticmethod
    def eval(pc, y_true, cfg, y_pred=None):

        orig_pc_wlabels = PC_Vis.draw_pc_sem_ins(pc, y_true, cfg)
        orig_pc_wlabels_obj = PC_Vis.draw_pc(orig_pc_wlabels)

        if y_pred is not None:

            pred_pc_wlabels = PC_Vis.draw_pc_sem_ins(pc, y_pred, cfg)
            pred_pc_wlabels_obj = PC_Vis.draw_pc(pred_pc_wlabels)

            vis = o3d.visualization.Visualizer()
            vis.create_window(window_name='Ground Truth', width=960, height=985, left=0, top=40)
            vis.add_geometry(orig_pc_wlabels_obj)

            vis2 = o3d.visualization.Visualizer()
            vis2.create_window(window_name='Predicted', width=960, height=985, left=960, top=40)
            vis2.add_geometry(pred_pc_wlabels_obj)

        else:

            vis = o3d.visualization.Visualizer()
            vis.create_window(window_name='Ground Truth')
            vis.add_geometry(orig_pc_wlabels_obj)

        while True:
            if not vis.poll_events():
                break
            vis.update_renderer()

            if y_pred is not None:
                if not vis2.poll_events():
                    break
                vis2.update_renderer()

        vis.destroy_window()
        if y_pred is not None:
            vis2.destroy_window()


    @staticmethod
    def draw_pc(pc_xyzrgb, vis_test=False):
        pc = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(pc_xyzrgb[:, 0:3])
        if pc_xyzrgb.shape[1] == 3:
            return pc

        if np.max(pc_xyzrgb[:, 3:6]) > 20:  ## 0-255
            pc.colors = o3d.utility.Vector3dVector(pc_xyzrgb[:, 3:6] / 255.)
        else:
            pc.colors = o3d.utility.Vector3dVector(pc_xyzrgb[:, 3:6])

        if vis_test:
            o3d.visualization.draw_geometries([pc])
            return None
        else:
            return pc

    @staticmethod
    def draw_pc_sem_ins(pc_xyz, pc_sem_ins, cfg):

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

        return Y_semins



if __name__ == '__main__':
    from _redundant.dataprep import get_labels, read_bin_velodyne
    BASE_DIR = 'D:/SemanticKITTI/dataset/sequences'
    # x = np.genfromtxt('samples/testpc.csv', delimiter=',')
    # y = get_labels('../samples/testpc.label')
    pc = os.path.join(BASE_DIR, '08', 'velodyne', '000000.bin')
    x = read_bin_velodyne(pc)
    y = get_labels(os.path.join(BASE_DIR, '08', 'labels', '000000.label'))
    ShowPC.draw_pc_sem_ins(x, y)